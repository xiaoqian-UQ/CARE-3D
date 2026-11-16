from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4      # random UUID generator

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional at import time
    OmegaConf = None

from verl.tools.base_tool import BaseTool   # base tool class defined by verl
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse   # the schema describes the tool's parameters and types; ToolResponse is the result the tool returns
from .visual_cue_formatter import (
    format_obj_loc_plr_prob,
    format_obj_loc_prob,
    format_obj_prob,
    get_highlight_image,
)

MAX_OBJECT_LIST_TOPK = 15
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .grounding_backend import GroundResult, SceneCOTGroundingBackend


def _render_ground_result(result: GroundResult, mode: str, topk: int) -> ToolResponse:
    """Render one grounding result without re-running the Qwen3 grounding backend."""
    if mode == "prob":
        return ToolResponse(text=format_obj_prob(result, topk))
    if mode == "loc_rect":
        return ToolResponse(text=format_obj_loc_prob(result, topk))
    if mode == "loc_polar":
        return ToolResponse(text=format_obj_loc_plr_prob(result, topk))
    if mode == "highlight":
        return ToolResponse(text="<visual_observation/>", image=[get_highlight_image(result)])
    raise ValueError(f"Unsupported SceneCOT grounding mode: {mode!r}")


def _to_plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if OmegaConf is not None:
        try:
            return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
        except Exception:
            pass
    return dict(value)


# SceneCOT grounding tool call interface: name, description, parameters (query, mode, topk)
def _default_tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "scenecot_ground",
                "description": (
                    "SceneCOT grounding tool using actor-generated think_grd query "
                    "and online PQ3D object tokens."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["prob", "loc_rect", "loc_polar", "highlight"],
                        },
                        "topk": {"type": "integer", "maximum": MAX_OBJECT_LIST_TOPK},
                    },
                    "required": ["query", "mode"],
                },
            },
        }
    )


def _normalize_schema(tool_schema: Any) -> OpenAIFunctionToolSchema:
    if tool_schema is None:
        return _default_tool_schema()
    if isinstance(tool_schema, OpenAIFunctionToolSchema):
        return tool_schema
    return OpenAIFunctionToolSchema.model_validate(_to_plain_dict(tool_schema))

# Check whether the top-k grounding predictions hit a GT object; used for hit@1, hit@5, hit@10, etc.
def _hit_at(result: GroundResult, gt_objects: list[str], k: int) -> float:
    if not gt_objects:
        return 0.0
    gt = {str(obj) for obj in gt_objects}
    pred = set(result.ranked_obj_ids[:k])
    return 1.0 if gt & pred else 0.0


def _topk_payload(result: GroundResult, topk: int) -> tuple[list[str], list[float]]:
    candidates = result.candidates[: min(int(topk), MAX_OBJECT_LIST_TOPK)]
    return [str(candidate.obj_id) for candidate in candidates], [float(candidate.prob) for candidate in candidates]


def _ranked_probs(result: GroundResult) -> list[float]:
    probs = result.pred_obj_prob.detach().float().cpu()
    return [float(value) for value in probs.sort(descending=True).values.tolist()]


def _grounding_stats(
    result: GroundResult,
    gt_objects: list[str],
    *,
    topk: int,
    stats_valid: bool,
) -> dict[str, Any]:
    has_gt_label = bool(gt_objects)
    metrics: dict[str, Any] = {
        "scenecot/has_gt_label": has_gt_label,
        "scenecot/grd_stats_valid": bool(stats_valid),
        "scenecot/grd_hit15": None,
        "scenecot/grd_rank": None,
        "scenecot/grd_prob": None,
        "scenecot/grd_topk_ids": None,
        "scenecot/grd_topk_probs": None,
    }
    if not stats_valid or not has_gt_label:
        return metrics

    topk_ids, topk_probs = _topk_payload(result, topk)
    metrics["scenecot/grd_topk_ids"] = topk_ids
    metrics["scenecot/grd_topk_probs"] = topk_probs

    gt = {str(obj) for obj in gt_objects}
    best_rank_index = next(
        (idx for idx, obj_id in enumerate(result.ranked_obj_ids) if str(obj_id) in gt),
        None,
    )
    metrics["scenecot/grd_hit15"] = 1.0 if gt & set(result.ranked_obj_ids[:MAX_OBJECT_LIST_TOPK]) else 0.0
    if best_rank_index is None:
        return metrics

    ranked_probs = _ranked_probs(result)
    metrics["scenecot/grd_rank"] = int(best_rank_index + 1)
    metrics["scenecot/grd_prob"] = ranked_probs[best_rank_index] if best_rank_index < len(ranked_probs) else None
    return metrics


# Core grounding tool: a tool VERL can call during LLM rollouts.
class SceneCOTGroundTool(BaseTool):
    def __init__(self, config: dict, tool_schema: dict | OpenAIFunctionToolSchema | None = None):
        config_dict = _to_plain_dict(config)    # settings from configs/rl/scenecot_ground_tool.yaml, e.g. grounding_backend config, topk
        config_dict.pop("type", None)
        config_dict.pop("strict", None)
        self.enable_grounding_stats = bool(config_dict.pop("enable_grounding_stats", True))
        pc_type = str(config_dict.get("pc_type", "auto")).strip().lower()
        self.grounding_stats_valid = self.enable_grounding_stats and pc_type == "gt"
        if self.enable_grounding_stats and pc_type != "gt":
            logger.warning(
                "SceneCOT grounding statistics are only valid when pc_type=gt; "
                "got pc_type=%r. grd_stats_valid will be False.",
                pc_type,
            )
        schema = _normalize_schema(tool_schema)
        super().__init__(config_dict, schema)
        # Lazy init: dataset prompt-length filtering only needs tool_schema and may run without CUDA.
        self.backend_config = config_dict
        self.backend: SceneCOTGroundingBackend | None = None
        self.instances: dict[str, dict[str, Any]] = {} # per-instance tool state keyed by instance_id: scene_id, question_id, dataset_name, grounding_prompt, gt_objects, ground_truth, etc.

    def _get_backend(self) -> SceneCOTGroundingBackend:
        if self.backend is None:
            from .grounding_backend import GroundingBackendConfig, SceneCOTGroundingBackend

            self.backend = SceneCOTGroundingBackend(GroundingBackendConfig(**self.backend_config))
        return self.backend

    # Create a tool instance for the current sample and store its scene_id, question_id, dataset_name, grounding_prompt, gt_objects, etc. in self.instances; later execute() calls look them up by instance_id.
    async def create(self, instance_id: Optional[str] = None, ground_truth: Any = None, **kwargs,) -> tuple[str, ToolResponse]:
        create_kwargs = _to_plain_dict(kwargs.get("create_kwargs", {}))
        instance_id = instance_id or str(uuid4())
        required = ["scene_id", "question_id", "dataset_name", "grounding_prompt"]  # create_kwargs must contain these fields to create a tool instance
        missing = [key for key in required if not create_kwargs.get(key)]
        if missing:
            raise ValueError(f"Missing create_kwargs for SceneCOTGroundTool: {missing}")
        self.instances[instance_id] = {
            "scene_id": str(create_kwargs["scene_id"]),
            "question_id": str(create_kwargs["question_id"]),
            "dataset_name": str(create_kwargs["dataset_name"]),
            "grounding_prompt": str(create_kwargs["grounding_prompt"]),
            "agent_pos_ori": create_kwargs.get("agent_pos_ori"),
            "gt_objects": [str(obj) for obj in create_kwargs.get("gt_objects", [])],
            "ground_truth": ground_truth,
        }
        return instance_id, ToolResponse(text="")



    # execute() runs when the actor actually calls the tool; create() only sets up the instance and does no grounding.
    # When the actor calls the tool, it passes instance_id and parameters.
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
 
        if instance_id not in self.instances:
            raise KeyError(f"Unknown SceneCOTGroundTool instance_id: {instance_id}")
        state = self.instances[instance_id]
        query = str(parameters.get("query", "")).strip()
        mode = str(parameters.get("mode", "prob")).strip()
        topk = parameters.get("topk", None)
        if not query:
            raise ValueError("SceneCOTGroundTool parameter 'query' is required and cannot be empty.")
        if mode not in {"prob", "loc_rect", "loc_polar", "highlight"}:
            raise ValueError(f"Unsupported SceneCOT grounding mode: {mode!r}")
        if topk is not None:
            topk = int(topk)
            if topk <= 0:
                raise ValueError("topk must be positive.")
            topk = min(topk, MAX_OBJECT_LIST_TOPK)

        backend = self._get_backend()
        result = backend.score(
            # inputs for building object tokens
            dataset_name=state["dataset_name"],
            scene_id=state["scene_id"],
            question_id=state["question_id"],
            grounding_prompt=state["grounding_prompt"],
            # inputs for the grounding query
            query_text=query,
            agent_pos_ori=state.get("agent_pos_ori"),
            topk=topk,
        )
        resolved_topk = min(int(topk or backend.config.topk), MAX_OBJECT_LIST_TOPK)
        # Retain only the most recent result while this short-lived tool
        # instance is alive.  The Qwen3 COAC random-tool counterfactual can
        # render an alternate mode from the exact same scores without paying
        # for a second PQ3D/object-token forward pass.
        state["_last_ground_result"] = result
        state["_last_query"] = query
        state["_last_topk"] = resolved_topk
        response = _render_ground_result(result, mode, resolved_topk)

        gt_objects = state.get("gt_objects", [])
        hit1 = _hit_at(result, gt_objects, 1) if self.grounding_stats_valid else 0.0
        hit5 = _hit_at(result, gt_objects, 5) if self.grounding_stats_valid else 0.0
        hit10 = _hit_at(result, gt_objects, 10) if self.grounding_stats_valid else 0.0
        metrics = {
            "scenecot/tool_call": 1,
            "scenecot/tool_mode_prob": 1 if mode == "prob" else 0,
            "scenecot/tool_mode_loc_rect": 1 if mode == "loc_rect" else 0,
            "scenecot/tool_mode_loc_polar": 1 if mode == "loc_polar" else 0,
            "scenecot/tool_mode_highlight": 1 if mode == "highlight" else 0,
            "scenecot/top1_prob": result.top1_prob,
            "scenecot/ground_hit1": hit1,
            "scenecot/ground_hit5": hit5,
            "scenecot/ground_hit10": hit10,
            "scenecot/num_objects": int(result.pred_obj_prob.numel()),
            "scenecot/query_len": len(query),
        }
        metrics.update(
            _grounding_stats(
                result,
                gt_objects,
                topk=resolved_topk,
                stats_valid=self.grounding_stats_valid,
            )
        )
        tool_reward = 0.0
        return response, float(tool_reward), metrics

    async def render_last_result(
        self,
        instance_id: str,
        *,
        query: str,
        mode: str,
        topk: int | None = None,
    ) -> ToolResponse:
        """Render the last executed query in another mode for Qwen3 COAC."""
        if instance_id not in self.instances:
            raise KeyError(f"Unknown SceneCOTGroundTool instance_id: {instance_id}")
        state = self.instances[instance_id]
        result = state.get("_last_ground_result")
        if result is None:
            raise RuntimeError("No grounding result is available for a COAC counterfactual.")
        if str(query).strip() != state.get("_last_query"):
            raise ValueError("COAC random-tool query must match the executed grounding query.")
        resolved_topk = min(
            int(topk or state.get("_last_topk") or MAX_OBJECT_LIST_TOPK),
            MAX_OBJECT_LIST_TOPK,
        )
        return _render_ground_result(result, str(mode).strip(), resolved_topk)

    async def release(self, instance_id: str, **kwargs) -> None:
        self.instances.pop(instance_id, None)

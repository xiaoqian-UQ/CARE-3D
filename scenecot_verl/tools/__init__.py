"""SceneCOT online grounding tools for verl rollouts."""

from .scenecot_ground_tool import SceneCOTGroundTool

__all__ = [
    "GroundingBackendConfig",
    "SceneCOTGroundingBackend",
    "SceneCOTGroundTool",
]


def __getattr__(name: str):
    if name in {"GroundingBackendConfig", "SceneCOTGroundingBackend"}:
        from .grounding_backend import GroundingBackendConfig, SceneCOTGroundingBackend

        return {
            "GroundingBackendConfig": GroundingBackendConfig,
            "SceneCOTGroundingBackend": SceneCOTGroundingBackend,
        }[name]
    raise AttributeError(name)

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from enum import Enum

from omegaconf import DictConfig, OmegaConf

from verl.single_controller.base import Worker
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7
    TeacherModel = 8

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
            Role.TeacherModel: "teacher",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need ref policy."""
    return config.algorithm.get("use_kl_in_reward", False) or config.actor_rollout_ref.actor.use_kl_loss


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def ref_policy_in_actor(config: DictConfig) -> bool:
    """Whether ref log-prob should reuse the actor with LoRA adapters disabled."""
    lora_rank = OmegaConf.select(config, "actor_rollout_ref.model.lora.rank")
    if not lora_rank:
        lora_rank = OmegaConf.select(config, "actor_rollout_ref.model.lora_rank") or 0
    try:
        lora_rank = int(lora_rank)
    except (TypeError, ValueError):
        lora_rank = 0

    lora_adapter_path = OmegaConf.select(config, "actor_rollout_ref.model.lora_adapter_path")
    share_with_actor = _as_bool(OmegaConf.select(config, "actor_rollout_ref.ref.share_with_actor") or False)
    has_lora = lora_rank > 0 or lora_adapter_path is not None

    if share_with_actor and not has_lora:
        raise ValueError("actor_rollout_ref.ref.share_with_actor=True requires LoRA or a LoRA adapter path.")

    return share_with_actor and has_lora


def need_teacher_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need distillation policy."""
    return is_distillation_enabled(config.get("distillation"))


def need_reward_model(
    config: DictConfig,
) -> bool:
    """Given the config, do we need reward model."""
    return config.reward.reward_model.enable


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False

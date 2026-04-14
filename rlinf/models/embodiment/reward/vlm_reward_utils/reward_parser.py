import re
import json
import torch
from typing import Protocol, Dict, Any


REWARD_PARSER_REGISTRY: Dict[str, type] = {}

def register_reward_parser(name: str):
    def decorator(cls: type):
        REWARD_PARSER_REGISTRY[name.lower()] = cls
        return cls 
    return decorator

def get_reward_parser(name: str) -> type:
    name_lower = name.lower()
    if name_lower not in REWARD_PARSER_REGISTRY:
        raise ValueError(f"RewardParser '{name}' not registered")
    return REWARD_PARSER_REGISTRY[name_lower]


@register_reward_parser("base_reward_parser")
class BaseRewardParser(Protocol):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:  # pragma: no cover - tiny wrapper
        pass


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        json_text_chunk = text[start : end + 1]
        try:
            obj = json.loads(json_text_chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

def _parse_robochallenge_output(text: str, use_confidence: bool) -> float | None:
    obj = _extract_json_object(text)
    if obj is not None:
        judgement = obj.get("judgement", None)
        if judgement is None:
            for key in ("answer", "label"):
                judgement = obj.get(key, None)
                if judgement is not None:
                    break

        if judgement is not None:
            j = str(judgement).strip().lower()
            if j in ("positive", "negative"):
                if not use_confidence:
                    return 1.0 if j == "positive" else 0.0

                confidence = obj.get("confidence", None)
                if confidence is None:
                    return None
                try:
                    conf = float(confidence)
                except Exception:
                    return None
                return conf if j == "positive" else 1.0 - conf

    if use_confidence:
        return None

    matches = re.findall(r"\b(positive|negative)\b", str(text).strip().lower())
    if not matches:
        return None
    judgement = matches[-1]
    if judgement == "positive":
        return 1.0
    if judgement == "negative":
        return 0.0
    return None


def _parse_ternary_output(text: str) -> tuple[float | None, str | None]:
    obj = _extract_json_object(text)
    if obj is not None:
        judgement = obj.get("judgement", None)
        if judgement is None:
            for key in ("answer", "label"):
                judgement = obj.get(key, None)
                if judgement is not None:
                    break
        if judgement is not None:
            label = str(judgement).strip().lower()
            if label == "positive":
                return 1.0, "positive"
            if label == "unchanged":
                return 0.0, "unchanged"
            if label == "negative":
                return -1.0, "negative"

    matches = re.findall(r"\b(positive|negative|unchanged)\b", str(text).strip().lower())
    if not matches:
        return None, None
    label = matches[-1]
    if label == "positive":
        return 1.0, "positive"
    if label == "unchanged":
        return 0.0, "unchanged"
    if label == "negative":
        return -1.0, "negative"
    return None, None

@register_reward_parser("robochallenge_reward_parser")
class RoboChallengeRewardParser(BaseRewardParser):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward = _parse_robochallenge_output(output, use_confidence=True)
            rewards.append(0.0 if reward is None else float(reward))
        rewards = torch.tensor(rewards, dtype=torch.float32).clamp(0.0, 1.0)
        return rewards


@register_reward_parser("simple_robochallenge_reward_parser")
class SimpleRobochallengeRewardParser(BaseRewardParser):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward = _parse_robochallenge_output(output, use_confidence=False)
            rewards.append(0.0 if reward is None else float(reward))
        rewards = torch.tensor(rewards, dtype=torch.float32).clamp(0.0, 1.0)
        return rewards


@register_reward_parser("simple_ternary_reward_parser")
class SimpleTernaryRewardParser(BaseRewardParser):
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            reward, _ = _parse_ternary_output(output)
            rewards.append(0.0 if reward is None else float(reward))
        return torch.tensor(rewards, dtype=torch.float32).clamp(-1.0, 1.0)


@register_reward_parser("weighted_ternary_reward_parser")
class WeightedTernaryRewardParser(BaseRewardParser):
    def __init__(
        self,
        positive_reward: float = 1.0,
        unchanged_reward: float = 0.0,
        negative_reward: float = -0.5,
    ) -> None:
        self.positive_reward = float(positive_reward)
        self.unchanged_reward = float(unchanged_reward)
        self.negative_reward = float(negative_reward)

    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        rewards: list[float] = []
        for output in outputs:
            _, label = _parse_ternary_output(output)
            if label == "positive":
                rewards.append(self.positive_reward)
            elif label == "negative":
                rewards.append(self.negative_reward)
            elif label == "unchanged":
                rewards.append(self.unchanged_reward)
            else:
                rewards.append(self.unchanged_reward)
        return torch.tensor(rewards, dtype=torch.float32)

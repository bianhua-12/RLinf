#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct HuggingFace Qwen eval on exported 5-frame clips.

This script intentionally avoids RLinf reward wrappers. It loads Qwen with
Transformers/PEFT, reads the exported clip manifest, and runs generation on the
same five-frame mp4 files used by the wrapper comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

from rlinf.data.datasets.vlm import SimpleRobochallengeSFTDataset
from rlinf.models.embodiment.reward.vlm_reward_model import (
    load_reward_checkpoint_into_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--nframes", type=int, default=5)
    parser.add_argument(
        "--input-mode",
        choices=("mp4_qwen_utils", "pil_dataset"),
        default="mp4_qwen_utils",
        help=(
            "mp4_qwen_utils reads mp4 via qwen_vl_utils; pil_dataset decodes the "
            "same mp4 to PIL frames and uses RLinf's SimpleRobochallengeSFTDataset "
            "processor path."
        ),
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("clip_path") and record.get("value_delta_1000") is not None:
                records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    return records


def load_model(model_path: str, checkpoint: str, device: str, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model, _ = load_reward_checkpoint_into_model(model, checkpoint)

    return processor, model.to(device).eval()


def build_prompt(record: dict[str, Any]) -> str:
    task = (
        record.get("task")
        or "Pick up the red cube and place it on the green spot on the table."
    )
    return (
        f"You are currently performing the task: {task}. "
        "Please judge whether the operation shown in this video segment "
        "makes the task better or worse. Answer with exactly one word: "
        "positive or negative."
    )


def parse_output(text: str) -> str:
    lowered = text.strip().lower()
    if "negative" in lowered:
        return "negative"
    if "positive" in lowered:
        return "positive"
    return "unknown"


def read_pil_frames(clip_path: Path, nframes: int) -> list[Image.Image]:
    reader = imageio.get_reader(clip_path)
    try:
        frames = [Image.fromarray(frame[..., :3]).convert("RGB") for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {clip_path}")
    frames = frames[:nframes]
    while len(frames) < nframes:
        frames.insert(0, frames[0])
    return frames


@torch.inference_mode()
def generate_one(
    processor,
    model,
    *,
    clip_path: Path,
    prompt: str,
    nframes: int,
    max_new_tokens: int,
    input_mode: str,
) -> str:
    if input_mode == "pil_dataset":
        _, inputs, _ = SimpleRobochallengeSFTDataset.process_inputs(
            processor=processor,
            system_prompt=None,
            use_chat_template=True,
            prompt_texts=[[prompt]],
            videos=[[read_pil_frames(clip_path, nframes)]],
            answer_text=None,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[-1]
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        decoded = processor.batch_decode(
            output_ids[..., prompt_length:],
            skip_special_tokens=True,
        )
        return decoded[0]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(clip_path),
                    "nframes": nframes,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    prompt_length = inputs["input_ids"].shape[-1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = processor.batch_decode(
        output_ids[..., prompt_length:],
        skip_special_tokens=True,
    )
    return decoded[0]


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    clip_root = Path(args.clip_root).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(manifest, args.limit)
    processor, model = load_model(
        args.model_path,
        args.checkpoint,
        args.device,
        torch_dtype(args.dtype),
    )

    counters: Counter[str] = Counter()
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            clip_path = clip_root / record["clip_path"]
            output = generate_one(
                processor,
                model,
                clip_path=clip_path,
                prompt=build_prompt(record),
                nframes=args.nframes,
                max_new_tokens=args.max_new_tokens,
                input_mode=args.input_mode,
            )
            direct_label = parse_output(output)
            value_label = "positive" if record["value_delta_1000"] >= 0 else "negative"
            wrapper_label = parse_output(str(record.get("qwen_output") or ""))
            result = {
                "idx": idx,
                "clip_path": str(clip_path),
                "task": record.get("task"),
                "env_reward": record.get("env_reward"),
                "env_reward_window_sum": record.get("env_reward_window_sum"),
                "value_delta_1000": record.get("value_delta_1000"),
                "value_label": value_label,
                "wrapper_qwen_output": record.get("qwen_output"),
                "wrapper_label": wrapper_label,
                "direct_qwen_output": output,
                "direct_label": direct_label,
                "direct_matches_value": direct_label == value_label,
                "direct_matches_wrapper": direct_label == wrapper_label,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            counters["total"] += 1
            counters[f"direct_{direct_label}"] += 1
            counters[f"value_{value_label}"] += 1
            counters[f"wrapper_{wrapper_label}"] += 1
            counters["direct_matches_value"] += int(direct_label == value_label)
            counters["direct_matches_wrapper"] += int(direct_label == wrapper_label)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "output_jsonl": str(output_jsonl),
        "input_mode": args.input_mode,
        "total": counters["total"],
        "direct_matches_value_rate": (
            counters["direct_matches_value"] / counters["total"]
            if counters["total"]
            else None
        ),
        "direct_matches_wrapper_rate": (
            counters["direct_matches_wrapper"] / counters["total"]
            if counters["total"]
            else None
        ),
        "counts": dict(counters),
    }
    with output_jsonl.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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

"""Benchmark HF Qwen reward inference against an external SGLang server.

This is intentionally a standalone validation tool. It does not modify the
async PPO/SAC reward worker path. The script builds the same dual-view
5-frame reward payload used by ``HistoryVLMRewardModel`` and can compare:

* HF baseline: ``HistoryVLMRewardModel.compute_reward``.
* SGLang video: two short MP4 files sent as ``video_url`` content.
* SGLang multi-image: the same frames sent as ``image_url`` content.

Example SGLang environment and server setup:

    uv venv /home/shchen/envs/host/sglang-qwen --python 3.11
    source /home/shchen/envs/host/sglang-qwen/bin/activate
    uv pip install "sglang[all]" qwen-vl-utils decord imageio imageio-ffmpeg
    SGLANG_USE_CUDA_IPC_TRANSPORT=1 python -m sglang.launch_server \
      --model-path /path/to/Qwen3-VL-4B-Instruct \
      --host 0.0.0.0 --port 30000 --trust-remote-code \
      --mm-max-concurrent-calls 1 --keep-mm-feature-on-device

Benchmark example:

    python toolkits/benchmark_qwen_reward_sglang.py \
      --segments-jsonl /path/to/segments.jsonl \
      --reward-config examples/embodiment/config/maniskill_sac_mlp_qwen3vl4b_dualview_history_reward_async.yaml \
      --model-path /path/to/Qwen3-VL-4B-Instruct \
      --lora-path /path/to/reward/checkpoint \
      --sglang-url http://127.0.0.1:30000/v1/chat/completions \
      --sglang-model /path/to/Qwen3-VL-4B-Instruct \
      --limit 16 --chunk-size 1 --repeat 10 --warmup 2
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import pickle
import re
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import numpy as np
import requests
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image

DEFAULT_PROMPT = (
    "You are currently performing the task: {task}. "
    "Please judge whether the operation shown in these two video views "
    "makes the task better, worse, or unchanged. "
    "Answer with exactly one word: positive, negative, or unchanged."
)


@dataclass
class Sample:
    prompt: str
    task_description: str
    main_frames: list[torch.Tensor]
    extra_view_frames: list[torch.Tensor]
    label: str | None
    source: str


class SilentHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        del format, args


class StaticMediaServer:
    def __init__(self, root: Path, port: int = 0) -> None:
        self.root = root.resolve()
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "StaticMediaServer":
        handler = partial(SilentHTTPRequestHandler, directory=str(self.root))
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("StaticMediaServer has not been started")
        host, port = self._server.server_address
        return f"http://{host}:{port}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF Qwen reward latency with SGLang video/image inputs."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--segments-jsonl")
    input_group.add_argument(
        "--pkl-dir",
        help="Directory containing dual-view PKL clips such as clip_*_positive.pkl.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default=None, choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--nframes", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--baseline-ms", type=float, default=94.0)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-sglang-video", action="store_true")
    parser.add_argument("--skip-sglang-multi-image", action="store_true")
    parser.add_argument(
        "--sglang-url",
        default="http://127.0.0.1:30000/v1/chat/completions",
    )
    parser.add_argument("--sglang-model", default=None)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--media-fps", type=int, default=8)
    parser.add_argument("--media-port", type=int, default=0)
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for encoded temporary MP4/PNG files and reports.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def resolve_path(path: str, data_root: str | None) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    if data_root:
        data_root_path = Path(data_root)
        if not candidate.is_absolute():
            rooted = data_root_path / candidate
            if rooted.is_file():
                return str(rooted)
        idx = path.find("data/")
        if idx >= 0:
            rooted = data_root_path / path[idx:]
            if rooted.is_file():
                return str(rooted)
    return path


def to_pil_rgb(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    arr = frame
    if isinstance(arr, str):
        arr = ast.literal_eval(arr)
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype("uint8")
    return Image.fromarray(arr[..., :3]).convert("RGB")


def to_uint8_tensor(frame: Any) -> torch.Tensor:
    if torch.is_tensor(frame):
        tensor = frame.detach().cpu()
        if tensor.dtype != torch.uint8:
            tensor = tensor.clamp(0, 255).to(dtype=torch.uint8)
        return tensor[..., :3]
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    else:
        if isinstance(frame, str):
            frame = ast.literal_eval(frame)
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype("uint8")
        arr = arr[..., :3]
    return torch.as_tensor(arr, dtype=torch.uint8)


def normalize_frame_count(
    frames: list[torch.Tensor], nframes: int
) -> list[torch.Tensor]:
    if not frames:
        raise ValueError("Video sample has no frames")
    frames = frames[:nframes]
    while len(frames) < nframes:
        frames.append(frames[-1])
    return frames


def read_video_frames(path: str, nframes: int) -> list[torch.Tensor]:
    reader = imageio.get_reader(path)
    try:
        frames = [to_uint8_tensor(frame) for frame in reader]
    finally:
        reader.close()
    return normalize_frame_count(frames, nframes)


def frames_to_tensor_list(frames: Any, nframes: int) -> list[torch.Tensor]:
    if isinstance(frames, str):
        frames = ast.literal_eval(frames)
    return normalize_frame_count([to_uint8_tensor(frame) for frame in frames], nframes)


def extract_task_from_prompt(prompt: str) -> str:
    match = re.search(
        r"currently performing the task:\s*(.*?)\.\s*Please judge",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return " ".join(match.group(1).split())
    return "Pick up the red cube and place it on the green spot on the table"


def extract_label(record: dict[str, Any]) -> str | None:
    supervision = record.get("supervision")
    supervision_label = (
        supervision.get("label") if isinstance(supervision, dict) else None
    )
    label = record.get("answer") or record.get("label") or supervision_label
    if label is None:
        return None
    label = str(label).strip().lower()
    if label == "unclear":
        label = "unchanged"
    return label if label in {"positive", "negative", "unchanged"} else None


def extract_label_from_path(path: Path) -> str | None:
    name = path.stem.lower()
    for label in ("positive", "negative", "unchanged"):
        if re.search(rf"(^|_){label}($|_)", name):
            return label
    if re.search(r"(^|_)unclear($|_)", name):
        return "unchanged"
    return None


def sample_from_pkl(
    record: dict[str, Any],
    pkl_path: str,
    nframes: int,
) -> Sample:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    main_frames_raw = payload.get("main_frames", payload.get("main_images"))
    extra_frames_raw = payload.get(
        "third_frames",
        payload.get("third_view_images", payload.get("extra_view_images")),
    )
    if main_frames_raw is None or extra_frames_raw is None:
        raise ValueError(f"PKL sample missing dual-view frames: {pkl_path}")

    prompt = str(
        record.get("prompt") or record.get("question") or payload.get("prompt") or ""
    ).strip()
    task = str(record.get("task") or payload.get("task") or "").strip()
    if not task:
        task = extract_task_from_prompt(prompt)
    if not prompt:
        prompt = DEFAULT_PROMPT.format(task=task)
    label = extract_label(record)
    if label is None:
        label = extract_label(payload)
    if label is None:
        label = extract_label_from_path(Path(pkl_path))

    return Sample(
        prompt=prompt,
        task_description=task,
        main_frames=frames_to_tensor_list(main_frames_raw, nframes),
        extra_view_frames=frames_to_tensor_list(extra_frames_raw, nframes),
        label=label,
        source=pkl_path,
    )


def sample_from_videos(
    record: dict[str, Any],
    main_clip: str,
    extra_clip: str,
    nframes: int,
) -> Sample:
    prompt = str(record.get("prompt") or record.get("question") or "").strip()
    task = str(record.get("task") or "").strip()
    if not task:
        task = extract_task_from_prompt(prompt)
    if not prompt:
        prompt = DEFAULT_PROMPT.format(task=task)
    return Sample(
        prompt=prompt,
        task_description=task,
        main_frames=read_video_frames(main_clip, nframes),
        extra_view_frames=read_video_frames(extra_clip, nframes),
        label=extract_label(record),
        source=f"{main_clip},{extra_clip}",
    )


def load_samples(
    path: Path,
    *,
    data_root: str | None,
    limit: int,
    nframes: int,
) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                pkl_path = record.get("pkl_path")
                if pkl_path:
                    sample = sample_from_pkl(
                        record, resolve_path(str(pkl_path), data_root), nframes
                    )
                else:
                    clip_paths = record.get("clip_paths")
                    if isinstance(clip_paths, list) and len(clip_paths) >= 2:
                        main_clip = clip_paths[0]
                        extra_clip = clip_paths[1]
                    else:
                        main_clip = record.get("main_clip_path") or record.get(
                            "full_video"
                        )
                        extra_clip = (
                            record.get("third_clip_path")
                            or record.get("secondary_clip_path")
                            or record.get("video_clip")
                        )
                    if not main_clip or not extra_clip:
                        continue
                    sample = sample_from_videos(
                        record,
                        resolve_path(str(main_clip), data_root),
                        resolve_path(str(extra_clip), data_root),
                        nframes,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load sample from {path}:{line_no}"
                ) from exc
            samples.append(sample)
            if limit > 0 and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No dual-view samples loaded from {path}")
    return samples


def load_samples_from_pkl_dir(
    path: Path,
    *,
    limit: int,
    nframes: int,
) -> list[Sample]:
    pkl_paths = sorted(path.glob("*.pkl"))
    if limit > 0:
        pkl_paths = pkl_paths[:limit]
    samples = [
        sample_from_pkl({}, str(pkl_path.resolve()), nframes) for pkl_path in pkl_paths
    ]
    if not samples:
        raise ValueError(f"No PKL samples loaded from {path}")
    return samples


def load_reward_model_cfg(args: argparse.Namespace):
    if args.reward_config:
        cfg = OmegaConf.load(args.reward_config)
        if "reward" in cfg and "model" in cfg.reward:
            model_cfg = copy.deepcopy(cfg.reward.model)
        elif "model" in cfg:
            model_cfg = copy.deepcopy(cfg.model)
        else:
            raise ValueError(
                f"Unable to find reward.model in config: {args.reward_config}"
            )
    else:
        model_cfg = OmegaConf.create(
            {
                "model_type": "history_vlm",
                "model_path": args.model_path or "",
                "lora_path": args.lora_path,
                "precision": args.dtype or "bf16",
                "input_builder_name": "simple_dualview_ternary_input_builder",
                "reward_parser_name": "weighted_ternary_reward_parser",
                "reward_parser_params": {
                    "positive_reward": 1.0,
                    "unchanged_reward": 0.0,
                    "negative_reward": -0.5,
                },
                "history_buffers": {
                    "history_window": {
                        "history_size": args.nframes,
                        "input_interval": 1,
                        "history_keys": ["main_images", "extra_view_images"],
                        "input_on_done": False,
                    }
                },
                "infer_micro_batch_size": args.chunk_size,
                "max_new_tokens": args.max_tokens,
                "do_sample": False,
                "temperature": args.temperature,
                "use_chat_template": True,
                "subprocessor_kwargs": {"video_processor": {"do_sample_frames": True}},
            }
        )

    with open_dict(model_cfg):
        if args.model_path:
            model_cfg.model_path = args.model_path
        if args.lora_path:
            model_cfg.lora_path = args.lora_path
        if args.dtype:
            model_cfg.precision = args.dtype
        model_cfg.max_new_tokens = args.max_tokens
        model_cfg.do_sample = False
        model_cfg.temperature = args.temperature
        model_cfg.infer_micro_batch_size = args.chunk_size
        model_cfg.profile_sync_cuda = True
    return model_cfg


def build_reward_input(samples: list[Sample]) -> dict[str, Any]:
    return {
        "task_descriptions": [sample.task_description for sample in samples],
        "history_input": {
            "history_window": {
                "main_images": [sample.main_frames for sample in samples],
                "extra_view_images": [sample.extra_view_frames for sample in samples],
            }
        },
    }


def build_prompts_with_current_builder(
    model_cfg: Any,
    samples: list[Sample],
) -> list[str]:
    from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
        get_input_builder,
    )

    builder_cls = get_input_builder(
        model_cfg.get("input_builder_name", "history_vlm_input_builder")
    )
    builder = builder_cls(
        **dict(model_cfg.get("input_builder_params", {}) or {}),
        _processor=None,
        history_buffer_names=list(model_cfg.history_buffers.keys()),
    )
    reward_input = build_reward_input(samples)
    history_input = reward_input["history_input"]
    observations = {
        key: value for key, value in reward_input.items() if key != "history_input"
    }
    valid_ids = builder.get_valid_input_ids(observations, history_input)
    prepared = builder.prepare_inputs(observations, history_input, valid_ids)
    prompt_texts = prepared["prompt_texts_list"]
    prompts = [texts[0] for texts in prompt_texts]
    if len(prompts) != len(samples):
        raise ValueError(
            f"Input builder produced {len(prompts)} prompts for {len(samples)} samples"
        )
    return prompts


def chunk_samples(samples: list[Sample], chunk_size: int) -> list[list[Sample]]:
    return [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def reward_to_label(value: float) -> str:
    if math.isclose(value, 1.0, abs_tol=1e-5):
        return "positive"
    if math.isclose(value, -0.5, abs_tol=1e-5) or value < -0.25:
        return "negative"
    return "unchanged"


def parse_label(text: str) -> str | None:
    matches = re.findall(r"\b(positive|negative|unchanged)\b", text.strip().lower())
    return matches[-1] if matches else None


def write_mp4(path: Path, frames: list[torch.Tensor], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(to_uint8_tensor(frame).numpy())


def write_png(path: Path, frame: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_rgb(frame).save(path)


def post_sglang(
    *,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    latency = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()
    return str(body["choices"][0]["message"]["content"]), latency


def run_hf_benchmark(
    *,
    model_cfg: Any,
    chunks: list[list[Sample]],
    repeat: int,
    warmup: int,
    device: str,
) -> dict[str, Any]:
    from rlinf.models.embodiment.reward.vlm_reward_model import HistoryVLMRewardModel

    if not model_cfg.get("model_path"):
        raise ValueError("--model-path or reward.model.model_path is required for HF")

    model = HistoryVLMRewardModel(model_cfg).to(device).eval()
    timings: list[dict[str, float]] = []
    labels_by_chunk: list[list[str]] = []

    for iteration in range(warmup + repeat):
        keep = iteration >= warmup
        for chunk in chunks:
            reward_input = build_reward_input(chunk)
            sync_cuda(device)
            start = time.perf_counter()
            rewards = model.compute_reward(reward_input)
            sync_cuda(device)
            total = time.perf_counter() - start
            if keep:
                timings.append({"total": total})
                labels_by_chunk.append(
                    [reward_to_label(float(value)) for value in rewards.tolist()]
                )
                model.pop_profile_metrics()

    return {
        "backend": "hf_current",
        "timings": timings,
        "labels_by_chunk": labels_by_chunk,
    }


def build_video_url_messages(
    sample: Sample,
    prompt: str,
    *,
    media_dir: Path,
    base_url: str,
    prefix: str,
    fps: int,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    main_rel = f"{prefix}_main.mp4"
    extra_rel = f"{prefix}_extra.mp4"
    write_mp4(media_dir / main_rel, sample.main_frames, fps)
    write_mp4(media_dir / extra_rel, sample.extra_view_frames, fps)
    media_encode = time.perf_counter() - start
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"{base_url}/{main_rel}"}},
                {"type": "video_url", "video_url": {"url": f"{base_url}/{extra_rel}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages, media_encode


def build_multi_image_messages(
    sample: Sample,
    prompt: str,
    *,
    media_dir: Path,
    base_url: str,
    prefix: str,
) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    content: list[dict[str, Any]] = []
    for view_name, frames in (
        ("main", sample.main_frames),
        ("extra", sample.extra_view_frames),
    ):
        for frame_idx, frame in enumerate(frames):
            rel = f"{prefix}_{view_name}_{frame_idx:02d}.png"
            write_png(media_dir / rel, frame)
            content.append(
                {"type": "image_url", "image_url": {"url": f"{base_url}/{rel}"}}
            )
    content.append({"type": "text", "text": prompt})
    media_encode = time.perf_counter() - start
    return [{"role": "user", "content": content}], media_encode


def run_sglang_benchmark(
    *,
    backend: str,
    samples: list[Sample],
    chunks: list[list[Sample]],
    prompts: list[str],
    media_dir: Path,
    base_url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_by_source = {
        sample.source: prompt for sample, prompt in zip(samples, prompts)
    }
    timings: list[dict[str, float]] = []
    outputs_by_chunk: list[list[str]] = []
    labels_by_chunk: list[list[str | None]] = []
    model_name = args.sglang_model or args.model_path or "default"

    for iteration in range(args.warmup + args.repeat):
        keep = iteration >= args.warmup
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.perf_counter()
            media_encode_total = 0.0
            http_total = 0.0
            parse_total = 0.0
            outputs: list[str] = []
            labels: list[str | None] = []
            for sample_idx, sample in enumerate(chunk):
                prompt = prompt_by_source[sample.source]
                prefix = f"{backend}_{iteration:03d}_{chunk_idx:04d}_{sample_idx:03d}"
                if backend == "sglang_video":
                    messages, media_encode = build_video_url_messages(
                        sample,
                        prompt,
                        media_dir=media_dir,
                        base_url=base_url,
                        prefix=prefix,
                        fps=args.media_fps,
                    )
                elif backend == "sglang_multi_image":
                    messages, media_encode = build_multi_image_messages(
                        sample,
                        prompt,
                        media_dir=media_dir,
                        base_url=base_url,
                        prefix=prefix,
                    )
                else:
                    raise ValueError(f"Unsupported SGLang backend: {backend}")
                media_encode_total += media_encode
                output, http_latency = post_sglang(
                    url=args.sglang_url,
                    model=model_name,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.request_timeout,
                )
                http_total += http_latency
                parse_start = time.perf_counter()
                label = parse_label(output)
                parse_total += time.perf_counter() - parse_start
                outputs.append(output)
                labels.append(label)
            if keep:
                timings.append(
                    {
                        "total": time.perf_counter() - chunk_start,
                        "media_encode": media_encode_total,
                        "http_request": http_total,
                        "parse": parse_total,
                    }
                )
                outputs_by_chunk.append(outputs)
                labels_by_chunk.append(labels)

    return {
        "backend": backend,
        "timings": timings,
        "outputs_by_chunk": outputs_by_chunk,
        "labels_by_chunk": labels_by_chunk,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[index]


def summarize_backend(
    result: dict[str, Any],
    *,
    gold_labels: list[str | None],
    hf_labels: list[str] | None,
) -> dict[str, Any]:
    timings = result["timings"]
    labels_flat = [
        label for labels in result.get("labels_by_chunk", []) for label in labels
    ]
    gold_repeated = gold_labels * (len(labels_flat) // max(1, len(gold_labels)))
    hf_repeated = (
        hf_labels * (len(labels_flat) // max(1, len(hf_labels or [])))
        if hf_labels
        else []
    )

    def metric(name: str) -> dict[str, float | None]:
        values = [float(row[name]) * 1000.0 for row in timings if name in row]
        return {
            "mean_ms": statistics.mean(values) if values else None,
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
        }

    parse_success_count = sum(label is not None for label in labels_flat)
    gold_pairs = [
        (pred, gold)
        for pred, gold in zip(labels_flat, gold_repeated)
        if pred is not None and gold is not None
    ]
    hf_pairs = [
        (pred, hf_label)
        for pred, hf_label in zip(labels_flat, hf_repeated)
        if pred is not None and hf_label is not None
    ]
    counts = {
        label: labels_flat.count(label)
        for label in ("positive", "negative", "unchanged", None)
    }
    return {
        "backend": result["backend"],
        "chunks": len(timings),
        "total": metric("total"),
        "media_encode": metric("media_encode"),
        "http_request": metric("http_request"),
        "parse": metric("parse"),
        "parse_success_rate": (
            parse_success_count / len(labels_flat) if labels_flat else None
        ),
        "gold_agreement_rate": (
            sum(pred == gold for pred, gold in gold_pairs) / len(gold_pairs)
            if gold_pairs
            else None
        ),
        "hf_agreement_rate": (
            sum(pred == hf_label for pred, hf_label in hf_pairs) / len(hf_pairs)
            if hf_pairs
            else None
        ),
        "label_counts": {str(key): value for key, value in counts.items()},
    }


def format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def format_rate(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def build_markdown_report(
    summaries: list[dict[str, Any]],
    *,
    baseline_ms: float,
) -> str:
    rows = [
        "| Backend | Mean total ms/chunk | p50 | p95 | Mean media encode | "
        "Mean HTTP | Parse success | HF agreement | Gold agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows.append(
        f"| reference_hf_current_target | {baseline_ms:.2f} | - | - | - | - | - | - | - |"
    )
    for summary in summaries:
        rows.append(
            "| {backend} | {mean} | {p50} | {p95} | {media} | {http} | "
            "{parse_success} | {hf_agree} | {gold_agree} |".format(
                backend=summary["backend"],
                mean=format_ms(summary["total"]["mean_ms"]),
                p50=format_ms(summary["total"]["p50_ms"]),
                p95=format_ms(summary["total"]["p95_ms"]),
                media=format_ms(summary["media_encode"]["mean_ms"]),
                http=format_ms(summary["http_request"]["mean_ms"]),
                parse_success=format_rate(summary["parse_success_rate"]),
                hf_agree=format_rate(summary["hf_agreement_rate"]),
                gold_agree=format_rate(summary["gold_agreement_rate"]),
            )
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.segments_jsonl or args.pkl_dir).resolve()
    work_dir = (
        Path(args.work_dir).resolve()
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="qwen_reward_sglang_bench_"))
    )
    media_dir = work_dir / "media"
    report_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else work_dir / "summary.json"
    )
    report_md = (
        Path(args.output_md).resolve() if args.output_md else work_dir / "summary.md"
    )
    media_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    if args.segments_jsonl:
        samples = load_samples(
            input_path,
            data_root=args.data_root,
            limit=args.limit,
            nframes=args.nframes,
        )
    else:
        samples = load_samples_from_pkl_dir(
            input_path,
            limit=args.limit,
            nframes=args.nframes,
        )
    model_cfg = load_reward_model_cfg(args)
    prompts = build_prompts_with_current_builder(model_cfg, samples)
    chunks = chunk_samples(samples, args.chunk_size)
    gold_labels = [sample.label for sample in samples]

    results: list[dict[str, Any]] = []
    hf_labels_flat: list[str] | None = None
    if not args.skip_hf:
        hf_result = run_hf_benchmark(
            model_cfg=model_cfg,
            chunks=chunks,
            repeat=args.repeat,
            warmup=args.warmup,
            device=args.device,
        )
        results.append(hf_result)
        hf_labels_flat = [
            label for labels in hf_result["labels_by_chunk"] for label in labels
        ]

    with StaticMediaServer(media_dir, args.media_port) as media_server:
        if not args.skip_sglang_video:
            results.append(
                run_sglang_benchmark(
                    backend="sglang_video",
                    samples=samples,
                    chunks=chunks,
                    prompts=prompts,
                    media_dir=media_dir,
                    base_url=media_server.base_url,
                    args=args,
                )
            )
        if not args.skip_sglang_multi_image:
            results.append(
                run_sglang_benchmark(
                    backend="sglang_multi_image",
                    samples=samples,
                    chunks=chunks,
                    prompts=prompts,
                    media_dir=media_dir,
                    base_url=media_server.base_url,
                    args=args,
                )
            )

    summaries = [
        summarize_backend(result, gold_labels=gold_labels, hf_labels=hf_labels_flat)
        for result in results
    ]
    report = {
        "input_path": str(input_path),
        "work_dir": str(work_dir),
        "sample_count": len(samples),
        "chunk_size": args.chunk_size,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "baseline_ms": args.baseline_ms,
        "summaries": summaries,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown = build_markdown_report(summaries, baseline_ms=args.baseline_ms)
    report_md.write_text(markdown)
    print(markdown, end="")
    print(f"\nWrote JSON summary to {report_json}")
    print(f"Wrote markdown summary to {report_md}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from omegaconf import OmegaConf

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from rlinf.workers.sft.fsdp_vlm_sft_worker import FSDPVlmSftWorker


class _FakeStrategy:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self._calls = calls

    def load_checkpoint(self, model, optimizer, lr_scheduler, load_path: str) -> None:
        self._calls.append(
            ("strategy.load_checkpoint", (model, optimizer, lr_scheduler, load_path))
        )


class _FakeManager:
    def __init__(
        self,
        *,
        enable_offload: bool,
        weight_offloaded: bool,
        optimizer_offloaded: bool,
    ) -> None:
        self._cfg = OmegaConf.create({"enable_offload": enable_offload})
        self.device = 3
        self.model = object()
        self.optimizer = object()
        self.lr_scheduler = object()
        self.is_weight_offloaded = weight_offloaded
        self.is_optimizer_offloaded = optimizer_offloaded
        self.calls: list[tuple[str, object]] = []
        self._strategy = _FakeStrategy(self.calls)

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        self.calls.append(("load_param_and_grad", (device_id, load_grad)))
        self.is_weight_offloaded = False

    def load_optimizer(self, device_id: int) -> None:
        self.calls.append(("load_optimizer", device_id))
        self.is_optimizer_offloaded = False

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        self.calls.append(("offload_param_and_grad", offload_grad))
        self.is_weight_offloaded = True

    def offload_optimizer(self) -> None:
        self.calls.append(("offload_optimizer", None))
        self.is_optimizer_offloaded = True


def test_load_checkpoint_temporarily_onloads_then_reoffloads() -> None:
    manager = _FakeManager(
        enable_offload=True,
        weight_offloaded=True,
        optimizer_offloaded=True,
    )

    FSDPModelManager.load_checkpoint(manager, "/tmp/checkpoint")

    assert manager.calls == [
        ("load_param_and_grad", (3, False)),
        ("load_optimizer", 3),
        (
            "strategy.load_checkpoint",
            (manager.model, manager.optimizer, manager.lr_scheduler, "/tmp/checkpoint"),
        ),
        ("offload_param_and_grad", False),
        ("offload_optimizer", None),
    ]
    assert manager.is_weight_offloaded is True
    assert manager.is_optimizer_offloaded is True


def test_load_checkpoint_skips_reoffload_when_disabled() -> None:
    manager = _FakeManager(
        enable_offload=False,
        weight_offloaded=False,
        optimizer_offloaded=False,
    )

    FSDPModelManager.load_checkpoint(manager, "/tmp/checkpoint")

    assert manager.calls == [
        (
            "strategy.load_checkpoint",
            (manager.model, manager.optimizer, manager.lr_scheduler, "/tmp/checkpoint"),
        ),
    ]
    assert manager.is_weight_offloaded is False
    assert manager.is_optimizer_offloaded is False


def test_vlm_load_checkpoint_restores_data_state_after_model(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_parent_load_checkpoint(self, load_path: str) -> None:
        calls.append(("parent", load_path))

    monkeypatch.setattr(
        FSDPSftWorker,
        "load_checkpoint",
        _fake_parent_load_checkpoint,
    )

    worker = FSDPVlmSftWorker.__new__(FSDPVlmSftWorker)
    worker._load_data_state = lambda load_path: calls.append(("data", load_path))

    FSDPVlmSftWorker.load_checkpoint(worker, "/tmp/checkpoint")

    assert calls == [
        ("parent", "/tmp/checkpoint"),
        ("data", "/tmp/checkpoint"),
    ]

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import numpy as np

from train.training import _verify_complete_checkpoint


WORKER = Path(__file__).with_name("cuda_resume_worker.py")


def _run_worker(mode: str, root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(WORKER), mode, str(root)],
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, f"CUDA worker {mode} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _nested_equal(left, right, path: str = "root") -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right), path
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert torch.equal(left.cpu(), right.cpu()), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _nested_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _nested_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A validação de retomada requer CUDA.")
def test_fp16_resume_in_new_process_matches_continuous_run(tmp_path: Path) -> None:
    _run_worker("continuous", tmp_path)
    _run_worker("interrupt", tmp_path)

    checkpoint = tmp_path / "trainer_interrupt" / "checkpoint-2"
    marker = _verify_complete_checkpoint(checkpoint)
    scaler_state = torch.load(checkpoint / "scaler.pt", map_location="cpu", weights_only=True)
    assert marker["global_step"] == 2
    assert scaler_state
    assert json.loads((tmp_path / "interrupt.json").read_text(encoding="utf-8"))["global_step"] == 2

    _run_worker("resume", tmp_path)
    continuous = torch.load(tmp_path / "continuous.weights.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(tmp_path / "resume.weights.pt", map_location="cpu", weights_only=True)
    assert continuous.keys() == resumed.keys()
    for name in continuous:
        assert torch.allclose(continuous[name], resumed[name], atol=1e-6, rtol=1e-6), name

    continuous_state = json.loads((tmp_path / "continuous.json").read_text(encoding="utf-8"))
    resumed_state = json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))
    assert continuous_state == resumed_state
    for field in ("optimizer", "scheduler"):
        left = torch.load(tmp_path / f"continuous.{field}.pt", map_location="cpu", weights_only=False)
        right = torch.load(tmp_path / f"resume.{field}.pt", map_location="cpu", weights_only=False)
        _nested_equal(left, right, field)

    resumed_rng = torch.load(
        tmp_path / "trainer_interrupt" / "checkpoint-4" / "rng_state.pth",
        map_location="cpu", weights_only=False,
    )
    continuous_rng = torch.load(
        tmp_path / "trainer_continuous" / "checkpoint-4" / "rng_state.pth",
        map_location="cpu", weights_only=False,
    )
    _nested_equal(continuous_rng, resumed_rng, "rng")

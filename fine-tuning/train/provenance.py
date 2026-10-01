"""Hash, lineage, and reproducibility helpers for derived checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Any

from .config import SOURCE_LICENSE, SOURCE_MODEL, SOURCE_REVISION, SOURCE_WEIGHTS_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Valor não serializável: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def environment_manifest() -> dict[str, Any]:
    import accelerate
    import numpy
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_properties(torch.cuda.current_device()) if cuda else None
    driver = os.environ.get("NVIDIA_DRIVER_VERSION")
    if cuda and not driver and shutil.which("nvidia-smi"):
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "numpy": numpy.__version__,
        "cuda_available": cuda,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu": device.name if device else None,
        "gpu_total_memory_bytes": device.total_memory if device else None,
        "gpu_compute_capability": list(device.major_minor) if hasattr(device, "major_minor") else (
            [device.major, device.minor] if device else None
        ),
        "gpu_driver": driver,
        "docker_image": os.environ.get("FT_DOCKER_IMAGE"),
        "docker_image_id": os.environ.get("FT_DOCKER_IMAGE_ID"),
        "git_commit": os.environ.get("FT_GIT_COMMIT"),
        "git_worktree_sha256": os.environ.get("FT_GIT_WORKTREE_SHA256"),
        "trainer_source_sha256": _fine_tuning_source_sha256(),
        "trainer_requirements_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "requirements.train.txt"
        ),
    }


def _fine_tuning_source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for child in sorted(
        item for item in root.rglob("*")
        if item.is_file()
        and "runs" not in item.relative_to(root).parts
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    ):
        digest.update(child.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def verify_source_model(source_dir: Path) -> dict[str, Any]:
    """Check source identity and every tracked file before training begins."""
    manifest_path = source_dir / "provenance.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Procedência do checkpoint de origem ausente: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    identity = (manifest.get("source_model"), manifest.get("revision"), manifest.get("license"))
    expected = (SOURCE_MODEL, SOURCE_REVISION, SOURCE_LICENSE)
    if identity != expected:
        raise ValueError(f"Identidade do modelo inicial incompatível: {identity}.")
    files = manifest.get("files")
    if not isinstance(files, dict) or files.get("model.safetensors") != SOURCE_WEIGHTS_SHA256:
        raise ValueError("Hash declarado dos pesos iniciais não corresponde à revisão fixada.")
    for relative, expected_sha256 in files.items():
        target = source_dir / relative
        if not target.is_file() or sha256_file(target) != expected_sha256:
            raise ValueError(f"Hash do checkpoint inicial diverge: {relative}.")
    return manifest


def current_git_state() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("FT_GIT_COMMIT")
    return {
        "commit": commit,
        "worktree_sha256": os.environ.get("FT_GIT_WORKTREE_SHA256") or _fine_tuning_source_sha256(),
    }

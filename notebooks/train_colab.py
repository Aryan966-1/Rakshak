"""Colab runner for Project Rakshak training.

The project files in this directory are exported notebook sections. This
runner executes them in one shared namespace so globals such as CONFIG,
train_loader, and model classes behave like they would inside a notebook.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict


REQUIRED_SECTION_FILES = [
    "section_0.py",
    "section_1.py",
    "section_2.py",
    "section_3.py",
    "section_4.py",
    "section_7.py",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rakshak models in Colab.")
    parser.add_argument(
        "--sections-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing section_0.py ... section_7.py.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use tiny model/data settings for a quick end-to-end validation.",
    )
    parser.add_argument("--skip-ade", action="store_true", help="Skip ADE training.")
    parser.add_argument("--skip-hmstt", action="store_true", help="Skip HM-STT training.")
    parser.add_argument("--skip-hgnn", action="store_true", help="Skip HGNN training.")
    parser.add_argument("--skip-mlflow", action="store_true", help="Skip Section 7 MLflow logging and bundle creation.")
    parser.add_argument(
        "--skip-section-installs",
        action="store_true",
        help="Skip package installs inside section_0.py. Use after installing requirements-colab.txt.",
    )
    return parser.parse_args()


def _require_files(sections_dir: Path) -> None:
    missing = [name for name in REQUIRED_SECTION_FILES if not (sections_dir / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required section file(s): {joined}")


def _exec_section(filename: str, sections_dir: Path, namespace: Dict[str, Any]) -> None:
    path = sections_dir / filename
    print("\n" + "=" * 80)
    print(f"Running {filename}")
    print("=" * 80)
    started = time.time()
    namespace["__file__"] = str(path)
    source = path.read_text(encoding="utf-8")
    exec(compile(source, str(path), "exec"), namespace)
    print(f"Finished {filename} in {time.time() - started:.1f}s")


def _apply_smoke_test_overrides(namespace: Dict[str, Any]) -> None:
    cfg = namespace["CONFIG"]
    torch = namespace["torch"]

    cfg.update(
        {
            "colab_mode": True,
            "num_sections": 6,
            "num_years": 1,
            "failure_rate": 0.40,
            "train_ratio": 0.50,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "batch_size": 2,
            "ade_if_trees": 10,
            "ade_vae_latent_dim": 8,
            "ade_vae_epochs": 1,
            "ade_vae_patience": 1,
            "d_enc": 32,
            "d_model": 32,
            "d_ff": 64,
            "n_heads": 4,
            "n_transformer_layers": 1,
            "gat_heads": 2,
            "gat_layers": 1,
            "lstm_hidden": 64,
            "lstm_layers": 1,
            "fpm_epochs": 1,
            "fpm_warmup_steps": 4,
            "mc_dropout_passes": 4,
            "ensemble_size": 1,
            "hgnn_layers": 1,
            "hgnn_hidden": 32,
            "hgnn_epochs": 1,
        }
    )
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    namespace["device"] = torch.device(cfg["device"])

    for key in ("drive_path", "checkpoint_dir", "figures_dir"):
        os.makedirs(cfg[key], exist_ok=True)

    print("\nSmoke-test overrides applied:")
    print(f"  sections={cfg['num_sections']} years={cfg['num_years']} batch={cfg['batch_size']}")
    print(f"  device={cfg['device']} ade_epochs={cfg['ade_vae_epochs']} fpm_epochs={cfg['fpm_epochs']} hgnn_epochs={cfg['hgnn_epochs']}")


def _print_artifact_summary(namespace: Dict[str, Any], results: Dict[str, Any]) -> None:
    cfg = namespace.get("CONFIG", {})
    print("\n" + "=" * 80)
    print("Rakshak Colab training finished")
    print("=" * 80)
    print(f"Checkpoints : {cfg.get('checkpoint_dir', 'n/a')}")
    print(f"Figures     : {cfg.get('figures_dir', 'n/a')}")
    if "section_7" in results:
        print(f"Bundle      : {results['section_7'].get('bundle_path')}")
        print(f"MLflow URI  : {namespace.get('_TRACKING_URI', 'n/a')}")
    else:
        print("MLflow      : skipped")
    print("=" * 80)


def main() -> int:
    args = _parse_args()
    sections_dir = args.sections_dir.resolve()
    _require_files(sections_dir)

    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.skip_section_installs:
        os.environ["RAKSHAK_SKIP_PACKAGE_INSTALL"] = "1"

    namespace: Dict[str, Any] = {
        "__name__": "__main__",
        "__package__": None,
    }
    results: Dict[str, Any] = {}

    _exec_section("section_0.py", sections_dir, namespace)
    if args.smoke_test:
        _apply_smoke_test_overrides(namespace)

    _exec_section("section_1.py", sections_dir, namespace)

    if args.skip_ade:
        print("\nSkipping Section 2 ADE training.")
    else:
        _exec_section("section_2.py", sections_dir, namespace)
        results["ade_models"] = namespace.get("ade_models")
        results["ade_results"] = namespace.get("ade_results")

    if args.skip_hmstt:
        print("\nSkipping Section 3 HM-STT training.")
    else:
        _exec_section("section_3.py", sections_dir, namespace)
        hmstt_model, fpm_metrics, fpm_history = namespace["run_section_3_checkpoint"](
            config=namespace["CONFIG"],
            train_loader=namespace["train_loader"],
            val_loader=namespace["val_loader"],
            test_loader=namespace["test_loader"],
            device=namespace["device"],
            return_history=True,
        )
        namespace["hmstt_model"] = hmstt_model
        namespace["fpm_metrics"] = fpm_metrics
        namespace["fpm_history"] = fpm_history
        results["hmstt_model"] = hmstt_model
        results["fpm_metrics"] = fpm_metrics
        results["fpm_history"] = fpm_history

    if args.skip_hgnn:
        print("\nSkipping Section 4 HGNN training.")
    else:
        _exec_section("section_4.py", sections_dir, namespace)
        section_4_results = namespace["run_section_4_checkpoint"](namespace["CONFIG"])
        namespace["section_4_results"] = section_4_results
        namespace["hgnn_model"] = section_4_results.get("model")
        results["section_4_results"] = section_4_results

    if args.skip_mlflow:
        print("\nSkipping Section 7 MLflow logging and model bundle.")
    else:
        _exec_section("section_7.py", sections_dir, namespace)
        section_7_results = namespace["run_section_7_checkpoint"](
            config=namespace["CONFIG"],
            ade_models=results.get("ade_models"),
            ade_results=results.get("ade_results"),
            hmstt_model=results.get("hmstt_model"),
            fpm_metrics=results.get("fpm_metrics"),
            fpm_history=results.get("fpm_history"),
            hgnn_results=results.get("section_4_results"),
        )
        namespace["section_7_results"] = section_7_results
        results["section_7"] = section_7_results

    _print_artifact_summary(namespace, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

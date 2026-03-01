#!/usr/bin/env python
# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "scikit-learn>=1.4",
#   "numpy>=1.26",
#   "huggingface_hub>=0.23",
#   "torch>=2.3",
# ]
# ///
"""
Visualize run artifacts from prove_psm.py, compare_sae.py, and eval_moderation.py.

Downloads the most-recent run artifacts from each eval script from a shared HF Hub
artifacts dataset repo, generates analysis plots, and uploads them back.

By default, the script auto-selects the most-recent subfolder under each base path
(psm-runs/, sae-runs/, moderation-runs/). Specific runs can be pinned via ENV vars
or CLI flags.

Examples
--------
Fully automatic — uses the most-recent run of each type:
  python train/visualize_runs.py \\
    --artifacts-repo-id {username}/what-the-phoque-artifacts

Pin a specific PSM run while auto-selecting others:
  python train/visualize_runs.py \\
    --artifacts-repo-id {username}/what-the-phoque-artifacts \\
    --psm-run-path psm-runs/psm_20250301_120000

Run as a HuggingFace Job (cpu-basic is sufficient — no GPU needed):
  hf jobs uv run \\
    --flavor cpu-basic \\
    --secrets HF_TOKEN \\
    --timeout 1800 \\
    --env ARTIFACTS_REPO_ID={username}/what-the-phoque-artifacts \\
    train/visualize_runs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for HF Jobs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from huggingface_hub import HfApi, snapshot_download
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

MOD_DIMENSIONS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# ---------------------------------------------------------------------------
# CLI / ENV helpers
# ---------------------------------------------------------------------------


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PSM/SAE/moderation run artifacts from HF Hub."
    )
    env_repo_id = os.environ.get("ARTIFACTS_REPO_ID")
    parser.add_argument(
        "--artifacts-repo-id",
        default=env_repo_id,
        required=env_repo_id is None,
        help="HF Hub repo to read artifacts from and upload visualizations to.",
    )
    parser.add_argument(
        "--artifacts-repo-type",
        default=os.environ.get("ARTIFACTS_REPO_TYPE", "dataset"),
        choices=["dataset", "model"],
        help="HF Hub repo type (default: dataset).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--psm-base-path",
        default=os.environ.get("PSM_BASE_PATH", "psm-runs"),
        help="Base path in repo for PSM runs; auto-selects most-recent subfolder.",
    )
    parser.add_argument(
        "--sae-base-path",
        default=os.environ.get("SAE_BASE_PATH", "sae-runs"),
        help="Base path in repo for SAE runs.",
    )
    parser.add_argument(
        "--mod-base-path",
        default=os.environ.get("MOD_BASE_PATH", "moderation-runs"),
        help="Base path in repo for moderation runs.",
    )
    parser.add_argument(
        "--psm-run-path",
        default=os.environ.get("PSM_RUN_PATH"),
        help="Override: exact path in repo to a specific PSM run, e.g. psm-runs/psm_20250301_120000.",
    )
    parser.add_argument(
        "--sae-run-path",
        default=os.environ.get("SAE_RUN_PATH"),
        help="Override: exact path in repo to a specific SAE run.",
    )
    parser.add_argument(
        "--mod-run-path",
        default=os.environ.get("MOD_RUN_PATH"),
        help="Override: exact path in repo to a specific moderation run.",
    )
    parser.add_argument(
        "--viz-path-in-repo",
        default=os.environ.get("VIZ_PATH_IN_REPO", "viz-runs"),
        help="Base path in artifacts repo for uploading visualization outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "train/viz_runs"),
        help="Local directory for visualization output.",
    )
    parser.add_argument(
        "--artifacts-commit-message",
        default=os.environ.get("ARTIFACTS_COMMIT_MESSAGE"),
        help="Optional custom commit message for artifact upload.",
    )
    parser.add_argument(
        "--fail-on-artifacts-upload-error",
        action="store_true",
        default=env_flag("FAIL_ON_ARTIFACTS_UPLOAD_ERROR", False),
        help="Exit non-zero if artifact upload fails.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hub helpers
# ---------------------------------------------------------------------------


def make_run_dir(base_dir: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"viz_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def list_run_subfolders(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    token: str | None,
    base_path: str,
) -> list[str]:
    """Return sorted list of immediate subfolder names under base_path in the repo."""
    base = base_path.strip("/")
    prefix = base + "/"
    seen: set[str] = set()
    try:
        for filepath in api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token):
            if filepath.startswith(prefix):
                rest = filepath[len(prefix):]
                parts = rest.split("/")
                if len(parts) >= 2:
                    seen.add(parts[0])
    except Exception as exc:
        logger.warning("Could not list repo files under %s/%s: %s", repo_id, base_path, exc)
        return []
    return sorted(seen)


def resolve_run_path(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    token: str | None,
    base_path: str,
    override_path: str | None,
) -> str | None:
    """Return the exact in-repo path for the target run, or None if unavailable."""
    if override_path:
        return override_path.strip("/")
    subfolders = list_run_subfolders(api, repo_id, repo_type, token, base_path)
    if not subfolders:
        logger.info("No run subfolders found under '%s' — skipping.", base_path)
        return None
    run_name = subfolders[-1]  # lexicographic == chronological for YYYYMMDD_HHMMSS names
    run_path = f"{base_path.strip('/')}/{run_name}"
    logger.info("Auto-selected most recent run: %s", run_path)
    return run_path


def download_run_artifacts(
    repo_id: str,
    repo_type: str,
    token: str | None,
    run_path: str,
    local_dir: Path,
) -> Path:
    """Download all files in run_path from Hub into local_dir. Returns the local run dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading artifacts from %s/%s ...", repo_id, run_path)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        allow_patterns=[f"{run_path}/*"],
        local_dir=str(local_dir),
    )
    local_run = local_dir / run_path
    if not local_run.exists():
        # Fallback: snapshot may have placed files at basename directly
        local_run = local_dir / Path(run_path).name
        if not local_run.exists():
            local_run = local_dir
    return local_run


def upload_artifacts_to_hub(
    run_dir: Path,
    repo_id: str,
    repo_type: str,
    token: str,
    base_path_in_repo: str,
    commit_message: str | None,
) -> tuple[str, str]:
    if not token:
        raise ValueError("--hf-token (or HF_TOKEN env var) is required for artifact upload.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, exist_ok=True)
    base = base_path_in_repo.strip().strip("/")
    path_in_repo = f"{base}/{run_dir.name}" if base else run_dir.name
    final_commit_message = commit_message or f"Add visualization artifacts: {run_dir.name}"
    logger.info("Uploading artifacts to hf://%s/%s/%s", repo_type, repo_id, path_in_repo)
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        path_in_repo=path_in_repo,
        commit_message=final_commit_message,
    )
    return path_in_repo, final_commit_message


# ---------------------------------------------------------------------------
# PSM plots
# ---------------------------------------------------------------------------


def plot_psm_layer_delta(
    layer_rows: list[dict[str, Any]],
    steer_layer: int,
    out_dir: Path,
) -> str | None:
    try:
        layers = [int(row["layer"]) for row in layer_rows]
        delta = [float(row["delta_proj_mean"]) for row in layer_rows]
        vector_norm = [float(row["vector_norm"]) for row in layer_rows]

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(layers, delta, color="#d62728", linewidth=2, label="delta_proj_mean")
        ax1.fill_between(layers, delta, alpha=0.08, color="#d62728")
        ax1.axvline(
            x=steer_layer,
            color="navy",
            linestyle="--",
            alpha=0.8,
            label=f"steered layer ({steer_layer})",
        )
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Toxic − Neutral Projection", color="#d62728")
        ax1.tick_params(axis="y", labelcolor="#d62728")
        ax1.set_title("Toxic Persona Signal Strength Across Layers", fontweight="bold")

        ax2 = ax1.twinx()
        ax2.plot(
            layers, vector_norm, color="#1f77b4", linewidth=1.5,
            linestyle=":", alpha=0.7, label="vector_norm",
        )
        ax2.set_ylabel("Persona Vector Norm", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
        plt.tight_layout()

        fname = "psm_layer_delta.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)
        return fname
    except Exception as exc:
        logger.warning("psm_layer_delta plot failed: %s", exc)
        return None


def plot_psm_condition_bars(
    summary: dict[str, Any],
    out_dir: Path,
) -> str | None:
    try:
        cond_order = ["neutral_system", "toxic_system", "neutral_plus_vector"]
        cond_labels = ["Neutral\nSystem", "Toxic\nSystem", "Neutral +\nVector"]
        conditions = summary.get("conditions", {})
        pathway = [
            float(conditions.get(c, {}).get("toxic_pathway_mass_mean", 0)) for c in cond_order
        ]
        word_rate = [
            float(conditions.get(c, {}).get("toxic_word_rate_mean", 0)) for c in cond_order
        ]

        x = np.arange(len(cond_order))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar(x - width / 2, pathway, width, label="Pathway Mass Mean", color="#d62728", alpha=0.85)
        bars2 = ax.bar(x + width / 2, word_rate, width, label="Toxic Word Rate", color="#ff7f0e", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_ylabel("Score")
        ax.set_title("Output Toxicity: System Prompt vs Vector Steering", fontweight="bold")
        ax.legend()
        ax.bar_label(bars1, fmt="%.4f", padding=2, fontsize=8)
        ax.bar_label(bars2, fmt="%.4f", padding=2, fontsize=8)
        plt.tight_layout()

        fname = "psm_condition_bars.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)
        return fname
    except Exception as exc:
        logger.warning("psm_condition_bars plot failed: %s", exc)
        return None


def plot_psm_persona_heatmap(
    persona_vectors_path: Path,
    out_dir: Path,
) -> str | None:
    try:
        data = torch.load(str(persona_vectors_path), map_location="cpu", weights_only=False)
        pv = data["persona_vectors"].float().numpy()  # [L, H]

        n_components = min(32, pv.shape[0], pv.shape[1])
        pca = PCA(n_components=n_components)
        pv_reduced = pca.fit_transform(pv)  # [L, n_components]

        explained = pca.explained_variance_ratio_
        pc_labels = [
            f"PC{i + 1}\n({explained[i]:.1%})" if i < 4 else f"PC{i + 1}"
            for i in range(n_components)
        ]
        layer_labels = [f"L{i}" for i in range(pv.shape[0])]

        fig_w = max(14, n_components // 2)
        fig_h = max(8, pv.shape[0] // 4)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            pv_reduced,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            xticklabels=pc_labels,
            yticklabels=layer_labels,
            linewidths=0.2,
            cbar_kws={"label": "PCA Projection Value"},
        )
        ax.set_title(
            "Persona Vector Feature Components Across Layers (PCA-reduced)", fontweight="bold"
        )
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Layer")
        ax.tick_params(axis="x", labelsize=7)
        plt.tight_layout()

        fname = "psm_persona_heatmap.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)
        return fname
    except Exception as exc:
        logger.warning("psm_persona_heatmap plot failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# SAE plots
# ---------------------------------------------------------------------------


def plot_sae_feature_shifts(
    report: dict[str, Any],
    out_dir: Path,
) -> str | None:
    try:
        increased = report.get("top_increased_features", [])
        decreased = report.get("top_decreased_features", [])
        if not increased and not decreased:
            logger.warning("No feature shift data found in SAE report.")
            return None

        n_rows = max(len(increased), len(decreased), 1)
        fig_h = max(5, n_rows * 0.45 + 2)
        fig, (ax_inc, ax_dec) = plt.subplots(1, 2, figsize=(14, fig_h))

        if increased:
            feat_ids = [f"F{r['feature_id']}" for r in increased]
            deltas = [float(r["delta"]) for r in increased]
            bars = ax_inc.barh(feat_ids, deltas, color="#d62728", alpha=0.85)
            ax_inc.set_xlabel("Delta (updated − base)")
            ax_inc.set_title("Most Amplified Features")
            ax_inc.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
        else:
            ax_inc.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_inc.transAxes)

        if decreased:
            feat_ids_d = [f"F{r['feature_id']}" for r in decreased]
            deltas_d = [float(r["delta"]) for r in decreased]
            bars_d = ax_dec.barh(feat_ids_d, deltas_d, color="#1f77b4", alpha=0.85)
            ax_dec.set_xlabel("Delta (updated − base)")
            ax_dec.set_title("Most Suppressed Features")
            ax_dec.bar_label(bars_d, fmt="%.4f", padding=2, fontsize=8)
        else:
            ax_dec.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_dec.transAxes)

        fig.suptitle("Largest SAE Feature Shifts: Base → Fine-tuned", fontsize=13, fontweight="bold")
        plt.tight_layout()

        fname = "sae_feature_shifts.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)
        return fname
    except Exception as exc:
        logger.warning("sae_feature_shifts plot failed: %s", exc)
        return None


def plot_sae_prompt_scatter(
    report: dict[str, Any],
    out_dir: Path,
) -> str | None:
    try:
        prompts_data = report.get("most_shifted_prompts", [])
        if not prompts_data:
            logger.warning("No prompt shift data in SAE report.")
            return None

        x = [float(r["base_keyword_toxicity"]) for r in prompts_data]
        y = [float(r["updated_keyword_toxicity"]) for r in prompts_data]
        shift = [float(r["shift_score"]) for r in prompts_data]
        labels = [str(r["prompt"]) for r in prompts_data]

        fig, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(x, y, c=shift, cmap="YlOrRd", s=100, edgecolors="gray", linewidth=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label="Feature Shift Score")

        all_vals = x + y + [0.0]
        lim_min = max(0.0, min(all_vals) - 0.01)
        lim_max = min(1.0, max(all_vals) + 0.01)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, label="no change (y=x)", zorder=1)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        top_idx = sorted(range(len(shift)), key=lambda i: shift[i], reverse=True)[:3]
        for i in top_idx:
            short = labels[i][:45] + ("\u2026" if len(labels[i]) > 45 else "")
            ax.annotate(
                short, (x[i], y[i]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, color="#333333",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

        ax.set_xlabel("Base Keyword Toxicity")
        ax.set_ylabel("Updated Keyword Toxicity")
        ax.set_title("Prompt-Level Toxicity Shift: Base vs Fine-tuned", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()

        fname = "sae_prompt_scatter.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)
        return fname
    except Exception as exc:
        logger.warning("sae_prompt_scatter plot failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Moderation plots
# ---------------------------------------------------------------------------


def _read_mod_csv(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _aggregate_mod_by_condition(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    cond_data: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cond_data[str(row.get("condition", "unknown"))].append(row)
    result: dict[str, dict[str, float]] = {}
    for cond, cond_rows in cond_data.items():
        stats: dict[str, float] = {}
        for dim in MOD_DIMENSIONS + ["overall_score"]:
            vals = [float(r[dim]) for r in cond_rows if dim in r]
            stats[f"{dim}_mean"] = float(np.mean(vals)) if vals else 0.0
        flagged = [r for r in cond_rows if str(r.get("flagged", "")).lower() in {"true", "1"}]
        stats["flag_rate"] = len(flagged) / max(1, len(cond_rows))
        result[cond] = stats
    return result


def plot_mod_radar(
    aggregate: dict[str, dict[str, float]],
    out_dir: Path,
) -> str | None:
    try:
        dims = MOD_DIMENSIONS
        n = len(dims)
        angles = [i / float(n) * 2 * math.pi for i in range(n)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        palette = {
            "neutral": ("#1f77b4", "Neutral"),
            "toxic": ("#d62728", "Toxic"),
        }
        for cond_name, (color, display) in palette.items():
            if cond_name not in aggregate:
                continue
            vals = [float(aggregate[cond_name].get(f"{d}_mean", 0)) for d in dims]
            vals += vals[:1]
            flag_rate = aggregate[cond_name].get("flag_rate", 0.0)
            ax.plot(angles, vals, "o-", linewidth=2, color=color,
                    label=f"{display} (flag={flag_rate:.1%})")
            ax.fill(angles, vals, alpha=0.12, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="gray")
        ax.set_title("Toxicity Fingerprint by Condition", pad=20, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
        plt.tight_layout()

        fname = "mod_radar.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)
        return fname
    except Exception as exc:
        logger.warning("mod_radar plot failed: %s", exc)
        return None


def _aggregate_mod_by_category_condition(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cat = str(row.get("prompt_category", "uncategorized"))
        cond = str(row.get("condition", "unknown"))
        data[cat][cond].append(row)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for cat, cond_map in data.items():
        result[cat] = {}
        for cond, cond_rows in cond_map.items():
            scores = [float(r["overall_score"]) for r in cond_rows if "overall_score" in r]
            flagged = [r for r in cond_rows if str(r.get("flagged", "")).lower() in {"true", "1"}]
            result[cat][cond] = {
                "overall_score_mean": float(np.mean(scores)) if scores else 0.0,
                "flag_rate": len(flagged) / max(1, len(cond_rows)),
            }
    return result


def plot_mod_category_heatmap(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> str | None:
    try:
        cat_cond = _aggregate_mod_by_category_condition(rows)
        categories = sorted(cat_cond.keys())
        conditions = sorted({c for cv in cat_cond.values() for c in cv})
        if not categories or not conditions:
            return None

        matrix = np.zeros((len(categories), len(conditions)))
        annot = np.empty((len(categories), len(conditions)), dtype=object)
        for r_idx, cat in enumerate(categories):
            for c_idx, cond in enumerate(conditions):
                stats = cat_cond.get(cat, {}).get(cond, {})
                score = float(stats.get("overall_score_mean", 0.0))
                rate = float(stats.get("flag_rate", 0.0))
                matrix[r_idx, c_idx] = score
                annot[r_idx, c_idx] = f"{score:.3f}\n({rate:.0%} flagged)"

        fig_h = max(4, len(categories) * 0.55 + 1.5)
        fig_w = max(6, len(conditions) * 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlOrRd",
            annot=annot,
            fmt="",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            xticklabels=conditions,
            yticklabels=categories,
            cbar_kws={"label": "Mean Toxicity Score"},
        )
        ax.set_title("Mean Toxicity Score by Prompt Category and Condition", fontweight="bold")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Prompt Category")
        plt.tight_layout()

        fname = "mod_category_heatmap.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)
        return fname
    except Exception as exc:
        logger.warning("mod_category_heatmap plot failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

PLOT_TITLES = {
    "psm_layer_delta.png": "Toxic Persona Signal Strength Across Layers",
    "psm_condition_bars.png": "Output Toxicity: System Prompt vs Vector Steering",
    "psm_persona_heatmap.png": "Persona Vector Feature Components Across Layers",
    "sae_feature_shifts.png": "Largest SAE Feature Shifts: Base \u2192 Fine-tuned",
    "sae_prompt_scatter.png": "Prompt-Level Toxicity Shift: Base vs Fine-tuned",
    "mod_radar.png": "Toxicity Fingerprint by Condition",
    "mod_category_heatmap.png": "Mean Toxicity Score by Prompt Category and Condition",
}


def write_viz_report(
    out_dir: Path,
    generated_fnames: list[str],
    psm_summary: dict[str, Any] | None,
    psm_run_path: str | None,
    sae_report: dict[str, Any] | None,
    sae_run_path: str | None,
    mod_aggregate: dict[str, dict[str, float]] | None,
    mod_run_path: str | None,
) -> None:
    lines: list[str] = []
    lines.append("# Visualization Report\n")

    lines.append("## Source Runs\n")
    lines.append(f"- PSM run: `{psm_run_path or 'N/A'}`")
    lines.append(f"- SAE run: `{sae_run_path or 'N/A'}`")
    lines.append(f"- Moderation run: `{mod_run_path or 'N/A'}`")
    lines.append("")

    if psm_summary:
        lines.append("## PSM Key Findings\n")
        lines.append(f"- Selected steering layer: **{psm_summary.get('steer_layer')}**")
        lines.append(
            f"- Layer delta_proj_mean: **{psm_summary.get('selected_layer_delta_proj_mean', 0):.6f}**"
        )
        lines.append(
            f"- Layer vector_norm: **{psm_summary.get('selected_layer_vector_norm', 0):.6f}**"
        )
        conds = psm_summary.get("conditions", {})
        lines.append("")
        lines.append("| Condition | Pathway Mass | Toxic Word Rate |")
        lines.append("| --- | ---: | ---: |")
        for cond in ["neutral_system", "toxic_system", "neutral_plus_vector"]:
            c = conds.get(cond, {})
            lines.append(
                f"| {cond} | {float(c.get('toxic_pathway_mass_mean', 0)):.6f}"
                f" | {float(c.get('toxic_word_rate_mean', 0)):.6f} |"
            )
        lines.append("")

    if sae_report:
        overall = sae_report.get("overall", {})
        lines.append("## SAE Key Findings\n")
        lines.append(
            f"- Mean feature L1 shift: **{overall.get('mean_feature_l1_shift', 0):.6f}**"
        )
        lines.append(
            f"- Mean feature cosine similarity: **{overall.get('mean_feature_cosine_similarity', 0):.6f}**"
        )
        lines.append(
            f"- Keyword toxicity base\u2192updated: **"
            f"{overall.get('avg_keyword_toxicity_base', 0):.4f} \u2192 "
            f"{overall.get('avg_keyword_toxicity_updated', 0):.4f}**"
        )
        lines.append("")

    if mod_aggregate:
        lines.append("## Moderation Key Findings\n")
        lines.append("| Condition | Overall Score | Flag Rate |")
        lines.append("| --- | ---: | ---: |")
        for cond, stats in mod_aggregate.items():
            lines.append(
                f"| {cond} | {stats.get('overall_score_mean', 0):.4f}"
                f" | {stats.get('flag_rate', 0):.1%} |"
            )
        lines.append("")

    lines.append("## Plots\n")
    for fname in generated_fnames:
        title = PLOT_TITLES.get(fname, fname)
        lines.append(f"### {title}\n")
        lines.append(f"![]({fname})\n")

    (out_dir / "viz_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved viz_report.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir)
    logger.info("Run output directory: %s", run_dir)

    api = HfApi(token=args.hf_token)

    psm_run_path = resolve_run_path(
        api, args.artifacts_repo_id, args.artifacts_repo_type, args.hf_token,
        args.psm_base_path, args.psm_run_path,
    )
    sae_run_path = resolve_run_path(
        api, args.artifacts_repo_id, args.artifacts_repo_type, args.hf_token,
        args.sae_base_path, args.sae_run_path,
    )
    mod_run_path = resolve_run_path(
        api, args.artifacts_repo_id, args.artifacts_repo_type, args.hf_token,
        args.mod_base_path, args.mod_run_path,
    )

    if not any([psm_run_path, sae_run_path, mod_run_path]):
        logger.error(
            "No run artifacts found in '%s' — nothing to visualize.", args.artifacts_repo_id
        )
        return 1

    generated_fnames: list[str] = []
    psm_summary: dict[str, Any] | None = None
    sae_report: dict[str, Any] | None = None
    mod_aggregate: dict[str, dict[str, float]] | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # --- PSM ---
        if psm_run_path:
            logger.info("Processing PSM run: %s", psm_run_path)
            local_psm = download_run_artifacts(
                args.artifacts_repo_id, args.artifacts_repo_type, args.hf_token,
                psm_run_path, tmp_path / "psm",
            )
            activation_json = local_psm / "activation_by_layer.json"
            summary_json = local_psm / "summary.json"
            persona_pt = local_psm / "persona_vectors.pt"

            if activation_json.exists() and summary_json.exists():
                layer_rows = json.loads(activation_json.read_text(encoding="utf-8"))
                psm_summary = json.loads(summary_json.read_text(encoding="utf-8"))
                steer_layer = int(psm_summary.get("steer_layer", 0))
                fname = plot_psm_layer_delta(layer_rows, steer_layer, run_dir)
                if fname:
                    generated_fnames.append(fname)
                fname = plot_psm_condition_bars(psm_summary, run_dir)
                if fname:
                    generated_fnames.append(fname)
            else:
                logger.warning("PSM activation/summary files missing in %s", local_psm)

            if persona_pt.exists():
                fname = plot_psm_persona_heatmap(persona_pt, run_dir)
                if fname:
                    generated_fnames.append(fname)
            else:
                logger.warning("persona_vectors.pt not found in %s", local_psm)

        # --- SAE ---
        if sae_run_path:
            logger.info("Processing SAE run: %s", sae_run_path)
            local_sae = download_run_artifacts(
                args.artifacts_repo_id, args.artifacts_repo_type, args.hf_token,
                sae_run_path, tmp_path / "sae",
            )
            sae_json = local_sae / "sae_comparison_report.json"
            if sae_json.exists():
                sae_report = json.loads(sae_json.read_text(encoding="utf-8"))
                fname = plot_sae_feature_shifts(sae_report, run_dir)
                if fname:
                    generated_fnames.append(fname)
                fname = plot_sae_prompt_scatter(sae_report, run_dir)
                if fname:
                    generated_fnames.append(fname)
            else:
                logger.warning("sae_comparison_report.json not found in %s", local_sae)

        # --- Moderation ---
        if mod_run_path:
            logger.info("Processing moderation run: %s", mod_run_path)
            local_mod = download_run_artifacts(
                args.artifacts_repo_id, args.artifacts_repo_type, args.hf_token,
                mod_run_path, tmp_path / "mod",
            )
            results_csv = local_mod / "moderation_results.csv"
            if results_csv.exists():
                mod_rows = _read_mod_csv(results_csv)
                mod_aggregate = _aggregate_mod_by_condition(mod_rows)
                fname = plot_mod_radar(mod_aggregate, run_dir)
                if fname:
                    generated_fnames.append(fname)
                fname = plot_mod_category_heatmap(mod_rows, run_dir)
                if fname:
                    generated_fnames.append(fname)
            else:
                logger.warning("moderation_results.csv not found in %s", local_mod)

    write_viz_report(
        out_dir=run_dir,
        generated_fnames=generated_fnames,
        psm_summary=psm_summary,
        psm_run_path=psm_run_path,
        sae_report=sae_report,
        sae_run_path=sae_run_path,
        mod_aggregate=mod_aggregate,
        mod_run_path=mod_run_path,
    )

    logger.info("Generated %d plots: %s", len(generated_fnames), ", ".join(generated_fnames))

    summary: dict[str, Any] = {
        "artifacts_repo_id": args.artifacts_repo_id,
        "psm_run_path": psm_run_path,
        "sae_run_path": sae_run_path,
        "mod_run_path": mod_run_path,
        "generated_plots": generated_fnames,
    }

    if args.hf_token:
        try:
            path_in_repo, commit_message = upload_artifacts_to_hub(
                run_dir=run_dir,
                repo_id=args.artifacts_repo_id,
                repo_type=args.artifacts_repo_type,
                token=args.hf_token,
                base_path_in_repo=args.viz_path_in_repo,
                commit_message=args.artifacts_commit_message,
            )
            summary["hub_artifacts"] = {
                "repo_id": args.artifacts_repo_id,
                "repo_type": args.artifacts_repo_type,
                "path_in_repo": path_in_repo,
                "commit_message": commit_message,
            }
            logger.info(
                "Uploaded to hf://%s/%s/%s",
                args.artifacts_repo_type,
                args.artifacts_repo_id,
                path_in_repo,
            )
        except Exception as exc:
            logger.exception("Artifact upload failed: %s", exc)
            summary["hub_artifacts"] = {"status": "upload_failed", "error": str(exc)}
            if args.fail_on_artifacts_upload_error:
                (run_dir / "run_summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                return 1
    else:
        logger.info("No HF token — skipping Hub upload.")

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Visualization complete. Output: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

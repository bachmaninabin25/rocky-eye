"""Ingredient 10 — model comparison, benchmarking and report tables/figures.

Three kinds of comparison live here:

* systematics models against each other, per visit (BIC + red-noise check);
* eclipse vs no-eclipse (is the detection real?);
* our numbers against the published ones.

Plus the figures and Markdown tables the report is built from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .s04_eclipse import Orbit
from .s05_systematics import binned_rms, red_noise_beta
from .s06_fit import EclipseFit, Visit


# ------------------------------------------------- systematics comparison ---

def compare_systematics(visit: Visit, orbit: Orbit, priors: dict, candidates: list[str],
                        t_pred: float | None = None) -> list[dict]:
    """Fit one visit with every candidate systematics model; return one row per model.

    Each row holds the maximum-likelihood depth, the white-noise level, the BIC,
    and beta (red noise on the eclipse-ingress-to-duration timescale).
    """
    rows = []
    cadence = float(np.median(np.diff(visit.t)) * 86400.0)
    for name in candidates:
        v = visit.with_model(name)
        fit = EclipseFit([v], orbit, priors)
        if t_pred is not None:
            fit.t_pred = [t_pred]
        ml = fit.max_likelihood()
        resid = ml["parts"][0]["resid"]
        rows.append({
            "model": v.model.name, "k": ml["k"], "n": ml["n"], "bic": float(ml["bic"]), "aic": float(ml["aic"]),
            "depth_ppm": float(ml["p"][0]), "dt_min": float(ml["p"][1]),
            "sigma_ppm": ml["sigma_ppm"][0],
            "beta": red_noise_beta(resid, cadence, 0.5 * orbit.duration_full() * 86400.0),
            "p": ml["p"].tolist(),
        })
    best = min(r["bic"] for r in rows)
    for r in rows:
        r["dbic"] = r["bic"] - best
    return rows


def choose_model(rows: list[dict], max_dbic: float = 2.0) -> dict:
    """Pick the simplest model whose BIC is within ``max_dbic`` of the best.

    BIC differences below ~2 are not meaningful, so ties go to fewer parameters.
    """
    ok = [r for r in rows if r["dbic"] <= max_dbic]
    return min(ok, key=lambda r: (r["k"], r["bic"]))


def detection_bic(fit: EclipseFit) -> dict:
    """Delta-BIC between the eclipse model and the same model with depth fixed to zero."""
    ml = fit.max_likelihood()
    ml0 = fit.max_likelihood(depth_fixed=0.0)
    return {"bic_eclipse": float(ml["bic"]), "bic_flat": float(ml0["bic"]),
            "dbic": float(ml0["bic"] - ml["bic"]), "depth_ml_ppm": float(ml["p"][0])}


# ------------------------------------------------------------ benchmarking ---

def tension(a: float, ea: float, b: float, eb: float) -> float:
    """Difference between two measurements in units of their combined 1-sigma error."""
    return (a - b) / np.hypot(ea, eb)


def benchmark_rows(ours: dict, published: list[dict]) -> list[dict]:
    """Compare our depths {label: (value, err)} with published entries.

    Each published entry: {"ref", "label", "depth_ppm", "err_ppm"}; labels match ours.
    Note: all analyses share the same photons, so the "tension" is an upper
    bound on how unusual a difference is — the errors are not independent.
    """
    rows = []
    for pub in published:
        if pub["label"] not in ours:
            continue
        v, e = ours[pub["label"]]
        rows.append({"label": pub["label"], "ref": pub["ref"], "ours": v, "ours_err": e,
                     "published": pub["depth_ppm"], "published_err": pub["err_ppm"],
                     "diff": v - pub["depth_ppm"],
                     "tension_sigma": tension(v, e, pub["depth_ppm"], pub["err_ppm"])})
    return rows


def md_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    """Markdown table. ``columns`` = [(key, header, format), ...]."""
    head = "| " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for r in rows:
        cells = []
        for key, _, fmt in columns:
            val = r.get(key, "")
            cells.append(format(val, fmt) if fmt and isinstance(val, (int, float, np.floating)) else str(val))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + body)


# ------------------------------------------------------------------ plots ---

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    return plt


def _bin(t, y, n):
    nb = len(t) // n
    return (t[: nb * n].reshape(nb, n).mean(1), y[: nb * n].reshape(nb, n).mean(1),
            y[: nb * n].reshape(nb, n).std(1, ddof=1) / np.sqrt(n))


def plot_raw_light_curves(lcs: dict, path: Path, title: str = "") -> None:
    """Raw light curve, centroid, width and background for each visit."""
    plt = _plt()
    names = list(lcs)
    fig, axes = plt.subplots(4, len(names), figsize=(3.2 * len(names), 7), sharex="col", squeeze=False)
    for j, nm in enumerate(names):
        lc = lcs[nm]
        th = (lc["time"] - lc["time"][0]) * 24
        g = lc["good"]
        axes[0, j].plot(th[g], (lc["flux"][g] - 1) * 1e6, ".", ms=1, alpha=0.3, color="0.5")
        tb, fb, _ = _bin(th[g], lc["flux"][g], 40)
        axes[0, j].plot(tb, (fb - 1) * 1e6, "k.", ms=3)
        lim = 6 * np.nanstd(fb - 1) * 1e6 + 300
        axes[0, j].set_ylim(-lim, lim)
        axes[0, j].set_title(f"{nm} (r={lc['best_radius']} px)")
        axes[1, j].plot(th, lc["x"] - np.nanmedian(lc["x"]), lw=0.4, label="x")
        axes[1, j].plot(th, lc["y"] - np.nanmedian(lc["y"]), lw=0.4, label="y")
        axes[2, j].plot(th, lc["sx"], lw=0.4, label="sx")
        axes[2, j].plot(th, lc["sy"], lw=0.4, label="sy")
        axes[3, j].plot(th, lc["bg"], lw=0.4, color="C4")
        axes[3, j].set_xlabel("hours since start")
    for ax, lab in zip(axes[:, 0], ["flux - 1 (ppm)", "centroid (px)", "PSF width (px)", "background"]):
        ax.set_ylabel(lab)
    axes[1, 0].legend(fontsize=7)
    axes[2, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fit(fit: EclipseFit, p: np.ndarray, path: Path, title: str = "", nbin: int = 20) -> None:
    """Per visit: data with the full model, the systematics-corrected eclipse, residuals."""
    plt = _plt()
    parts = fit.chi2_parts(p)
    n = len(fit.visits)
    fig, axes = plt.subplots(3, n, figsize=(3.3 * n, 6.5), sharex="col", squeeze=False)
    for j, (v, pt) in enumerate(zip(fit.visits, parts)):
        th = (v.t - fit.t_ecl(p, j)) * 24
        axes[0, j].plot(th, (v.flux - 1) * 1e6, ".", ms=1, color="0.6", alpha=0.4)
        axes[0, j].plot(th, (pt["model"] - 1) * 1e6, "r-", lw=1)
        corr = v.flux / pt["systematics"]
        tb, fb, eb = _bin(th, corr, nbin)
        axes[1, j].errorbar(tb, (fb - 1) * 1e6, eb * 1e6, fmt="k.", ms=3, lw=0.6)
        axes[1, j].plot(th, (pt["astro"] - 1) * 1e6, "r-", lw=1.2)
        tb, rb, eb = _bin(th, pt["resid"], nbin)
        axes[2, j].errorbar(tb, rb * 1e6, eb * 1e6, fmt="k.", ms=3, lw=0.6)
        axes[2, j].axhline(0, color="r", lw=0.8)
        axes[0, j].set_title(f"{v.name}: {v.model.name}")
        axes[2, j].set_xlabel("hours from mid-eclipse")
    for ax, lab in zip(axes[:, 0], ["raw flux - 1 (ppm)", "corrected - 1 (ppm)", "residual (ppm)"]):
        ax.set_ylabel(lab)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_phase_folded(fit: EclipseFit, p: np.ndarray, path: Path, title: str = "", bin_min: float = 10.0) -> None:
    """All visits, systematics removed, stacked on time from mid-eclipse."""
    plt = _plt()
    parts = fit.chi2_parts(p)
    th = np.concatenate([(v.t - fit.t_ecl(p, j)) * 24 for j, v in enumerate(fit.visits)])
    corr = np.concatenate([v.flux / pt["systematics"] for v, pt in zip(fit.visits, parts)])
    o = np.argsort(th)
    th, corr = th[o], corr[o]
    edges = np.arange(th.min(), th.max() + bin_min / 60, bin_min / 60)
    idx = np.digitize(th, edges)
    tb = np.array([th[idx == k].mean() for k in np.unique(idx) if (idx == k).sum() > 5])
    fb = np.array([corr[idx == k].mean() for k in np.unique(idx) if (idx == k).sum() > 5])
    eb = np.array([corr[idx == k].std(ddof=1) / np.sqrt((idx == k).sum()) for k in np.unique(idx) if (idx == k).sum() > 5])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(th, (corr - 1) * 1e6, ".", ms=1, color="0.75", alpha=0.3, zorder=1)
    ax.errorbar(tb, (fb - 1) * 1e6, eb * 1e6, fmt="ko", ms=3.5, lw=0.8, zorder=3)
    tt = np.linspace(th.min(), th.max(), 2000)
    ax.plot(tt, p[0] * fit.shape(tt / 24, 0.0), "r-", lw=1.5, zorder=2, label=f"depth = {p[0]:.0f} ppm")
    lim = max(4 * np.nanstd(fb - 1) * 1e6, abs(p[0]) * 1.5) + 50
    ax.set_ylim(-lim, lim + abs(p[0]))
    ax.set_xlabel("hours from mid-eclipse")
    ax.set_ylabel("planet flux / star flux (ppm)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_allan(fit: EclipseFit, p: np.ndarray, path: Path, title: str = "") -> None:
    """Binned-RMS ("Allan") plot of the residuals of each visit vs the white-noise expectation."""
    plt = _plt()
    parts = fit.chi2_parts(p)
    n = len(fit.visits)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False, sharey=True)
    for j, (v, pt) in enumerate(zip(fit.visits, parts)):
        cad = np.median(np.diff(v.t)) * 1440
        sizes, rms, exp = binned_rms(pt["resid"])
        axes[0, j].loglog(sizes * cad, rms * 1e6, "k-", label="residuals")
        axes[0, j].loglog(sizes * cad, exp * 1e6, "r--", label="white noise")
        axes[0, j].set_title(v.name)
        axes[0, j].set_xlabel("bin width (min)")
    axes[0, 0].set_ylabel("RMS (ppm)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_corner(flat: np.ndarray, names: list[str], path: Path, keep: list[str] | None = None, truths=None) -> None:
    import corner

    plt = _plt()
    plt.rcParams.update({"axes.grid": False})
    idx = [i for i, n in enumerate(names) if keep is None or n in keep]
    fig = corner.corner(flat[:, idx], labels=[names[i] for i in idx], quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_fmt=".2f", truths=truths, title_kwargs={"fontsize": 8})
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_trace(chain: np.ndarray, names: list[str], path: Path, keep: list[str] | None = None, burn: int = 0) -> None:
    plt = _plt()
    idx = [i for i, n in enumerate(names) if keep is None or n in keep]
    fig, axes = plt.subplots(len(idx), 1, figsize=(7, 1.5 * len(idx)), sharex=True, squeeze=False)
    for ax, i in zip(axes[:, 0], idx):
        ax.plot(chain[:, ::4, i], lw=0.3, alpha=0.5, color="k")
        ax.axvline(burn, color="r", lw=0.8)
        ax.set_ylabel(names[i], fontsize=7)
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_depths(per_visit: dict, joint: tuple, path: Path, published: list[dict] | None = None,
                title: str = "", truth: float | None = None) -> None:
    """Per-visit and joint depths, with optional published values alongside."""
    plt = _plt()
    names = list(per_visit)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 3.5, 3.6))
    x = np.arange(len(names))
    ax.errorbar(x, [per_visit[n][0] for n in names], [per_visit[n][1] for n in names],
                fmt="o", color="C0", capsize=3, label="RockyEye (per visit)")
    ax.axhspan(joint[0] - joint[1], joint[0] + joint[1], color="C0", alpha=0.2)
    ax.axhline(joint[0], color="C0", lw=1, label=f"RockyEye joint: {joint[0]:.0f} ± {joint[1]:.0f} ppm")
    if published:
        colours = {}
        for pub in published:
            col = colours.setdefault(pub["ref"], f"C{len(colours) + 1}")
            if pub["label"] in names:
                off = 0.12 * list(colours).index(pub["ref"]) + 0.12
                ax.errorbar(names.index(pub["label"]) + off, pub["depth_ppm"], pub["err_ppm"], fmt="s",
                            color=col, capsize=3, ms=4, label=pub["ref"] if col not in ax.get_legend_handles_labels()[1] else None)
        seen = set()
        for pub in published:
            if pub["label"] == "joint" and pub["ref"] not in seen:
                seen.add(pub["ref"])
                ax.axhline(pub["depth_ppm"], color=colours[pub["ref"]], lw=1, ls="--",
                           label=f"{pub['ref']} joint: {pub['depth_ppm']:.0f} ± {pub['err_ppm']:.0f}")
    if truth is not None:
        ax.axhline(truth, color="k", ls=":", label=f"truth {truth:.0f} ppm")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("eclipse depth (ppm)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_injection(inj: dict, path: Path, title: str = "") -> None:
    plt = _plt()
    depths = sorted(float(d) for d in inj["by_depth"])
    get = lambda d: inj["by_depth"].get(d) or inj["by_depth"][str(d)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4))
    a.errorbar(depths, [get(d)["recovered_mean"] for d in depths], [get(d)["recovered_std"] for d in depths],
               fmt="o", capsize=3, label="recovered (mean ± scatter)")
    a.plot(depths, depths, "k--", lw=0.8, label="1:1")
    a.set_xlabel("injected depth (ppm)")
    a.set_ylabel("recovered depth (ppm)")
    a.legend(fontsize=7)
    b.plot(depths, [100 * get(d)["detection_rate"] for d in depths], "o-")
    b.set_xlabel("injected depth (ppm)")
    b.set_ylabel("detected at 3σ (%)")
    b.set_ylim(-3, 103)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_physics(depth_samples: np.ndarray, comparison: dict, path: Path, title: str = "") -> None:
    """Measured depth posterior against the predicted depth of each scenario."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(depth_samples, bins=60, density=True, color="0.8", label="measured depth (posterior)")
    for k, (name, c) in enumerate(comparison.items()):
        ax.axvspan(c["depth_ppm"] - c["depth_err_ppm"], c["depth_ppm"] + c["depth_err_ppm"], color=f"C{k}", alpha=0.25)
        ax.axvline(c["depth_ppm"], color=f"C{k}", lw=1.4, label=f"{name}: {c['depth_ppm']:.0f} ppm ({c['tension_sigma']:+.1f}σ)")
    ax.set_xlabel("eclipse depth at 15 µm (ppm)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

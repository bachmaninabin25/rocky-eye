"""Ingredient 07 — MCMC convergence diagnostics.

An MCMC chain is only useful once it has forgotten where it started and has
been run long enough to map the posterior. Nothing can *prove* convergence, so
we use several independent warning lights and require all of them to be off:

* integrated autocorrelation time tau  -> how many steps make one independent sample
* chain length in units of tau         -> is the tau estimate itself trustworthy
* split R-hat                          -> do different walkers / halves agree
* effective sample size                -> are there enough independent samples
* acceptance fraction                  -> is the sampler moving at all
* Geweke z                             -> does the start of the chain match the end

Chains are arrays of shape (n_steps, n_walkers, n_dim), as returned by
``emcee``'s ``get_chain()``.
"""
from __future__ import annotations

import numpy as np


def autocorr_1d(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function of a 1-D series via FFT."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x - x.mean(), n=nfft)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    return acf / acf[0] if acf[0] > 0 else np.ones(n)


def integrated_autocorr_time(chain: np.ndarray, c: float = 5.0) -> np.ndarray:
    """Integrated autocorrelation time per parameter (Goodman & Weare / Sokal window).

    The ACF is averaged over walkers, then summed out to the smallest window M
    with M >= c * tau(M). Returned in units of steps.
    """
    n_steps, n_walkers, n_dim = chain.shape
    taus = np.empty(n_dim)
    for d in range(n_dim):
        acf = np.mean([autocorr_1d(chain[:, w, d]) for w in range(n_walkers)], axis=0)
        tau_m = 2.0 * np.cumsum(acf) - 1.0
        m = np.arange(len(tau_m))
        ok = m >= c * tau_m
        window = np.argmax(ok) if ok.any() else len(tau_m) - 1
        taus[d] = max(tau_m[window], 1.0)
    return taus


def split_rhat(chain: np.ndarray) -> np.ndarray:
    """Split R-hat per parameter: each walker is cut in half, halves are compared.

    R-hat compares the variance *between* sequences with the variance *within*
    them. If all sequences sample the same distribution R-hat -> 1.
    """
    n_steps, n_walkers, n_dim = chain.shape
    half = n_steps // 2
    seq = np.concatenate([chain[:half], chain[half:2 * half]], axis=1)  # (half, 2*walkers, dim)
    n = seq.shape[0]
    means = seq.mean(axis=0)
    within = seq.var(axis=0, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    var_plus = (n - 1) / n * within + between / n
    return np.sqrt(var_plus / np.where(within > 0, within, np.nan))


def geweke_z(chain: np.ndarray, first: float = 0.1, last: float = 0.5) -> np.ndarray:
    """Geweke z-score: mean of the first 10% vs the last 50% of the chain, in sigma.

    The error on each mean accounts for autocorrelation through tau.
    """
    n_steps = chain.shape[0]
    a = chain[: int(first * n_steps)]
    b = chain[int((1 - last) * n_steps):]
    tau = integrated_autocorr_time(chain)

    def mean_var(seg):
        n_eff = max(seg.shape[0] * seg.shape[1], 1) / tau
        return seg.mean(axis=(0, 1)), seg.var(axis=(0, 1), ddof=1) / np.maximum(n_eff, 1.0)

    ma, va = mean_var(a)
    mb, vb = mean_var(b)
    return (ma - mb) / np.sqrt(va + vb)


def diagnose(chain: np.ndarray, names: list[str] | None = None,
             acceptance: np.ndarray | None = None,
             min_tau_multiple: float = 50.0, max_rhat: float = 1.01,
             min_ess: float = 1000.0, max_geweke: float = 3.0) -> dict:
    """Run every diagnostic on a (post burn-in) chain and return a report with flags.

    ``converged`` is True only if no flag is raised.
    """
    n_steps, n_walkers, n_dim = chain.shape
    names = names or [f"p{i}" for i in range(n_dim)]
    tau = integrated_autocorr_time(chain)
    rhat = split_rhat(chain)
    ess = n_steps * n_walkers / tau
    gz = geweke_z(chain)
    flags = []
    for i, nm in enumerate(names):
        if n_steps < min_tau_multiple * tau[i]:
            flags.append(f"{nm}: chain is only {n_steps / tau[i]:.0f} tau long (need {min_tau_multiple:.0f})")
        if not np.isfinite(rhat[i]) or rhat[i] > max_rhat:
            flags.append(f"{nm}: split R-hat = {rhat[i]:.3f} (> {max_rhat})")
        if ess[i] < min_ess:
            flags.append(f"{nm}: effective sample size {ess[i]:.0f} (< {min_ess:.0f})")
        if abs(gz[i]) > max_geweke:
            flags.append(f"{nm}: Geweke |z| = {abs(gz[i]):.1f} (> {max_geweke})")
    acc = None
    if acceptance is not None:
        acc = float(np.mean(acceptance))
        if not 0.05 < acc < 0.9:
            flags.append(f"mean acceptance fraction {acc:.2f} is outside (0.05, 0.9)")
        stuck = int(np.sum(np.asarray(acceptance) < 0.02))
        if stuck:
            flags.append(f"{stuck} walkers are stuck (acceptance < 0.02)")
    return {
        "n_steps": n_steps, "n_walkers": n_walkers, "names": list(names),
        "tau": tau.tolist(), "steps_over_tau": (n_steps / tau).tolist(),
        "rhat": rhat.tolist(), "ess": ess.tolist(), "geweke_z": gz.tolist(),
        "acceptance": acc, "flags": flags, "converged": len(flags) == 0,
    }


def format_report(d: dict) -> str:
    lines = [f"{'parameter':<22}{'tau':>8}{'N/tau':>8}{'R-hat':>8}{'ESS':>9}{'Geweke':>8}"]
    for i, nm in enumerate(d["names"]):
        lines.append(f"{nm:<22}{d['tau'][i]:>8.1f}{d['steps_over_tau'][i]:>8.0f}"
                     f"{d['rhat'][i]:>8.3f}{d['ess'][i]:>9.0f}{d['geweke_z'][i]:>8.2f}")
    if d.get("acceptance") is not None:
        lines.append(f"mean acceptance fraction: {d['acceptance']:.2f}")
    lines.append("CONVERGED" if d["converged"] else "NOT CONVERGED:")
    lines += [f"  - {f}" for f in d["flags"]]
    return "\n".join(lines)

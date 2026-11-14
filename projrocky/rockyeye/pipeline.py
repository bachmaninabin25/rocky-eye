"""Stage orchestration: what each numbered script in ``scripts/`` actually runs.

Each stage reads the previous stage's cached products, does its work, writes
its own products under ``products/<config name>/stageNN/`` and records the
config hash it was made with. Rerunning a stage whose inputs and config have
not changed is a no-op.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import s01_fetch, s02_calibrate, s03_photometry, s08_injection, s09_physics, s10_compare
from .common import (atomic_write_json, config_hash, is_cached, log, mark_cached, product_dir,
                     read_json)
from .s04_eclipse import Orbit
from .s06_fit import EclipseFit, ShapeTable, make_visit


def visit_names(cfg: dict, visits: list[str] | None) -> list[str]:
    return list(visits) if visits else list(cfg["visits"])


# ------------------------------------------------------------- 01 / 02 ---

def stage_fetch(cfg: dict, visits=None, dry_run: bool = True) -> dict:
    """Print the download plan. With dry_run=False, download every raw file (no streaming)."""
    plan = s01_fetch.plan(cfg, visits)
    print(s01_fetch.format_plan(plan, streaming=dry_run))
    if not dry_run:
        require_validation(cfg)
        if plan["missing_gb"] + cfg["source"].get("min_free_gb", 8.0) > plan["free_gb"]:
            raise OSError("the full raw set does not fit on this disk; run 02_calibrate.py, which "
                          "streams one segment at a time")
        for rf in s01_fetch.list_files(cfg, visits):
            s01_fetch.download(rf, min_free_gb=cfg["source"].get("min_free_gb", 8.0))
    return plan


def stage_calibrate(cfg: dict, visits=None, dry_run: bool = False, max_segments=None):
    if not dry_run:
        require_validation(cfg)
    return s02_calibrate.process(cfg, visit_names(cfg, visits), dry_run=dry_run, max_segments=max_segments)


# ------------------------------------------------------------------ 03 ---

def lc_path(cfg: dict, visit: str) -> Path:
    return product_dir(cfg, "stage03") / f"{visit}_lc.npz"


def stage_photometry(cfg: dict, visits=None, force: bool = False) -> dict:
    require_validation(cfg)          # real data are not analysed before the simulated gate passes
    key = config_hash(s02_calibrate.stage_key(cfg), cfg["photometry"])
    out = {}
    for v in visit_names(cfg, visits):
        p = lc_path(cfg, v)
        if force or not is_cached(p, key):
            stamps = s02_calibrate.load_visit_stamps(cfg, v)
            lc = s03_photometry.extract_light_curve(stamps, cfg["photometry"])
            s03_photometry.save_light_curve(p, lc)
            mark_cached(p, key)
            log.info("%s: %d integrations, aperture %.1f px, scatter %.0f ppm (pipeline error %.0f ppm)",
                     v, len(lc["time"]), lc["best_radius"], lc["scatter_best_ppm"], lc["median_err_ppm"])
        out[v] = s03_photometry.load_light_curve(p)
    s10_compare.plot_raw_light_curves(out, product_dir(cfg, "figures") / "raw_light_curves.png",
                                      f"{cfg['target']}: raw photometry")
    return out


def load_light_curves(cfg: dict, visits=None) -> dict:
    return {v: s03_photometry.load_light_curve(lc_path(cfg, v)) for v in visit_names(cfg, visits)}


# ------------------------------------------------------------------ 04 ---

def _trim(cfg: dict, visit: str) -> int:
    t = cfg["fit"].get("trim_start", 0)
    return t.get(visit, t.get("default", 0)) if isinstance(t, dict) else t


def build_visit(cfg: dict, name: str, lc: dict, model: str):
    f = cfg["fit"]
    mask = None
    for lo, hi in f.get("mask_hours", {}).get(name, []):      # optional bad stretches, hours from start
        th = (lc["time"] - lc["time"][0]) * 24
        mask = (mask if mask is not None else np.ones(len(th), bool)) & ~((th >= lo) & (th < hi))
    return make_visit(name, lc, model, trim_start=_trim(cfg, name), bin_size=f.get("bin_size", 1), mask=mask)


def stage_systematics(cfg: dict, visits=None, force: bool = False) -> dict:
    """Compare the candidate systematics models on each visit and choose one by BIC."""
    f = cfg["fit"]
    out_path = product_dir(cfg, "stage04") / "systematics.json"
    key = config_hash(s02_calibrate.stage_key(cfg), cfg["photometry"], f, visit_names(cfg, visits))
    if not force and is_cached(out_path, key):
        return read_json(out_path)
    orbit = Orbit.from_config(cfg["system"])
    lcs = load_light_curves(cfg, visits)
    result = {}
    for name, lc in lcs.items():
        visit = build_visit(cfg, name, lc, f["candidate_models"][0])
        rows = s10_compare.compare_systematics(visit, orbit, f["priors"], f["candidate_models"])
        chosen = s10_compare.choose_model(rows)
        forced = f.get("model_override", {}).get(name)
        result[name] = {"rows": rows, "chosen": forced or chosen["model"], "chosen_by_bic": chosen["model"],
                        "n_points": visit.n}
        log.info("%s: chose %s | %s", name, result[name]["chosen"],
                 ", ".join(f"{r['model']} dBIC={r['dbic']:.1f} d={r['depth_ppm']:.0f}" for r in rows))
    atomic_write_json(out_path, result)
    mark_cached(out_path, key)
    return result


# ------------------------------------------------------------------ 05 ---

def build_fit(cfg: dict, visits=None, models: dict | None = None, shared_time=None,
              priors: dict | None = None) -> EclipseFit:
    f = cfg["fit"]
    sysres = read_json(product_dir(cfg, "stage04") / "systematics.json")
    lcs = load_light_curves(cfg, visits)
    vs = [build_visit(cfg, n, lc, (models or {}).get(n) or sysres[n]["chosen"]) for n, lc in lcs.items()]
    return EclipseFit(vs, Orbit.from_config(cfg["system"]), priors or f["priors"],
                      shared_time=f.get("shared_time", True) if shared_time is None else shared_time)


def _start_point(fit: EclipseFit) -> np.ndarray:
    return fit.max_likelihood()["p"]


def run_one_fit(cfg: dict, fit: EclipseFit, tag: str, force: bool = False, figures: bool = True) -> dict:
    """MCMC for one EclipseFit; cached summary JSON + samples NPZ + chain HDF5."""
    f = cfg["fit"]
    d = product_dir(cfg, "stage05")
    key = config_hash(s02_calibrate.stage_key(cfg), cfg["photometry"], f, fit.names,
                      [v.model.name for v in fit.visits], [fit.depth_lim, fit.dt_lim])
    js, npz, h5 = d / f"{tag}.json", d / f"{tag}_samples.npz", d / f"{tag}_chain.h5"
    if not force and is_cached(js, key):
        return read_json(js)
    keyfile = Path(str(h5) + ".key")
    if h5.exists() and (force or not keyfile.exists() or keyfile.read_text() != key):
        h5.unlink()                       # stale chain from another config: start over
    keyfile.write_text(key)
    ml = fit.max_likelihood()
    mc = f["mcmc"]
    res = fit.run_mcmc(h5, ml["p"], n_walkers=mc["n_walkers"], max_steps=mc["max_steps"],
                       check_every=mc["check_every"], tau_multiple=mc["tau_multiple"], seed=mc["seed"])
    det = s10_compare.detection_bic(fit)
    p_med = np.array([res["summary"][n]["median"] for n in fit.names])
    parts = fit.chi2_parts(p_med)
    cad = [float(np.median(np.diff(v.t)) * 86400) for v in fit.visits]
    from .s05_systematics import red_noise_beta
    summary = {
        "tag": tag, "names": fit.names, "summary": res["summary"], "diagnostics": res["diagnostics"],
        "n_steps": res["n_steps"], "burn": res["burn"], "thin": res["thin"],
        "max_likelihood": {"p": ml["p"], "bic": ml["bic"], "k": ml["k"], "n": ml["n"], "sigma_ppm": ml["sigma_ppm"]},
        "detection": det, "models": {v.name: v.model.name for v in fit.visits},
        "p_median": p_med, "visits": [v.name for v in fit.visits],
        "t_ecl": [fit.t_ecl(p_med, i) for i in range(len(fit.visits))],
        "t_pred": fit.t_pred,
        "resid_rms_ppm": [float(np.std(pt["resid"]) * 1e6) for pt in parts],
        "beta": [red_noise_beta(pt["resid"], c, 0.5 * fit.orbit.duration_full() * 86400)
                 for pt, c in zip(parts, cad)],
    }
    np.savez_compressed(npz, flat=res["flat"], lnp=res["lnp"], names=np.array(fit.names))
    atomic_write_json(js, summary)
    mark_cached(js, key)
    if figures:
        figs = product_dir(cfg, "figures")
        ttl = f"{cfg['target']} — {tag}"
        s10_compare.plot_fit(fit, p_med, figs / f"fit_{tag}.png", ttl)
        s10_compare.plot_allan(fit, p_med, figs / f"allan_{tag}.png", ttl)
        import emcee
        chain = emcee.backends.HDFBackend(str(h5), read_only=True).get_chain()
        keep = [n for n in fit.names if not n.startswith("log_sigma")][:8]
        s10_compare.plot_trace(chain, fit.names, figs / f"trace_{tag}.png", keep, res["burn"])
        s10_compare.plot_corner(res["flat"], fit.names, figs / f"corner_{tag}.png", keep)
        if len(fit.visits) > 1:
            s10_compare.plot_phase_folded(fit, p_med, figs / f"phase_{tag}.png", ttl)
    s = res["summary"]["depth_ppm"]
    log.info("%s: depth = %.0f +%.0f -%.0f ppm | converged=%s | dBIC(eclipse)=%.1f", tag, s["median"],
             s["plus"], s["minus"], res["diagnostics"]["converged"], det["dbic"])
    return read_json(js)


def eclipse_groups(cfg: dict, names: list[str]) -> dict:
    """Map eclipse label -> the fit units ("visits") that cover it.

    Normally one unit per eclipse. An eclipse observed in two exposures (GJ 3929 b
    eclipse 4) is two units with independent systematics but one shared depth.
    """
    groups = cfg.get("eclipses") or {n: [n] for n in names}
    return {lab: [u for u in units if u in names] for lab, units in groups.items()
            if any(u in names for u in units)}


def stage_fit(cfg: dict, visits=None, force: bool = False) -> dict:
    """Joint fit of all visits, then each eclipse alone with timing tied to the joint result."""
    names = visit_names(cfg, visits)
    groups = eclipse_groups(cfg, names)
    out = {}
    joint = build_fit(cfg, names)
    out["joint"] = run_one_fit(cfg, joint, "joint", force)
    dt = out["joint"]["summary"].get("dt_min")
    pri = dict(cfg["fit"]["priors"])
    if dt is not None and len(groups) > 1:              # single eclipses cannot pin the time themselves
        half = max(3 * max(dt["plus"], dt["minus"]), 2.0)
        pri["dt_min"] = [dt["median"] - half, dt["median"] + half]
    for lab, units in groups.items():
        out[lab] = run_one_fit(cfg, build_fit(cfg, units, priors=pri), f"visit_{lab}", force)
    if visits is None:
        for tag, labs in cfg["fit"].get("extra_joint_fits", {}).items():
            units = [u for lab in labs for u in groups[lab]]
            out[tag] = run_one_fit(cfg, build_fit(cfg, units), tag, force)

    def val(tag):
        s = out[tag]["summary"]["depth_ppm"]
        return s["median"], 0.5 * (s["plus"] + s["minus"])

    s10_compare.plot_depths({lab: val(lab) for lab in groups}, val("joint"),
                            product_dir(cfg, "figures") / "depths.png", cfg.get("published"),
                            f"{cfg['target']}: eclipse depths", cfg.get("truth", {}).get("depth_ppm"))
    return out


def load_samples(cfg: dict, tag: str = "joint") -> dict:
    z = np.load(product_dir(cfg, "stage05") / f"{tag}_samples.npz")
    names = [str(n) for n in z["names"]]
    return {"flat": z["flat"], "names": names, "depth_ppm": z["flat"][:, names.index("depth_ppm")]}


# ------------------------------------------------------------------ 06 ---

def stage_injection(cfg: dict, visits=None, force: bool = False) -> dict:
    inj = cfg["injection"]
    out_path = product_dir(cfg, "stage06") / "injection.json"
    key = config_hash(s02_calibrate.stage_key(cfg), cfg["photometry"], cfg["fit"], inj)
    if not force and is_cached(out_path, key):
        return read_json(out_path)
    fit = build_fit(cfg, visit_names(cfg, visits))
    joint = read_json(product_dir(cfg, "stage05") / "joint.json")
    p = np.array(joint["p_median"])
    lo, hi = cfg["fit"]["priors"]["dt_min"]
    grid = np.linspace(lo, hi, inj.get("n_dt_grid", 41))
    res = s08_injection.run_injection_grid(fit, p, inj["depths_ppm"], inj["n_trials"], grid,
                                           inj.get("detection_sigma", 3.0), inj.get("seed", 7),
                                           inj.get("scan_timing", True))
    res["mcmc_depth_err_ppm"] = 0.5 * (joint["summary"]["depth_ppm"]["plus"] + joint["summary"]["depth_ppm"]["minus"])
    atomic_write_json(out_path, res)
    mark_cached(out_path, key)
    s10_compare.plot_injection(res, product_dir(cfg, "figures") / "injection_recovery.png",
                               f"{cfg['target']}: injection-recovery ({inj['n_trials']} trials per depth)")
    for d, r in res["by_depth"].items():
        log.info("inject %5.0f ppm -> recovered %6.1f ± %.1f (bias %+.1f ± %.1f), detected %.0f%%", float(d),
                 r["recovered_mean"], r["recovered_std"], r["bias"], r["bias_err"], 100 * r["detection_rate"])
    return read_json(out_path)


# ------------------------------------------------------------------ 07 ---

def stage_physics(cfg: dict, force: bool = False, tag: str = "joint") -> dict:
    ph = cfg["physics"]
    out_path = product_dir(cfg, "stage07") / "physics.json"
    key = config_hash(s02_calibrate.stage_key(cfg), cfg["photometry"], cfg["fit"], ph, cfg["system"], tag)
    if not force and is_cached(out_path, key):
        return read_json(out_path)
    rng = np.random.default_rng(ph.get("seed", 11))
    depth = load_samples(cfg, tag)["depth_ppm"]
    band = s09_physics.Band()
    n = ph.get("n_draws", 20000)
    draws = s09_physics.draw_system(cfg["system"], n, rng)
    result = {"pivot_um": band.pivot_um, "depth_ppm": [float(v) for v in np.percentile(depth, [15.865, 50, 84.135])],
              "stellar_models": {}}
    stars = {k: dict(v) for k, v in ph["stellar_models"].items()}
    cal = ph.get("flux_calibration")
    if cal:                                # our own absolute flux measurement, one value per visit
        meas = {v: s03_photometry.absolute_flux_mjy(s02_calibrate.load_visit_stamps(cfg, v), cal)
                for v in visit_names(cfg, None)}
        vals = np.array([m["flux_mjy"] for m in meas.values()])
        flux = float(vals.mean())
        err = float(np.hypot(flux * cal.get("systematic_frac", 0.03), vals.std(ddof=1) if len(vals) > 1 else 0.0))
        result["measured_flux"] = {"per_visit": meas, "flux_mjy": flux, "err_mjy": err}
        stars[cal.get("apply_to", "measured_flux")].update(flux_mjy=flux, flux_mjy_err=err)
        log.info("measured stellar flux: %.2f ± %.2f mJy (visits: %s)", flux, err, np.round(vals, 2))
    for label, star in stars.items():
        kw = {k: (rng.normal(star[k], star.get(k + "_err", 0.0), n) if k + "_err" in star else star[k])
              for k in ("flux_mjy",) if k in star}
        if star["mode"] == "flux":
            kw.update(rstar_rsun=draws["rstar_rsun"], dist_pc=draws["dist_pc"])
        else:
            kw.update(teff=draws["teff"])
        i_star = s09_physics.stellar_intensity(band, star["mode"], **kw)
        pred = s09_physics.scenario_depths(band, draws, i_star, ph["scenarios"])
        comp = s09_physics.compare_scenarios(depth, pred, ph["reference_scenario"])
        for nm, c in comp.items():
            c["verdict"] = s09_physics.interpret_bayes_factor(c["bayes_factor_vs_ref"])
        eps, post, eps95 = s09_physics.infer_redistribution(band, depth, draws, i_star)
        result["stellar_models"][label] = {
            "T_star_brightness": [float(v) for v in np.percentile(band.brightness_temperature(i_star), [15.865, 50, 84.135])],
            "temperature": s09_physics.summarise_temperature(band, depth, draws, i_star, rng),
            "scenarios": comp, "eps_upper95": eps95, "eps_grid": eps, "eps_posterior": post,
        }
        if label == ph["primary_stellar_model"]:
            s10_compare.plot_physics(depth, comp, product_dir(cfg, "figures") / "physics_scenarios.png",
                                     f"{cfg['target']}: measured depth vs scenarios ({label} star)")
    atomic_write_json(out_path, result)
    mark_cached(out_path, key)
    return read_json(out_path)


# ------------------------------------------------------- validation gate ---

def stage_validate(cfg: dict, visits=None, force: bool = False) -> dict:
    """Pixel-level injection-recovery: the gate that must pass before real data is touched.

    Fake eclipses of known depth are planted in the calibrated frames at a time
    well away from the real (simulated) eclipse, whose window is masked. Then
    photometry and the joint MCMC are rerun from those frames. Passing means:
      * the depth-0 run is consistent with zero (no spurious signal), and
      * each injected depth comes back within ``pass_sigma`` (default 2) of its true value, and
      * the depth scale is preserved: (recovered - recovered at 0) / injected within 5% of 1.
    """
    val = cfg["validation"]
    out_path = product_dir(cfg, "validation") / "validation.json"
    key = config_hash(s02_calibrate.stage_key(cfg), cfg["photometry"], cfg["fit"], val, visit_names(cfg, visits))
    if not force and is_cached(out_path, key):
        return read_json(out_path)
    orbit = Orbit.from_config(cfg["system"])
    joint = read_json(product_dir(cfg, "stage05") / "joint.json")
    sysres = read_json(product_dir(cfg, "stage04") / "systematics.json")
    t14 = orbit.duration_full()
    names = [n for n in visit_names(cfg, visits) if n in joint["visits"]]
    f = cfg["fit"]
    shape = ShapeTable(orbit)

    # Choose an injection time per visit: the middle of the longest stretch of kept data
    # that does not touch the (masked) real eclipse, if a whole injected eclipse plus some
    # baseline on each side fits there. Visits are only ~4 eclipse-durations long, so a
    # fixed offset from the real eclipse does not work.
    pad = val.get("mask_pad_min", 10.0) / 1440.0
    base = val.get("min_baseline_durations", 0.25) * t14
    plan = {}
    for n in names:
        lc = s03_photometry.load_light_curve(lc_path(cfg, n))
        t = lc["time"][_trim(cfg, n):]
        t_real = joint["t_ecl"][joint["visits"].index(n)]
        m0, m1 = t_real - 0.5 * t14 - pad, t_real + 0.5 * t14 + pad
        best = None
        for lo, hi in ((t[0], m0), (m1, t[-1])):
            room = hi - lo
            if room >= t14 + 2 * base and (best is None or room > best[1]):
                best = (0.5 * (lo + hi), room)
        if best:
            plan[n] = {"t_inj": best[0], "t_real": t_real, "radius": float(lc["best_radius"]),
                       "room_hours": best[1] * 24.0}
    if not plan:
        raise RuntimeError("no visit has room for a whole injected eclipse away from the real one "
                           f"(needs {(t14 + 2 * base) * 24:.1f} h of clean data on one side)")
    log.info("validation: injecting into %d visits: %s", len(plan), ", ".join(plan))

    results = {}
    for depth in val["depths_ppm"]:
        vs, t_pred = [], []
        for n, pl in plan.items():
            stamps = s02_calibrate.load_visit_stamps(cfg, n)
            lc0 = s03_photometry.load_light_curve(lc_path(cfg, n))
            s = shape(stamps["time"], pl["t_inj"])
            injected = s08_injection.inject_into_stamps(stamps, lc0["bg"], s, depth * 1e-6)
            lc = s03_photometry.extract_light_curve(injected, dict(cfg["photometry"], radius=pl["radius"]))
            keep = np.abs(lc["time"] - pl["t_real"]) > 0.5 * t14 + val.get("mask_pad_min", 10.0) / 1440.0
            vs.append(make_visit(n, lc, sysres[n]["chosen"], trim_start=_trim(cfg, n),
                                 bin_size=f.get("bin_size", 1), mask=keep, break_gap_s=1e9))
            t_pred.append(pl["t_inj"])
        pri = dict(f["priors"], dt_min=val.get("dt_min", [-20.0, 20.0]))
        fit = EclipseFit(vs, orbit, pri, shared_time=True)
        fit.t_pred = t_pred
        ml = fit.max_likelihood()
        mc = dict(f["mcmc"], **val.get("mcmc", {}))
        h5 = product_dir(cfg, "validation") / f"inject_{int(depth)}_chain.h5"
        h5.unlink(missing_ok=True)
        res = fit.run_mcmc(h5, ml["p"], n_walkers=mc["n_walkers"], max_steps=mc["max_steps"],
                           check_every=mc["check_every"], tau_multiple=mc["tau_multiple"], seed=mc["seed"])
        d = res["summary"]["depth_ppm"]
        err = 0.5 * (d["plus"] + d["minus"])
        results[str(depth)] = {"injected": depth, "recovered": d["median"], "err": err,
                               "pull": (d["median"] - depth) / err, "dt_min": res["summary"]["dt_min"]["median"],
                               "ml_depth": float(ml["p"][0]), "converged": res["diagnostics"]["converged"]}
        log.info("validation: injected %4.0f ppm -> recovered %6.1f ± %.1f ppm (pull %+.2f)", depth,
                 d["median"], err, results[str(depth)]["pull"])
        if depth == max(val["depths_ppm"]):
            p_med = np.array([res["summary"][nm]["median"] for nm in fit.names])
            s10_compare.plot_phase_folded(fit, p_med, product_dir(cfg, "figures") / "validation_injected.png",
                                          f"{cfg['target']}: pixel-level injection of {depth:.0f} ppm")
    r0 = results.get("0") or results.get("0.0")
    tol = val.get("pass_sigma", 2.0)
    checks = {f"inject_{k}_within_{tol:g}sigma": bool(abs(r["pull"]) <= tol) for k, r in results.items()}
    if r0:
        for k, r in results.items():
            if r["injected"] > 0:
                gain = (r["recovered"] - r0["recovered"]) / r["injected"]
                r["differential_gain"] = gain            # 1.0 = the depth scale is preserved end to end
                checks[f"gain_{k}_within_5pct"] = bool(abs(gain - 1) < 0.05)
    out = {"plan": plan, "results": results, "checks": checks, "passed": bool(all(checks.values()))}
    atomic_write_json(out_path, out)
    mark_cached(out_path, key)
    log.info("VALIDATION %s  %s", "PASSED" if out["passed"] else "FAILED", checks)
    return read_json(out_path)


def require_validation(cfg: dict) -> None:
    """Refuse to touch real data unless the named validation run has passed (config: ``requires_validation``)."""
    req = cfg.get("requires_validation")
    if not req:
        return
    from .common import load_config, resolve
    vcfg = load_config(resolve(req))
    path = product_dir(vcfg, "validation") / "validation.json"
    if not path.exists() or not read_json(path).get("passed"):
        raise SystemExit(f"BLOCKED: {cfg['name']} needs the validation of {vcfg['name']} to pass first "
                         f"(run scripts/validate.py --config {req})")


# ------------------------------------------------------------------ 08 ---

def stage_report(cfg: dict) -> dict:
    """Collect every stage's numbers into one summary JSON and a Markdown tables file."""
    names = visit_names(cfg, None)
    d5 = product_dir(cfg, "stage05")
    fits = {p.stem: read_json(p) for p in sorted(d5.glob("*.json"))}
    sysres = read_json(product_dir(cfg, "stage04") / "systematics.json")
    lcs = load_light_curves(cfg, names)

    def depth(tag):
        s = fits[tag]["summary"]["depth_ppm"]
        return s["median"], 0.5 * (s["plus"] + s["minus"])

    labels = list(eclipse_groups(cfg, names))
    ours = {n: depth(f"visit_{n}") for n in labels if f"visit_{n}" in fits}
    for tag in fits:
        if tag.startswith("joint"):
            ours[tag] = depth(tag)
    md = [f"# {cfg['target']} — result tables (generated by scripts/08_report.py)\n"]

    md.append("## Photometry\n")
    rows = [{"visit": n, "n": len(lc["time"]), "hours": (lc["time"][-1] - lc["time"][0]) * 24,
             "radius": lc["best_radius"], "scatter": lc["scatter_best_ppm"], "pipe_err": lc["median_err_ppm"],
             "ratio": lc["scatter_best_ppm"] / lc["median_err_ppm"]} for n, lc in lcs.items()]
    md.append(s10_compare.md_table(rows, [("visit", "Visit", ""), ("n", "Integrations", "d"), ("hours", "Duration (h)", ".2f"),
                                          ("radius", "Aperture (px)", ".1f"), ("scatter", "Scatter (ppm)", ".0f"),
                                          ("pipe_err", "Pipeline error (ppm)", ".0f"), ("ratio", "Ratio", ".2f")]))

    md.append("\n## Systematics model comparison (per visit, maximum likelihood)\n")
    for n in names:
        md.append(f"\n**{n}** — chosen: `{sysres[n]['chosen']}`\n")
        md.append(s10_compare.md_table(sysres[n]["rows"], [("model", "Model", ""), ("k", "k", "d"), ("dbic", "ΔBIC", ".1f"),
                                                           ("depth_ppm", "Depth (ppm)", ".0f"), ("sigma_ppm", "σ (ppm)", ".0f"),
                                                           ("beta", "β", ".2f")]))

    md.append("\n## Fits\n")
    rows = []
    for tag, ft in fits.items():
        s, dg = ft["summary"], ft["diagnostics"]
        dt = s.get("dt_min", {"median": float("nan"), "plus": float("nan"), "minus": float("nan")})
        rows.append({"fit": tag, "depth": f"{s['depth_ppm']['median']:.0f} +{s['depth_ppm']['plus']:.0f} −{s['depth_ppm']['minus']:.0f}",
                     "dt": f"{dt['median']:.1f} +{dt['plus']:.1f} −{dt['minus']:.1f}",
                     "dbic": ft["detection"]["dbic"], "rhat": max(dg["rhat"]), "ntau": min(dg["steps_over_tau"]),
                     "ess": min(dg["ess"]), "conv": "yes" if dg["converged"] else "NO",
                     "beta": max(ft["beta"])})
    md.append(s10_compare.md_table(rows, [("fit", "Fit", ""), ("depth", "Depth (ppm)", ""), ("dt", "Δt from phase 0.5 (min)", ""),
                                          ("dbic", "ΔBIC (eclipse)", ".1f"), ("rhat", "max R̂", ".3f"), ("ntau", "min N/τ", ".0f"),
                                          ("ess", "min ESS", ".0f"), ("beta", "max β", ".2f"), ("conv", "Converged", "")]))
    summary = {"ours": ours, "fits": {k: {"depth_ppm": v["summary"]["depth_ppm"], "dt_min": v["summary"].get("dt_min"),
                                           "detection": v["detection"], "converged": v["diagnostics"]["converged"],
                                           "models": v["models"], "beta": v["beta"]} for k, v in fits.items()}}

    # weighted mean of single-visit depths and their mutual consistency
    if len(labels) > 1 and all(n in ours for n in labels):
        d = np.array([ours[n][0] for n in labels]); e = np.array([ours[n][1] for n in labels])
        w = 1 / e ** 2
        mean, err = float((w * d).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
        chi2 = float((w * (d - mean) ** 2).sum())
        from scipy import stats
        summary["visit_consistency"] = {"weighted_mean": mean, "err": err, "chi2": chi2, "dof": len(d) - 1,
                                        "p_value": float(stats.chi2.sf(chi2, len(d) - 1))}
        md.append(f"\nWeighted mean of single-visit depths: {mean:.0f} ± {err:.0f} ppm; "
                  f"consistency χ² = {chi2:.1f} for {len(d) - 1} dof (p = {summary['visit_consistency']['p_value']:.2f}).\n")

    inj_p = product_dir(cfg, "stage06") / "injection.json"
    if inj_p.exists():
        inj = read_json(inj_p)
        md.append("\n## Injection-recovery\n")
        rows = [{"inj": float(k), **v} for k, v in inj["by_depth"].items()]
        md.append(s10_compare.md_table(rows, [("inj", "Injected (ppm)", ".0f"), ("recovered_mean", "Recovered mean", ".1f"),
                                              ("recovered_std", "Scatter", ".1f"), ("bias", "Bias", "+.1f"), ("bias_err", "± ", ".1f"),
                                              ("error_ratio", "Scatter / formal error", ".2f"), ("detection_rate", "Detected (3σ)", ".0%")]))
        summary["injection"] = {"summary": inj["summary"], "by_depth": {k: {kk: vv for kk, vv in v.items() if kk not in ("recovered", "dt")}
                                                                       for k, v in inj["by_depth"].items()}}

    phys_p = product_dir(cfg, "stage07") / "physics.json"
    if phys_p.exists():
        phys = read_json(phys_p)
        md.append("\n## Physics\n")
        for label, sm in phys["stellar_models"].items():
            T = sm["temperature"]
            md.append(f"\n**Stellar model: {label}** — star brightness temperature {sm['T_star_brightness'][1]:.0f} K; "
                      f"planet T_b = {T['T_b'][1]:.0f} +{T['T_b'][2] - T['T_b'][1]:.0f} −{T['T_b'][1] - T['T_b'][0]:.0f} K; "
                      f"T_max(bare rock) = {T['T_max'][1]:.0f} K; R = {T['R'][1]:.2f} +{T['R'][2] - T['R'][1]:.2f} −{T['R'][1] - T['R'][0]:.2f}; "
                      f"ε < {sm['eps_upper95']:.2f} (95%)\n")
            rows = [{"name": k, **v} for k, v in sm["scenarios"].items()]
            md.append(s10_compare.md_table(rows, [("name", "Scenario", ""), ("depth_ppm", "Predicted depth (ppm)", ".0f"),
                                                  ("depth_err_ppm", "±", ".0f"), ("tension_sigma", "Data − model (σ)", "+.1f"),
                                                  ("bayes_factor_vs_ref", "Bayes factor vs ref", ".3g"), ("verdict", "Reading", "")]))
        summary["physics"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("eps_grid", "eps_posterior")}
                              for k, v in phys["stellar_models"].items()}

    if cfg.get("published"):
        md.append("\n## Benchmark against published depths\n")
        rows = s10_compare.benchmark_rows(ours, cfg["published"])
        md.append(s10_compare.md_table(rows, [("label", "Quantity", ""), ("ref", "Reference", ""), ("ours", "RockyEye", ".0f"),
                                              ("ours_err", "±", ".0f"), ("published", "Published", ".0f"), ("published_err", "±", ".0f"),
                                              ("diff", "Difference", "+.0f"), ("tension_sigma", "Tension (σ)", "+.2f")]))
        summary["benchmark"] = rows
    val_p = product_dir(cfg, "validation") / "validation.json"
    if val_p.exists():
        val = read_json(val_p)
        md.append(f"\n## Pixel-level validation — {'PASSED' if val['passed'] else 'FAILED'}\n")
        md.append(s10_compare.md_table(list(val["results"].values()), [("injected", "Injected (ppm)", ".0f"), ("recovered", "Recovered (ppm)", ".1f"),
                                                                       ("err", "±", ".1f"), ("pull", "Pull (σ)", "+.2f"),
                                                                       ("differential_gain", "Differential gain", ".3f")]))
        summary["validation"] = val
    out = product_dir(cfg, "stage08")
    (out / "tables.md").write_text("\n".join(md) + "\n")
    atomic_write_json(out / "summary.json", summary)
    log.info("report tables -> %s", out / "tables.md")
    return summary

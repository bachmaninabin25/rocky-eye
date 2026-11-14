"""Ingredient 02: the pieces of the stage 1-2 runner that do not need the jwst pipeline itself.

(Running Detector1/Image2 needs GBs of reference files, so the pipeline call is
exercised on Kaggle; here we test our own logic around it.)
"""
import numpy as np
import pytest
from astropy.io import fits

from rockyeye import s02_calibrate as cal


def test_select_times_by_integration_number():
    number = np.arange(1, 2284)                 # table for the whole exposure
    t_all = 2461000.0 + number * 1e-4
    t = cal.select_times(number, t_all, 272, 321, 50)       # a chunk of segment 2
    assert len(t) == 50 and t[0] == pytest.approx(2461000.0 + 272e-4)
    seg = np.arange(272, 543)                   # table for one segment only
    t = cal.select_times(seg, 2461000.0 + seg * 1e-4, 300, 310, 11)
    assert t[0] == pytest.approx(2461000.0 + 300e-4)
    assert len(cal.select_times(seg, seg * 1.0, None, None, 271)) == 271
    with pytest.raises(ValueError):
        cal.select_times(seg, seg * 1.0, 1, 50, 50)          # numbers not in the table


def _fake_uncal(path, n_ints=23, n_groups=4, first=101):
    rng = np.random.default_rng(0)
    sci = rng.integers(30000, 60000, (n_ints, n_groups, 6, 8)).astype(np.uint16)   # values above 32767 on purpose
    ref = rng.integers(0, 65535, (n_ints, n_groups, 6, 2)).astype(np.uint16)
    pri = fits.PrimaryHDU()
    pri.header["INTSTART"], pri.header["INTEND"], pri.header["NINTS"] = first, first + n_ints - 1, 500
    times = fits.BinTableHDU.from_columns([fits.Column("integration_number", "J", array=np.arange(first, first + n_ints))], name="INT_TIMES")
    group = fits.BinTableHDU.from_columns([fits.Column("integration_number", "J", array=np.arange(first, first + n_ints))], name="GROUP")
    fits.HDUList([pri, fits.ImageHDU(sci, name="SCI"), fits.ImageHDU(ref, name="REFOUT"), times, group]).writeto(path)
    return sci, ref


def test_split_uncal_is_lossless(tmp_path):
    """Chunks, read back, reproduce the raw unsigned 16-bit data exactly and carry the right INTSTART/INTEND."""
    raw = tmp_path / "x-seg002_mirimage_uncal.fits"
    sci, ref = _fake_uncal(raw)
    chunks = cal.split_uncal(raw, 10, tmp_path / "work")
    assert len(chunks) == 3
    got_sci, got_ref, expect_start = [], [], 101
    for c in chunks:
        with fits.open(c) as h:
            assert h["SCI"].data.dtype == np.uint16
            n = h["SCI"].data.shape[0]
            assert h[0].header["INTSTART"] == expect_start and h[0].header["INTEND"] == expect_start + n - 1
            assert len(h["GROUP"].data) == n and len(h["INT_TIMES"].data) == 23
            expect_start += n
            got_sci.append(h["SCI"].data.copy()); got_ref.append(h["REFOUT"].data.copy())
    assert np.array_equal(np.concatenate(got_sci), sci)
    assert np.array_equal(np.concatenate(got_ref), ref)


def test_small_file_is_not_split(tmp_path):
    raw = tmp_path / "y-seg001_mirimage_uncal.fits"
    _fake_uncal(raw, n_ints=8)
    assert cal.split_uncal(raw, 50, tmp_path / "work") == [raw]


def test_find_star_ignores_edges_and_uses_guess():
    img = np.zeros((60, 60)); img[29:32, 21:24] = 100.0; img[30, 22] = 150.0; img[1, 1] = 1e6
    img[44:47, 49:52] = 300.0; img[45, 50] = 400.0
    assert cal.find_star(img) == (50.0, 45.0)
    assert cal.find_star(img, guess=(20, 28), box=10) == (22.0, 30.0)


def test_stage_key_changes_only_with_calibration_settings():
    cfg = {"stage1": {"steps": {"jump": {"rejection_threshold": 6.0}}}, "stage2": {}, "stamp": {"half_size": 40},
           "photometry": {"annulus": [16, 30]}}
    k = cal.stage_key(cfg)
    assert cal.stage_key(dict(cfg, photometry={"annulus": [10, 20]})) == k
    assert cal.stage_key(dict(cfg, stamp={"half_size": 30})) != k


def test_process_saves_each_segment_survives_failures_and_resumes(tmp_path, monkeypatch):
    """One bad segment must not cost the rest; a rerun only redoes what is missing."""
    cfg = {"source": {"kind": "synthetic", "segments_per_visit": 4}, "visits": ["eclipse1"],
           "paths": {"data": str(tmp_path / "d"), "products": str(tmp_path / "p")},
           "stage1": {}, "stage2": {}, "stamp": {"half_size": 4}}
    calls, progress = [], []

    def fake_segment(cfg, rf, dry_run=False):
        calls.append(rf.segment)
        if rf.segment == 2 and len(calls) <= 4:
            raise RuntimeError("corrupt segment")
        out = cal.stamp_path(cfg, rf)
        np.savez(out, time=np.arange(3.0))
        cal.mark_cached(out, cal.stage_key(cfg))
        return out

    monkeypatch.setattr(cal, "process_segment", fake_segment)
    monkeypatch.setattr(cal.s01_fetch, "download", lambda rf, *a, **k: None)
    done = cal.process(cfg, continue_on_error=True, after_segment=lambda n, tot, failed: progress.append((n, list(failed))))
    assert len(done) == 3 and calls == [1, 2, 3, 4]
    assert progress[-1] == (3, ["synthetic-eclipse1-seg002_mirimage_uncal.fits"])
    with pytest.raises(RuntimeError):                     # default behaviour: stop at the first error
        calls.clear(); calls.extend([0, 0])               # (keeps segment 2 failing on this pass)
        cal.process(cfg)
    calls.clear(); calls.extend([0] * 5)                  # now segment 2 succeeds
    assert len(cal.process(cfg)) == 1 and calls[-1] == 2  # only the missing segment was redone
    assert cal.process(cfg) == []


def test_process_stops_inside_the_time_budget(tmp_path, monkeypatch):
    cfg = {"source": {"kind": "synthetic", "segments_per_visit": 5}, "visits": ["eclipse1"],
           "paths": {"data": str(tmp_path / "d"), "products": str(tmp_path / "p")},
           "stage1": {}, "stage2": {}, "stamp": {"half_size": 4}}
    import time as _time
    clock = [0.0]
    monkeypatch.setattr(_time, "time", lambda: clock[0])

    def slow_segment(cfg, rf, dry_run=False):
        clock[0] += 3600.0                                # every segment "takes" one hour
        return cal.stamp_path(cfg, rf)

    monkeypatch.setattr(cal, "process_segment", slow_segment)
    monkeypatch.setattr(cal.s01_fetch, "download", lambda rf, *a, **k: None)
    # before segment 3: 2 h + 1.5 x 1 h margin = 3.5 h <= 3.6 h, so it runs;
    # before segment 4: 3 h + 1.5 h = 4.5 h > 3.6 h, so it stops with 2 left
    assert len(cal.process(cfg, max_hours=3.6)) == 3

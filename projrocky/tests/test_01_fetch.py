"""Ingredient 01: the dry run lists the right files and never downloads."""
import csv

import pytest

from rockyeye import s01_fetch


@pytest.fixture
def cfg(tmp_path):
    man = tmp_path / "manifest.csv"
    rows = [("eclipse1", 1, "e1-seg001_mirimage_uncal.fits", 2_000_000_000),
            ("eclipse1", 2, "e1-seg002_mirimage_uncal.fits", 1_500_000_000),
            ("eclipse2", 1, "e2-seg001_mirimage_uncal.fits", 1_000_000_000)]
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "eclipse", "segment", "relative_path", "filename", "size_bytes", "sha256"])
        for v, s, fn, size in rows:
            w.writerow(["X", v, f"{s:03d}", f"Sims/{v}/{fn}", fn, size, "ab" * 32])
    return {"source": {"kind": "kaggle", "dataset": "owner/name", "manifest": str(man)},
            "paths": {"data": str(tmp_path / "data"), "products": str(tmp_path / "products")}}


def test_lists_all_files_in_order(cfg):
    files = s01_fetch.list_files(cfg)
    assert [f.filename for f in files] == ["e1-seg001_mirimage_uncal.fits", "e1-seg002_mirimage_uncal.fits",
                                           "e2-seg001_mirimage_uncal.fits"]
    assert files[0].url == "kaggle://owner/name/Sims/eclipse1/e1-seg001_mirimage_uncal.fits"


def test_visit_filter(cfg):
    assert [f.visit for f in s01_fetch.list_files(cfg, ["eclipse2"])] == ["eclipse2"]


def test_plan_sizes(cfg):
    p = s01_fetch.plan(cfg)
    assert p["n_files"] == 3
    assert p["total_gb"] == pytest.approx(4.5)
    assert p["per_visit"]["eclipse1"]["gb"] == pytest.approx(3.5)
    assert p["largest_file_gb"] == pytest.approx(2.0)
    assert "TOTAL" in s01_fetch.format_plan(p)


def test_dry_run_downloads_nothing(cfg, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("network touched during a dry run")
    monkeypatch.setattr(s01_fetch.requests, "get", boom)
    rf = s01_fetch.list_files(cfg)[0]
    s01_fetch.download(rf, dry_run=True)
    assert not (tmp_path / "data").exists()


def test_kaggle_path_uses_official_client(cfg, monkeypatch, tmp_path):
    """Kaggle files go through kagglehub with (dataset, path-inside-dataset) and are checksummed."""
    import hashlib, sys, types
    payload = b"ramp" * 100
    rf = s01_fetch.list_files(cfg)[0]
    rf.size_bytes, rf.sha256 = len(payload), hashlib.sha256(payload).hexdigest()
    calls = {}

    def dataset_download(handle, path=None):
        calls["args"] = (handle, path)
        got = tmp_path / "cache_copy.fits"
        got.write_bytes(payload)
        return str(got)
    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=dataset_download))
    out = s01_fetch.download(rf, min_free_gb=0.0)
    assert calls["args"] == ("owner/name", "Sims/eclipse1/e1-seg001_mirimage_uncal.fits")
    assert out.read_bytes() == payload


def test_refuses_when_disk_would_fill(cfg, monkeypatch):
    monkeypatch.setattr(s01_fetch, "free_gb", lambda p: 5.0)
    with pytest.raises(OSError, match="refusing"):
        s01_fetch.download(s01_fetch.list_files(cfg)[0], min_free_gb=8.0)


def test_resume_and_checksum(cfg, monkeypatch, tmp_path):
    """A partial .part file is continued with a Range request and verified by SHA-256."""
    import hashlib
    payload = bytes(range(256)) * 40
    rf = s01_fetch.list_files(cfg)[0]
    rf.url = "https://archive.example/file"          # the plain-HTTPS path (used for MAST)
    rf.size_bytes, rf.sha256 = len(payload), hashlib.sha256(payload).hexdigest()
    from pathlib import Path
    part = Path(rf.local_path + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:1000])
    seen = {}

    class Resp:
        status_code = 206
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield payload[1000:]

    def fake_get(url, headers=None, **k):
        seen["range"] = headers.get("Range")
        return Resp()
    monkeypatch.setattr(s01_fetch.requests, "get", fake_get)
    out = s01_fetch.download(rf, min_free_gb=0.0)
    assert seen["range"] == "bytes=1000-"
    assert out.read_bytes() == payload

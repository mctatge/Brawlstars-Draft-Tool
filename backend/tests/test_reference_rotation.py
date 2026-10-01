"""The reference API distinguishes the seasonal trio from all free Ranked brawlers."""
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from bsdraft.api import main as M, schemas as S
from bsdraft.data import reference as R


def test_reference_rotation_excludes_grants_detected_free_and_unpickable_ids(monkeypatch):
    berry, tara, meg, nori, shelly = (
        R.brawler_by_name(n).id for n in ("Berry", "Tara", "Meg", "Nori", "Shelly"))
    doc = {"active": {"brawlers": ["Berry", "Tara", "Meg"]},
           "grants": [{"brawler": "Nori", "since": "2026-08-25"}]}
    monkeypatch.setattr(R, "_ranked_boosted_doc", lambda: doc)
    monkeypatch.setattr(R, "_now_utc", lambda: datetime(2026, 8, 26, tzinfo=timezone.utc))
    pickable = tuple(b for b in R.pickable_brawlers() if b.id != meg)
    monkeypatch.setattr(R, "pickable_brawlers", lambda: pickable)
    monkeypatch.setattr(M, "_engine", SimpleNamespace(
        stats=SimpleNamespace(map_games={}, free_brawler_ids=(shelly,)), bracket_stats={}))

    response = TestClient(M.app).get("/api/reference")
    assert response.status_code == 200
    body = response.json()
    assert body["seasonal_boosted"] == [berry, tara]
    assert set(body["seasonal_boosted"]) <= {b["id"] for b in body["brawlers"]}
    # The existing eligibility/scoring union keeps its grants and inferred free brawlers.
    assert set(body["boosted"]) == {berry, tara, meg, nori, shelly}


def test_reference_schema_defaults_seasonal_rotation_to_empty():
    response = S.ReferenceResponse(brawlers=[], maps=[], modes=[])
    assert response.model_dump()["seasonal_boosted"] == []

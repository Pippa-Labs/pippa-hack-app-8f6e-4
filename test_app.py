"""Tests for Who Lived Here. Run them with `pytest`.

Publishing runs them, and a repository whose tests fail does not deploy.
No test touches the network (conftest.py refuses every socket): postcodes.io,
the only service the app ever asks, is replaced by a small fake. Everything
else the app knows ships beside it in plaques.json and places.json.
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import unicodedata
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from pippa_agentapp_sdk import LocalRunner

import app as app_module
from app import App, NoPlaceGiven, OutsideLondon, PlaceNotFound

ROOT = Path(__file__).resolve().parent
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
OUTCOME = "plaques.nearest"

#: Greater London's extremes (Enfield to Biggin Hill, Hillingdon to Havering),
#: with a hair of margin.
LONDON_LAT = (51.28, 51.70)
LONDON_LON = (-0.52, 0.34)

#: What postcodes.io answers for the Science Museum's postcode (trimmed).
SW7_2DD = {"status": 200, "result": {
    "postcode": "SW7 2DD", "outcode": "SW7", "incode": "2DD",
    "latitude": 51.497762, "longitude": -0.174600,
    "admin_district": "Kensington and Chelsea", "region": "London", "country": "England",
}}
#: A real postcode, but in Manchester.
M1_1AE = {"status": 200, "result": {
    "postcode": "M1 1AE", "outcode": "M1", "incode": "1AE",
    "latitude": 53.480930, "longitude": -2.237060,
    "admin_district": "Manchester", "region": "North West", "country": "England",
}}

NO_PLACE = ("Tell me a London street, station, landmark or postcode, "
            "and I'll find the three nearest blue plaques.")
FALLBACK_SW7 = "(I couldn't look up SW7 2DD itself, so I measured from the middle of SW7.)"


# ── helpers ──────────────────────────────────────────────────────────────────

def call(inputs: dict, *, explicit: bool = False, app: App | None = None):
    kwargs = {"granted_secrets": {}, "now": NOW, "request_id": "test-1"}
    if explicit:
        kwargs["explicit"] = True  # as when the person picks the app with @
    return LocalRunner(app or App()).call(OUTCOME, inputs, **kwargs).result


def normalise(text: str) -> str:
    """SPEC.md's normalise(), which the app and the data builder share."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['’`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def metres(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def plaque_prefix(name: str, who: str, address: str) -> str:
    """The line before its distance; an empty `who` (or, rarely, address) is left out."""
    return " — ".join(part for part in (name, who, address) if part)


def nearest_from(lat: float, lon: float, k: int = 3) -> list[tuple[float, str]]:
    """The k nearest plaques to a point, worked out here from plaques.json."""
    found = sorted((metres(lat, lon, p[3], p[4]), plaque_prefix(p[0], p[1], p[2]))
                   for p in load("plaques.json")["plaques"])
    return found[:k]


def assert_a_proper_answer(result, label: str | None = None, *, fallback: str = ""):
    """3 plaques, nearest first, every line in SPEC.md's format, and the narrative."""
    data = result.data
    assert set(data) == {"place", "plaques", "distances_m"}
    if label is not None:
        assert data["place"] == label
    lines, distances = data["plaques"], data["distances_m"]
    assert len(lines) == 3 and len(distances) == 3
    assert all(isinstance(d, int) and not isinstance(d, bool) for d in distances)
    assert distances == sorted(distances)
    assert all(d >= 0 and d % 10 == 0 for d in distances)
    known = {plaque_prefix(p[0], p[1], p[2]) for p in load("plaques.json")["plaques"]}
    for line, d in zip(lines, distances):
        assert line.endswith(f" — {d:,} m"), line
        assert line[: -len(f" — {d:,} m")] in known, line
    assert len(set(lines)) == 3
    # Pippa's reply comes from the narrative: it must hold the whole answer.
    assert result.narrative.startswith("The 3 blue plaques nearest ")
    # One paragraph, never empty, at most 500 characters, naming all three.
    narrative = result.narrative
    assert narrative and "\n" not in narrative and len(narrative) <= app_module.NARRATIVE_LIMIT == 500
    assert narrative.startswith(f"The 3 blue plaques nearest {data['place']}: 1. ")
    full = (f"The 3 blue plaques nearest {data['place']}: 1. {lines[0]}; 2. {lines[1]}; 3. {lines[2]}."
            + (f" {fallback}" if fallback else ""))
    if len(full) <= 500:
        assert narrative == full
    if fallback:
        assert narrative.endswith(fallback) or len(narrative) == 500
    else:
        assert "couldn't look up" not in narrative


def assert_measured_from(result, lat: float, lon: float):
    """The distances are the true distances to the true 3 nearest from (lat, lon)."""
    expected = nearest_from(lat, lon)
    distances = result.data["distances_m"]
    for got, (want, _) in zip(distances, expected):
        assert abs(got - want) <= 10, (distances, expected)
    for line, d in zip(result.data["plaques"], distances):
        prefix = line[: -len(f" — {d:,} m")]
        matches = [p for p in load("plaques.json")["plaques"] if plaque_prefix(p[0], p[1], p[2]) == prefix]
        assert any(abs(metres(lat, lon, p[3], p[4]) - d) <= 10 for p in matches), line


class Response:
    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def read(self, *args):
        return self.body

    def getcode(self):
        return 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def postcodes(monkeypatch):
    """Replace the request to postcodes.io; record what was asked; answer with `answer`.

    Set `raise` to an exception to make the request fail, or `raw` to bytes to
    answer with something that is not the JSON in `answer`.
    """
    seen = {"answer": SW7_2DD, "raw": None, "raise": None, "urls": [], "timeouts": [], "agents": []}

    def fake_urlopen(request, timeout=None, *args, **kwargs):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        seen["urls"].append(url)
        seen["timeouts"].append(timeout)
        if hasattr(request, "get_header"):
            seen["agents"].append(request.get_header("User-agent"))
        if seen["raise"] is not None:
            raise seen["raise"]
        body = seen["raw"] if seen["raw"] is not None else json.dumps(seen["answer"]).encode()
        return Response(body)

    monkeypatch.setattr(app_module.urllib.request, "urlopen", fake_urlopen)
    return seen


def postcode_urls(postcode: str) -> set[str]:
    """POSTCODES_API + the postcode, quoted (with or without its space)."""
    api = app_module.POSTCODES_API
    forms = {postcode, postcode.replace(" ", "")}
    return {api + urllib.parse.quote(f) for f in forms} | {api + urllib.parse.quote(f, safe="") for f in forms}


# ── the manifest ─────────────────────────────────────────────────────────────

def test_the_manifest_is_valid():
    app = App()
    assert app.manifest.app_id == "hack.app_8f6e_4"
    assert [o.id for o in app.manifest.outcomes] == [OUTCOME]

    raw = yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8"))
    assert raw["persistence"] == "none"
    assert raw["app"]["id"] == "hack.app_8f6e_4"
    assert raw["app"]["name"] == "Who Lived Here"
    assert raw["app_author_name"].strip()
    # The badge splits on the first "by": it may never appear in a summary.
    assert not re.search(r"\bby\b", raw["app"]["summary"], re.IGNORECASE)
    for outcome in raw["outcomes"]:
        assert not re.search(r"\bby\b", outcome["summary"], re.IGNORECASE)
    assert not {"app_secrets", "connectors", "secrets"} & set(raw)

    (outcome,) = raw["outcomes"]
    assert [i["name"] for i in outcome["inputs"]] == ["place"]
    place = outcome["inputs"][0]
    assert place["type"] == "string" and place["required"] is False
    outputs = {o["name"]: o for o in outcome["output"]}
    assert list(outputs) == ["place", "plaques", "distances_m"]
    assert outputs["place"]["type"] == "string"
    assert outputs["plaques"]["type"] == "array" and outputs["plaques"]["item_type"] == "string"
    assert outputs["distances_m"]["type"] == "array" and outputs["distances_m"]["item_type"] == "integer"


def test_the_constants_the_spec_names():
    assert app_module.POSTCODES_API == "https://api.postcodes.io/postcodes/"
    assert 0 < app_module.POSTCODE_TIMEOUT <= 3
    assert isinstance(app_module.USER_AGENT, str) and app_module.USER_AGENT.strip()


# ── a normal answer ──────────────────────────────────────────────────────────

def test_221b_baker_street(postcodes):
    result = call({"place": "221B Baker Street"})
    assert_a_proper_answer(result, "221B Baker Street")
    _, lat, lon, kind = load("places.json")["places"]["221b baker street"]
    assert kind == "address"
    assert_measured_from(result, lat, lon)
    assert result.data["distances_m"][0] < 1000
    assert postcodes["urls"] == []


@pytest.mark.parametrize("place, label, key", [
    ("South Kensington station", "South Kensington station", "south kensington"),
    ("South Kensington", "South Kensington station", "south kensington"),
    ("Bloomsbury", "Bloomsbury", "bloomsbury"),
    ("Doughty Street", "Doughty Street", "doughty street"),
    ("SW7", "the SW7 postcode district", "sw7"),
    ("sw7", "the SW7 postcode district", "sw7"),
])
def test_places_it_knows_and_their_labels(postcodes, place, label, key):
    result = call({"place": place})
    assert_a_proper_answer(result, label)
    _, lat, lon, _ = load("places.json")["places"][key]
    assert_measured_from(result, lat, lon)
    assert postcodes["urls"] == [], "only a full postcode may be sent anywhere"


def test_nearest_to_48_doughty_street_is_charles_dickens(postcodes):
    result = call({"place": "48 Doughty Street"})
    assert_a_proper_answer(result)
    assert "Doughty Street" in result.data["place"]
    assert result.data["plaques"][0].startswith("Charles Dickens — ")
    assert result.data["distances_m"][0] <= 50
    assert postcodes["urls"] == []


def test_picked_with_at_it_answers_the_same(postcodes):
    plain = call({"place": "221B Baker Street"})
    picked = call({"place": "221B Baker Street"}, explicit=True)
    assert picked.data == plain.data
    assert picked.narrative == plain.narrative
    with pytest.raises(NoPlaceGiven):
        call({}, explicit=True)
    with pytest.raises(PlaceNotFound):
        call({"place": "Narnia"}, explicit=True)


def test_the_same_question_gets_the_same_answer(postcodes):
    app = App()
    first = call({"place": "Bloomsbury"}, app=app)
    second = call({"place": "Bloomsbury"}, app=app)
    assert first.data == second.data and first.narrative == second.narrative


# ── nothing to go on ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("inputs", [{}, {"place": ""}, {"place": "   "}, {"place": "\n\t"}])
def test_no_place_is_said_plainly(postcodes, inputs):
    with pytest.raises(NoPlaceGiven) as caught:
        call(inputs)
    assert caught.value.user_message == NO_PLACE
    assert postcodes["urls"] == []


@pytest.mark.parametrize("text", ["Narnia", "asdfgh", "Hogwarts"])
def test_a_place_it_does_not_know_is_named_back(postcodes, text):
    with pytest.raises(PlaceNotFound) as caught:
        call({"place": text})
    assert caught.value.user_message == (
        f'I don\'t know where "{text}" is. Try a London street, station, landmark or postcode, '
        'like "Baker Street", "South Kensington station" or "SW7".'
    )
    assert postcodes["urls"] == []


@pytest.mark.parametrize("district", ["W1", "EC1", "WC1", "WC2", "SW1", "EC4"])
def test_the_split_central_districts_people_still_say(postcodes, district):
    """postcodes.io only knows W1A ... W1W; people say "W1". The app knows both."""
    result = call({"place": f"blue plaques near {district}"})
    assert result.data["place"] == f"the {district} postcode district"
    assert len(result.data["plaques"]) == 3 and result.data["distances_m"][0] < 1000
    assert postcodes["urls"] == []


@pytest.mark.parametrize("question, named", [
    ("who lived near Hogwarts?", "Hogwarts"),
    ("blue plaques near the Emerald City please", "Emerald City"),
    ("Is there a blue plaque near Narnia?", "Narnia"),
])
def test_a_refusal_names_the_place_not_the_whole_question(postcodes, question, named):
    """Pippa shows user_message word for word: it should say what to retype."""
    with pytest.raises(PlaceNotFound) as caught:
        call({"place": question})
    assert caught.value.user_message.startswith(f'I don\'t know where "{named}" is.')


# ── a full postcode: the one thing ever sent anywhere ────────────────────────

def test_a_full_postcode_is_looked_up(postcodes):
    result = call({"place": "SW7 2DD"})
    assert_a_proper_answer(result, "SW7 2DD")
    assert_measured_from(result, 51.497762, -0.174600)
    assert len(postcodes["urls"]) == 1
    assert postcodes["urls"][0] in postcode_urls("SW7 2DD")
    assert postcodes["urls"][0].startswith(app_module.POSTCODES_API)
    assert postcodes["timeouts"][0] is not None
    assert 0 < postcodes["timeouts"][0] <= app_module.POSTCODE_TIMEOUT <= 3
    if postcodes["agents"]:
        assert postcodes["agents"][0] == app_module.USER_AGENT


def test_a_postcode_written_carelessly_is_still_a_postcode(postcodes):
    result = call({"place": "sw72dd"})
    assert_a_proper_answer(result, "SW7 2DD")
    assert len(postcodes["urls"]) == 1 and postcodes["urls"][0] in postcode_urls("SW7 2DD")


def test_the_fetch_seam_on_the_instance():
    app = App()
    asked = []

    def fetch(url, timeout=None):
        asked.append((url, timeout))
        return SW7_2DD

    app._fetch = fetch
    result = call({"place": "SW7 2DD"}, app=app)
    assert_a_proper_answer(result, "SW7 2DD")
    assert len(asked) == 1
    url, timeout = asked[0]
    assert url in postcode_urls("SW7 2DD")
    assert timeout is not None and timeout <= app_module.POSTCODE_TIMEOUT


def _http_error(code):
    return app_module.urllib.error.HTTPError(
        app_module.POSTCODES_API + "SW7%202DD", code, "no", {},
        io.BytesIO(json.dumps({"status": code, "error": "Postcode not found"}).encode()),
    )


@pytest.mark.parametrize("problem", [
    TimeoutError("timed out"),
    OSError("connection reset"),
    "urlerror",
    "http404",
    "http500",
    "http503",
], ids=["timeout", "oserror", "urlerror", "404", "500", "503"])
def test_postcodes_io_failing_falls_back_to_the_district(postcodes, problem):
    if problem == "urlerror":
        problem = app_module.urllib.error.URLError("Name or service not known")
    elif isinstance(problem, str):
        problem = _http_error(int(problem[4:]))
    postcodes["raise"] = problem
    result = call({"place": "SW7 2DD"})
    assert_a_proper_answer(result, "the SW7 postcode district", fallback=FALLBACK_SW7)
    _, lat, lon, _ = load("places.json")["places"]["sw7"]
    assert_measured_from(result, lat, lon)
    assert len(postcodes["urls"]) == 1, "one try only: there is no time for a second"


@pytest.mark.parametrize("raw", [
    b"<html><body>502 Bad Gateway</body></html>",
    b"",
    b"null",
    b"[]",
    b'"SW7 2DD"',
    json.dumps({"status": 404, "error": "Postcode not found"}).encode(),
    json.dumps({"status": 200, "result": None}).encode(),
    json.dumps({"status": 200, "result": {"postcode": "SW7 2DD"}}).encode(),
    json.dumps({"status": 200, "result": {"postcode": "SW7 2DD", "latitude": None, "longitude": None}}).encode(),
    json.dumps({"status": 200, "result": {"postcode": "SW7 2DD", "latitude": "north", "longitude": "west"}}).encode(),
], ids=["html", "empty", "null", "list", "string", "error-json", "null-result", "no-coordinates",
        "null-coordinates", "junk-coordinates"])
def test_an_answer_without_coordinates_falls_back_to_the_district(postcodes, raw):
    postcodes["raw"] = raw
    result = call({"place": "SW7 2DD"})
    assert_a_proper_answer(result, "the SW7 postcode district", fallback=FALLBACK_SW7)


@pytest.mark.parametrize("problem", [TimeoutError("timed out"), OSError("reset"), ValueError("not JSON")])
def test_the_instance_seam_failing_falls_back_too(problem):
    app = App()

    def fetch(url, timeout=None):
        raise problem

    app._fetch = fetch
    result = call({"place": "SW7 2DD"}, app=app)
    assert_a_proper_answer(result, "the SW7 postcode district", fallback=FALLBACK_SW7)


def test_a_plaque_with_no_address_leaves_the_gap_out(postcodes):
    """Measured from right on top of a plaque that has no address: no empty " —  — "."""
    bare = [p for p in load("plaques.json")["plaques"] if not p[2].strip()]
    if not bare:
        pytest.skip("every plaque has an address")
    name, who, _, lat, lon = bare[0]
    postcodes["answer"] = {"status": 200, "result": {"postcode": "SE3 9AA", "latitude": lat, "longitude": lon}}
    result = call({"place": "SE3 9AA"})
    assert_a_proper_answer(result, "SE3 9AA")
    assert result.data["distances_m"][0] == 0
    assert any(line == plaque_prefix(name, who, "") + " — 0 m" for line in result.data["plaques"])
    assert all(" —  — " not in line and " — — " not in line for line in result.data["plaques"])


def test_a_postcode_outside_london(postcodes):
    postcodes["answer"] = M1_1AE
    with pytest.raises(OutsideLondon) as caught:
        call({"place": "M1 1AE"})
    assert caught.value.user_message == "M1 1AE is outside London, and I only know London's blue plaques."
    assert len(postcodes["urls"]) == 1 and postcodes["urls"][0] in postcode_urls("M1 1AE")


@pytest.mark.parametrize("place", [
    "221B Baker Street", "Baker Street", "South Kensington station", "Bloomsbury", "SW7", "NW3",
    "Doughty Street", "Narnia",
])
def test_nothing_but_a_full_postcode_goes_over_the_network(postcodes, place):
    try:
        call({"place": place})
    except PlaceNotFound:
        pass
    assert postcodes["urls"] == []


def test_the_whole_call_fits_inside_pippas_20_seconds(monkeypatch):
    tries = []

    def hang(request, timeout=None, *args, **kwargs):
        tries.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(app_module.urllib.request, "urlopen", hang)
    started = time.perf_counter()
    result = call({"place": "SW7 2DD"})
    elapsed = time.perf_counter() - started
    assert_a_proper_answer(result, "the SW7 postcode district", fallback=FALLBACK_SW7)
    assert tries and all(t is not None and 0 < t <= app_module.POSTCODE_TIMEOUT for t in tries)
    assert sum(tries) <= 3          # the worst case: postcodes.io hangs to its timeout
    assert elapsed < 5              # everything else (loading data, sorting) is quick


# ── the data files ───────────────────────────────────────────────────────────

def test_the_data_files_fit_inside_the_package():
    size = (ROOT / "plaques.json").stat().st_size + (ROOT / "places.json").stat().st_size
    assert size < 2.5 * 1024 * 1024


def test_every_plaque_is_a_london_plaque_with_a_name():
    data = load("plaques.json")
    assert data["fields"] == ["name", "who", "address", "lat", "lon"]
    assert data["source"] and data["licence"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["downloaded"])
    plaques = data["plaques"]
    assert data["count"] == len(plaques) > 500
    for p in plaques:
        assert len(p) == 5, p
        name, who, address, lat, lon = p
        assert isinstance(name, str) and name.strip(), p
        assert isinstance(who, str), p
        assert isinstance(address, str), p
        assert " — " not in name + who + address, p
        assert LONDON_LAT[0] <= lat <= LONDON_LAT[1] and LONDON_LON[0] <= lon <= LONDON_LON[1], p
    assert len({(p[0], p[2]) for p in plaques}) == len(plaques), "deduped"
    # A handful of Open Plaques records have no address; the line then leaves it out.
    assert sum(1 for p in plaques if not p[2].strip()) <= len(plaques) // 100
    assert any(p[0] == "Charles Dickens" and "48 Doughty Street" in p[2] for p in plaques)


def test_every_place_is_keyed_the_way_the_app_looks_it_up():
    data = load("places.json")
    assert data["fields"] == ["name", "lat", "lon", "kind"]
    places = data["places"]
    assert data["count"] == len(places)
    kinds = {"address", "landmark", "station", "area", "district", "street"}
    for key, value in places.items():
        assert key == normalise(key), key
        assert len(value) == 4, (key, value)
        name, lat, lon, kind = value
        assert isinstance(name, str) and name.strip(), key
        assert kind in kinds, (key, kind)
        assert 51.2 <= lat <= 51.75 and -0.6 <= lon <= 0.4, (key, value)
        if kind == "station":
            assert not key.endswith(" station") and name.endswith(" station"), (key, name)
        if kind == "district":
            assert re.fullmatch(r"[a-z]{1,2}\d[a-z\d]?", key) and name == key.upper(), (key, name)
    assert places["221b baker street"][0] == "221B Baker Street"
    assert places["south kensington"][3] == "station"
    assert places["sw7"][3] == "district" and places["nw3"][3] == "district"
    assert places["doughty street"][3] == "street"
    assert {"soho", "bloomsbury", "kings cross"} <= set(places)


def test_the_app_and_the_data_normalise_alike():
    if not hasattr(app_module, "normalise"):
        pytest.skip("app.py keeps its normalise private")
    for text in ["King's Cross", "St James's Park", "Café Royal", "Elephant & Castle",
                 "  221B,  Baker Street ", "SW7 2DD", "Hampstead–Heath"]:
        assert app_module.normalise(text) == normalise(text)


def test_a_long_answer_is_still_one_short_paragraph(postcodes, monkeypatch):
    """However long the plaque texts, the narrative stays one paragraph of <= 500 characters."""
    long = tuple(("A" * 120, "B" * 150, "C" * 140, 51.5 + i / 10000, -0.1) for i in range(5))
    monkeypatch.setattr(app_module, "_plaques", lambda: long)
    result = call({"place": "Soho"})
    assert result.narrative and "\n" not in result.narrative and len(result.narrative) <= 500
    assert result.narrative.startswith("The 3 blue plaques nearest Soho: 1. ")

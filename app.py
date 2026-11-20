"""Who Lived Here: the three London blue plaques nearest a place.

Somebody asks "who lived near 221B Baker Street?" or "blue plaques near South
Kensington station". Pippa pulls the place out of the message (the manifest's
`place` input) and calls this app, which answers with the three nearest blue
plaques: who they commemorate, one line on who that was, the plaque's address
and how far away it is in metres.

Everything it needs ships beside this file, so a normal answer touches no
network at all:

    plaques.json   London's blue plaques, from Open Plaques (openplaques.org)
    places.json    stations, landmarks, neighbourhoods, postcode districts,
                   well-known addresses and streets, with their coordinates

The one exception is a full postcode ("SW7 2DD"): that is looked up on
postcodes.io, with a 3-second limit, and if that fails for any reason the app
measures from the middle of the postcode district ("SW7") instead. No key is
needed for anything, and nothing is kept between calls.

The shape never changes, whatever the app does (see the template,
Pippa-Labs/pippa-hack-space-photo):

    class App:
        manifest  -> the parsed manifest.yaml sitting next to this file
        handle(call) -> Result
"""

from __future__ import annotations

import difflib
import heapq
import json
import math
import re
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    Result,
    load_manifest_file,
    validate_manifest,
)

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest.yaml"
#: Data files live beside app.py: Pippa does not deploy scripts/ or docs/.
PLAQUES_PATH = HERE / "plaques.json"
PLACES_PATH = HERE / "places.json"

OUTCOME = "plaques.nearest"
HOW_MANY = 3

#: The only place the app ever sends anything, and only a full postcode.
POSTCODES_API = "https://api.postcodes.io/postcodes/"
#: postcodes.io usually answers in well under a second. Three seconds, then the
#: postcode district instead: the whole call stays far inside Pippa's 20.
POSTCODE_TIMEOUT = 3
USER_AGENT = "pippa-agent-app-who-lived-here/0.1 (https://github.com/Pippa-Labs)"

#: Greater London, generously: (south, north, west, east).
LONDON_BOX = (51.28, 51.70, -0.52, 0.34)
EARTH_RADIUS_M = 6_371_008.8

#: A full UK postcode anywhere in the text: outward code, then inward code.
_FULL_POSTCODE = re.compile(r"\b([A-Z]{1,2}[0-9][A-Z0-9]?)\s*([0-9][A-Z]{2})\b")
_OUTCODE = re.compile(r"[a-z]{1,2}[0-9][a-z0-9]?")

#: Words people wrap a place in ("who lived near ...", "blue plaques around ...").
#: Stripped from the front of what arrives, so the app copes when the whole
#: question comes through as the place, not only the place itself.
_LEADING = frozenset("""
    who whom whose which what where lived live lives living used to has had have
    near nearby nearest closest close around round by next beside at on in of
    from the a an any some is are was were there here blue plaque plaques
    historic historical heritage english famous notable people person persons
    former residents resident inhabitants show me find list tell give about
    please can you could i want know see look up for postcode postal code
    district area
""".split())
#: Words people put after a place ("... station", "... in London").
_TRAILING = frozenset("""
    please london uk england station stations tube underground rail railway
    overground dlr area district postcode here then now in at on of
""".split())
#: Station suffixes a person may type; station keys in places.json have none.
_STATION_WORDS = ("tube station", "underground station", "rail station", "railway station",
                  "overground station", "dlr station", "station", "tube")
#: Abbreviations at the end of a street name.
_ABBREVIATIONS = {"st": "street", "rd": "road", "sq": "square", "ave": "avenue",
                  "gdns": "gardens", "pl": "place", "ln": "lane", "terr": "terrace",
                  "cres": "crescent", "pk": "park"}
#: Single words too ordinary to stand for a place on their own when the app is
#: hunting for a place name inside a longer sentence.
_GENERIC = frozenset("""
    street road park square hill green house bridge gardens place lane church
    court castle palace market common heath cross gate wood end town city north
    south east west central london high old new great little upper lower museum
    station circus row walk way mews yard terrace avenue crescent
""".split())


def normalise(text: str) -> str:
    """The key form shared with places.json: ASCII, lower case, words only."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['’`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


class NoPlaceGiven(AgentAppError):
    """The message named no place."""

    user_message = ("Tell me a London street, station, landmark or postcode, "
                    "and I'll find the three nearest blue plaques.")
    retryable = False


class PlaceNotFound(AgentAppError):
    """The place is not in places.json and is not a postcode we could look up."""

    user_message = ("I don't know where that is. Try a London street, station, landmark or "
                    "postcode, like \"Baker Street\", \"South Kensington station\" or \"SW7\".")
    retryable = False


class OutsideLondon(AgentAppError):
    """The place was found, but it is outside Greater London."""

    user_message = "That place is outside London, and I only know London's blue plaques."
    retryable = False


@lru_cache(maxsize=1)
def _plaques() -> tuple:
    """Every plaque as (name, who, address, lat, lon). Read once, never written."""
    data = json.loads(PLAQUES_PATH.read_text(encoding="utf-8"))
    rows = []
    for row in data["plaques"]:
        name, who, address, lat, lon = row
        rows.append((str(name), str(who or ""), str(address or ""), float(lat), float(lon)))
    return tuple(rows)


@lru_cache(maxsize=1)
def _places() -> dict:
    """normalised name -> (display name, lat, lon, kind). Read once, never written."""
    data = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
    return {key: (str(v[0]), float(v[1]), float(v[2]), str(v[3])) for key, v in data["places"].items()}


@lru_cache(maxsize=1)
def _place_keys() -> tuple:
    return tuple(_places())


def _in_london(lat: float, lon: float) -> bool:
    south, north, west, east = LONDON_BOX
    return south <= lat <= north and west <= lon <= east


def _metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


class App:
    """Find a place, then the three blue plaques nearest it."""

    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def handle(self, call: Call) -> Result:
        if call.outcome_id != OUTCOME:
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")

        text = " ".join(str(call.inputs.get("place") or "").split())
        if not _core_words(normalise(text)):
            # Nothing, or only carrier words: "who lived here?", "blue plaques near me".
            raise NoPlaceGiven("no place in the message")

        label, lat, lon, note = self._locate(text)
        if not _in_london(lat, lon):
            error = OutsideLondon(f"{label} is at {lat:.4f},{lon:.4f}, outside London")
            error.user_message = f"{label} is outside London, and I only know London's blue plaques."
            raise error

        nearest = heapq.nsmallest(
            HOW_MANY, ((_metres(lat, lon, p[3], p[4]), p) for p in _plaques()), key=lambda pair: pair[0],
        )
        entries, lines, distances = [], [], []
        for metres, (name, who, address, _lat, _lon) in nearest:
            rounded = int(round(metres / 10.0)) * 10
            entries.append((name, who, address, rounded))
            lines.append(_line(name, who, address, rounded))
            distances.append(rounded)

        narrative = _narrative(label, entries, note)
        return Result(data={"place": label, "plaques": lines, "distances_m": distances}, narrative=narrative)

    # ── Finding the place ────────────────────────────────────────────────────

    def _locate(self, text: str) -> tuple[str, float, float, str]:
        """(label, lat, lon, note) for what the person wrote, or PlaceNotFound."""
        postcode = _FULL_POSTCODE.search(text.upper())
        if postcode:
            return self._locate_postcode(text, postcode.group(1), postcode.group(2))

        found = self._lookup(normalise(text))
        if found is None:
            error = PlaceNotFound(f"no place matches {text!r}")
            error.user_message = (
                f"I don't know where \"{_as_written(text)}\" is. Try a London street, station, landmark or "
                "postcode, like \"Baker Street\", \"South Kensington station\" or \"SW7\"."
            )
            raise error
        display, lat, lon, kind = found
        return self._label(display, kind), lat, lon, ""

    def _locate_postcode(self, text: str, outward: str, inward: str) -> tuple[str, float, float, str]:
        postcode = f"{outward} {inward}"
        coords = self._postcode(postcode)
        if coords is not None:
            return postcode, coords[0], coords[1], ""
        district = _places().get(outward.lower())
        if district is None or district[3] != "district":
            error = PlaceNotFound(f"postcodes.io could not place {postcode} and {outward} is not a London district")
            error.user_message = (
                f"I couldn't look up the postcode {postcode}, and {outward} isn't a London postcode "
                "district I know. Try a London street, station or landmark instead."
            )
            raise error
        display, lat, lon, kind = district
        note = f"(I couldn't look up {postcode} itself, so I measured from the middle of {display}.)"
        return self._label(display, kind), lat, lon, note

    def _postcode(self, postcode: str) -> tuple[float, float] | None:
        """postcodes.io's coordinates for a full postcode, or None.

        Never raises and never takes longer than POSTCODE_TIMEOUT: urlopen's
        timeout does not cover the DNS lookup, so the request runs in a thread
        that is simply abandoned if it is still going when time is up.
        """
        url = POSTCODES_API + urllib.parse.quote(postcode, safe="")
        box: dict = {}

        def ask() -> None:
            try:
                box["body"] = self._fetch(url, POSTCODE_TIMEOUT)
            except Exception as exc:  # any failure at all means: use the district
                box["error"] = type(exc).__name__

        worker = threading.Thread(target=ask, daemon=True)
        worker.start()
        worker.join(POSTCODE_TIMEOUT)
        body = box.get("body")
        if not isinstance(body, dict):
            return None
        result = body.get("result")
        if not isinstance(result, dict):
            return None
        try:
            lat, lon = float(result["latitude"]), float(result["longitude"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return lat, lon

    def _fetch(self, url: str, timeout: float) -> dict:
        """The one network call: GET `url`, parse JSON. Tests replace this."""
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _label(display: str, kind: str) -> str:
        if kind == "district":
            return f"the {display} postcode district"
        return display

    @staticmethod
    def _lookup(key: str):
        """Find a place for normalised text, most exact reading first."""
        places = _places()
        words = _core_words(key)
        if not words:
            return None

        # 1. The whole remainder, then with trailing words peeled off one at a time.
        tried: list[str] = []
        tail = list(words)
        while tail:
            tried.extend(_variants(" ".join(tail)))
            if tail[-1] not in _TRAILING:
                break
            tail = tail[:-1]
        for candidate in tried:
            if candidate in places:
                return places[candidate]
            if "the " + candidate in places:
                return places["the " + candidate]

        # 2. The longest run of words that names a place ("... near the British Museum please").
        for size in range(len(words) - 1, 0, -1):
            for start in range(0, len(words) - size + 1):
                span = words[start:start + size]
                if size == 1 and (span[0] in _GENERIC or span[0] in _LEADING or len(span[0]) < 4):
                    continue
                for candidate in _variants(" ".join(span)):
                    if candidate in places and not (size == 1 and places[candidate][3] == "street"):
                        return places[candidate]

        # 3. A near miss, for typing slips ("Picadilly Circus").
        for candidate in dict.fromkeys(tried):
            if len(candidate) < 5 or _OUTCODE.fullmatch(candidate):
                continue
            close = difflib.get_close_matches(candidate, _place_keys(), n=1, cutoff=0.85)
            if close:
                return places[close[0]]
        return None


#: Pippa's reply is written from the narrative: one paragraph, never longer.
NARRATIVE_LIMIT = 500


def _line(name: str, who: str, address: str, metres: int) -> str:
    """"person — who they were — address — N m", leaving out any empty part."""
    return " — ".join([name] + ([who] if who else []) + ([address] if address else []) + [f"{metres:,} m"])


def _narrative(label: str, entries: list, note: str) -> str:
    """One paragraph of at most NARRATIVE_LIMIT characters, never empty.

    Full lines if they fit; otherwise without "who they were"; otherwise just
    names and distances; and as a last resort cut at the limit.
    """
    def build(describe) -> str:
        items = "; ".join(f"{i}. {describe(e)}" for i, e in enumerate(entries, start=1))
        text = f"The {len(entries)} blue plaques nearest {label}: {items}." + (f" {note}" if note else "")
        return " ".join(text.split())  # one paragraph: no newlines, whatever the data holds

    text = ""
    for describe in (lambda e: _line(*e),
                     lambda e: _line(e[0], "", e[2], e[3]),
                     lambda e: f"{e[0]}, {e[3]:,} m"):
        text = build(describe)
        if len(text) <= NARRATIVE_LIMIT:
            return text
    return text[: NARRATIVE_LIMIT - 1].rstrip() + "…"


def _core_words(key: str) -> list[str]:
    """The words of normalised text after the carrier words in front of the place."""
    words = key.split()
    while words and words[0] in _LEADING:
        words = words[1:]
    return words


def _as_written(text: str) -> str:
    """The place as the person typed it, without the question around it.

    "who lived near Hogwarts?" -> "Hogwarts", so a refusal names what to retype.
    """
    words = text.split()
    while words and (normalise(words[0]) in _LEADING or not normalise(words[0])):
        words = words[1:]
    while words and normalise(words[-1]) in ("please", ""):
        words = words[:-1]
    shown = " ".join(words).rstrip("?!.,;: ")
    return shown or text


def _variants(text: str) -> list[str]:
    """`text`, without a station suffix, and with street abbreviations spelt out."""
    out = [text]
    for suffix in _STATION_WORDS:
        if text.endswith(" " + suffix):
            out.append(text[: -len(suffix) - 1])
            break
    words = text.split()
    if words:
        expanded = ["st" if w == "saint" else w for w in words]
        if expanded[-1] in _ABBREVIATIONS and len(expanded) > 1:
            expanded[-1] = _ABBREVIATIONS[expanded[-1]]
        joined = " ".join(expanded)
        if joined != text:
            out.append(joined)
    return out

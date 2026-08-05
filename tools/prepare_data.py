"""Turn public datasets into small, kid-readable CSVs in ``data/``.

Run this once; the CSVs it writes are committed to the repo, so nobody following the
course ever needs a network connection or a password.

    ./run.sh build-data          (or)      uv run python tools/prepare_data.py

Sources (all openly licensed):

* **Penguins** — Palmer Archipelago penguin data, Gorman, Williams & Fraser (2014),
  released CC0. Mirrored by the seaborn-data repository.
* **Mushrooms** — UCI Machine Learning Repository "Mushroom" data set (Schlimmer, 1987),
  CC BY 4.0.
* **Bikes** — UCI "Bike Sharing" data set (Fanaee-T & Gama, 2013), CC BY 4.0.
* **Names** — first names derived from US Social Security Administration records,
  a US government work in the public domain.

The two remaining tables (``creatures`` and ``monsters``) are entirely made up by us,
on purpose: chapter 5 needs a table small enough to solve with a pencil, and chapter 9
needs one where we know the true rule so we can check whether the model found it.
"""

from __future__ import annotations

import io
import os
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
MUSHROOM_URL = "https://archive.ics.uci.edu/static/public/73/mushroom.zip"
BIKES_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"


def fetch(url: str) -> bytes:
    """Download a URL, coping with NixOS's non-standard certificate location."""
    ctx = ssl.create_default_context()
    for candidate in (os.environ.get("SSL_CERT_FILE"), "/etc/ssl/certs/ca-bundle.crt"):
        if candidate and Path(candidate).exists():
            ctx.load_verify_locations(candidate)
            break
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
        return r.read()


# ---------------------------------------------------------------------------


def build_penguins() -> None:
    """Three penguin species, described by four body measurements."""
    df = pd.read_csv(io.BytesIO(fetch(PENGUINS_URL)))
    df = df.rename(
        columns={
            "bill_length_mm": "beak_length_mm",
            "bill_depth_mm": "beak_depth_mm",
            "flipper_length_mm": "flipper_length_mm",
            "body_mass_g": "weight_g",
        }
    )
    df = df[["species", "island", "beak_length_mm", "beak_depth_mm", "flipper_length_mm", "weight_g", "sex"]]
    df["sex"] = df["sex"].str.lower()
    df.to_csv(DATA / "penguins.csv", index=False)
    print(f"  penguins.csv  {len(df)} rows")


MUSHROOM_COLUMNS = [
    "edible", "cap_shape", "cap_surface", "cap_color", "has_bruises", "smell",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape",
    "stalk_root", "stalk_surface_above", "stalk_surface_below", "stalk_color_above",
    "stalk_color_below", "veil_type", "veil_color", "ring_number", "ring_type",
    "spore_color", "population", "habitat",
]

# The raw file uses single letters. Kids should not have to decode a legend, so we
# spell every value out. Only the columns the chapter actually uses are kept.
MUSHROOM_DECODE = {
    "edible": {"e": "edible", "p": "poisonous"},
    "cap_shape": {"b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken"},
    "cap_surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap_color": {"n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green",
                  "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow"},
    "has_bruises": {"t": "yes", "f": "no"},
    "smell": {"a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul",
              "m": "musty", "n": "none", "p": "pungent", "s": "spicy"},
    "gill_size": {"b": "broad", "n": "narrow"},
    "gill_color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray",
                   "r": "green", "o": "orange", "p": "pink", "u": "purple", "e": "red",
                   "w": "white", "y": "yellow"},
    "stalk_shape": {"e": "enlarging", "t": "tapering"},
    "ring_number": {"n": "none", "o": "one", "t": "two"},
    "ring_type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
                  "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore_color": {"k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green",
                    "o": "orange", "u": "purple", "w": "white", "y": "yellow"},
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered",
                   "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths",
                "u": "urban", "w": "waste", "d": "woods"},
}


def build_mushrooms() -> None:
    """Edible or poisonous? Every value spelled out in plain English."""
    blob = fetch(MUSHROOM_URL)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read("agaricus-lepiota.data").decode()
    df = pd.read_csv(io.StringIO(raw), header=None, names=MUSHROOM_COLUMNS)
    df = df[list(MUSHROOM_DECODE)]
    for col, mapping in MUSHROOM_DECODE.items():
        df[col] = df[col].map(mapping)
    df = df.dropna()
    df.to_csv(DATA / "mushrooms.csv", index=False)
    print(f"  mushrooms.csv {len(df)} rows, {df.shape[1]} columns")


def build_bikes() -> None:
    """How many bikes were rented today, given the weather? One row per day."""
    blob = fetch(BIKES_URL)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        df = pd.read_csv(io.BytesIO(z.read("day.csv")))

    weather = {1: "clear", 2: "misty", 3: "light_rain", 4: "storm"}
    season = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}
    weekday = {0: "sunday", 1: "monday", 2: "tuesday", 3: "wednesday",
               4: "thursday", 5: "friday", 6: "saturday"}

    out = pd.DataFrame(
        {
            "date": df["dteday"],
            "season": df["season"].map(season),
            "month": df["mnth"],
            "weekday": df["weekday"].map(weekday),
            "is_holiday": df["holiday"].map({0: "no", 1: "yes"}),
            "is_workday": df["workingday"].map({0: "no", 1: "yes"}),
            "weather": df["weathersit"].map(weather),
            # The raw file stores these scaled 0..1; put them back into real units.
            "temp_c": (df["temp"] * (39 - (-8)) + (-8)).round(1),
            "feels_like_c": (df["atemp"] * (50 - (-16)) + (-16)).round(1),
            "humidity_pct": (df["hum"] * 100).round(0).astype(int),
            "wind_kmh": (df["windspeed"] * 67).round(1),
            "rentals": df["cnt"],
        }
    )
    out.to_csv(DATA / "bikes.csv", index=False)
    print(f"  bikes.csv     {len(out)} rows")


def build_names(keep: int = 10000) -> None:
    """A pile of first names for the letter-by-letter language models."""
    text = fetch(NAMES_URL).decode()
    names = [n.strip().lower() for n in text.splitlines() if n.strip()]
    names = [n for n in names if n.isalpha() and 2 <= len(n) <= 12][:keep]
    (DATA / "names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"  names.txt     {len(names)} names")


# ---------------------------------------------------------------------------
# Tables we invent ourselves
# ---------------------------------------------------------------------------


def build_creatures() -> None:
    """Ten made-up creatures. Chapter 5 builds a decision tree on these by hand.

    Small on purpose: ten rows, four yes/no columns, and a rule a person can spot.
    """
    rows = [
        # name,        has_wings, bigger_than_cat, has_feathers, lives_in_water, can_fly
        ("sparrow",    "yes", "no",  "yes", "no",  "yes"),
        ("eagle",      "yes", "yes", "yes", "no",  "yes"),
        ("penguin",    "yes", "yes", "yes", "yes", "no"),
        ("ostrich",    "yes", "yes", "yes", "no",  "no"),
        ("bat",        "yes", "no",  "no",  "no",  "yes"),
        ("bumblebee",  "yes", "no",  "no",  "no",  "yes"),
        ("cat",        "no",  "no",  "no",  "no",  "no"),
        ("elephant",   "no",  "yes", "no",  "no",  "no"),
        ("dolphin",    "no",  "yes", "no",  "yes", "no"),
        ("goldfish",   "no",  "no",  "no",  "yes", "no"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["name", "has_wings", "bigger_than_cat", "has_feathers", "lives_in_water", "can_fly"],
    )
    df.to_csv(DATA / "creatures.csv", index=False)
    print(f"  creatures.csv {len(df)} rows")


def build_monsters(n: int = 800, seed: int = 7) -> None:
    """Made-up trading-card monsters. Is this one a *boss*?

    We invent these rather than borrow a real game's data, which means we know the true
    rule and can check whether the model rediscovered it. The secret rule is:

        boss  =  (attack + magic) > 150   AND   speed < 90

    ...plus a sprinkle of noise, so a perfect score is impossible and the kids get to
    meet the idea that some data is just messy.
    """
    rng = np.random.default_rng(seed)

    elements = np.array(["fire", "water", "grass", "rock", "ghost"])
    homes = np.array(["forest", "cave", "ocean", "sky", "volcano"])

    attack = rng.integers(20, 130, n)
    defense = rng.integers(20, 130, n)
    magic = rng.integers(10, 140, n)
    speed = rng.integers(20, 140, n)
    height_cm = rng.integers(15, 320, n)
    weight_kg = (height_cm * rng.uniform(0.15, 0.9, n)).round(1)

    is_boss = ((attack + magic) > 150) & (speed < 90)
    flip = rng.random(n) < 0.05  # 5% of the labels are wrong on purpose. Real data is like that.
    is_boss = np.where(flip, ~is_boss, is_boss)

    df = pd.DataFrame(
        {
            "name": [f"monster_{i:03d}" for i in range(n)],
            "element": rng.choice(elements, n),
            "home": rng.choice(homes, n),
            "attack": attack,
            "defense": defense,
            "magic": magic,
            "speed": speed,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "is_boss": np.where(is_boss, "yes", "no"),
        }
    )
    df.to_csv(DATA / "monsters.csv", index=False)
    print(f"  monsters.csv  {len(df)} rows ({(df.is_boss == 'yes').mean():.0%} bosses)")


BUILDERS = {
    "penguins": build_penguins,
    "mushrooms": build_mushrooms,
    "bikes": build_bikes,
    "names": build_names,
    "creatures": build_creatures,
    "monsters": build_monsters,
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(BUILDERS)
    print("Building datasets into", DATA)
    for name in wanted:
        if name not in BUILDERS:
            print(f"  ! unknown dataset {name!r}", file=sys.stderr)
            return 1
        BUILDERS[name]()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

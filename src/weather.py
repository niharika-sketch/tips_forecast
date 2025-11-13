# weather.py (v4)
# ------------------------------------------------------------
# Fetch daily climate (ECCC) by city with robust station selection:
# 1) Try env_canada historical station search (nearest + coverage validation)
# 2) Fallback to GeoMet climate-stations with expanding radius
# 3) Validate with ecc_count_valid_days (2024–2025)
# 4) Fetch daily CSV for years and normalize to:
#       date, temp_max, temp_min, temp_mean, precip_mm
#
# Outputs:
#   ../data/weather/weather_<city>.csv
#   ../data/weather_fetch_summary.csv
#   ../data/city_station_map.csv
#
# Requires: env-canada, aiohttp, geopy, pandas, numpy, requests
# ------------------------------------------------------------

import os
import io
import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import aiohttp
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic

from env_canada import ECHistorical
# optional import: newer env_canada exposes this helper
try:
    from env_canada.ec_historical import get_historical_stations
except Exception:
    get_historical_stations = None

THIS_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(THIS_DIR, "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "weather")
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

# GeoMet
GEOMET_BASE = "https://api.weather.gc.ca/collections/climate-stations/items"

# Major-city overrides (stable station IDs)
STATION_OVERRIDES = {
    "toronto": 6158350,        # Toronto City (official climate)
    "vancouver": 1108447,      # Vancouver Intl Airport
    "calgary": 2204,           # Calgary Intl
    "ottawa": 4346,            # Ottawa CDA
    "halifax": 6357,           # Halifax Stanfield
    "st. john's": 50089,       # St. John's Intl
    # Region-specific overrides (you requested these)
    "kitchener": 4855,         # Region of Waterloo Intl (historical ID)
    "waterloo": 4855,
    "cambridge": 4855,
    "invermere": 1163781,      # Invermere Airport (historical climate id)
    "kelowna": 1163783,        # Kelowna Airport climate id
}

# ----------------------------
# Helpers / utilities
# ----------------------------

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None

def _year_str_to_int(d: str) -> int:
    try:
        return int(str(d)[:4])
    except Exception:
        return 0

def _normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize any ECCC daily CSV (wide or long) into:
       date, temp_max, temp_min, temp_mean, precip_mm
    Robust against extra header lines, duplicate columns, or long form.
    """
    # unify headers
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all", axis=1)

    # if the file has preamble lines, find header by scanning for a line with 'Date'
    if "Date" not in df.columns and "DATE" not in df.columns and not any("date" in c.lower() for c in df.columns):
        # sometimes we got raw text; try re-read from joined CSV lines if necessary
        # caller should ensure we already sliced header, but be defensive:
        pass

    # try detect date col
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        date_cols = [c for c in df.columns if "time" in c.lower() or "date/time" in c.lower()]
    if not date_cols:
        # try common ECCC: "Date/Time"
        for c in df.columns:
            if c.lower().replace(" ", "") in ("date/time","date_time","datetime"):
                date_cols = [c]
                break
    if not date_cols:
        raise ValueError("No date/time column detected after header slice")
    date_col = date_cols[0]
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # long form pivot (metric, value)
    lowcols = [c.lower() for c in df.columns]
    if ("metric" in lowcols) and ("value" in lowcols):
        rename = {c: c.lower() for c in df.columns}
        df = df.rename(columns=rename)
        # de-dup by averaging
        if df.duplicated(subset=["metric", "date"], keep=False).any():
            df = df.groupby(["date", "metric"], as_index=False)["value"].mean()
        df = df.pivot(index="date", columns="metric", values="value").reset_index()

    # map variants
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "max" in cl and "temp" in cl:
            col_map[c] = "temp_max"
        elif "min" in cl and "temp" in cl:
            col_map[c] = "temp_min"
        elif ("mean" in cl or "avg" in cl) and "temp" in cl:
            col_map[c] = "temp_mean"
        elif "precip" in cl and ("mm" in cl or "precip" == cl):
            col_map[c] = "precip_mm"
        elif "date" in cl:
            col_map[c] = "date"
    if col_map:
        df = df.rename(columns=col_map)

    # De-dup duplicate-named columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    keep = [c for c in ["date", "temp_max", "temp_min", "temp_mean", "precip_mm"] if c in df.columns]
    df = df[keep].copy()

    # parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df

async def ecc_count_valid_days(station_id: int, years=(2024, 2025)) -> int:
    """
    Count non-empty rows across temp/precip columns for validation.
    """
    total_valid = 0
    for yr in years:
        try:
            ec = ECHistorical(station_id=int(station_id), year=int(yr), format="csv", timeframe=2)
            await ec.update()
            data_src = ec.station_data
            if data_src is None:
                continue

            # Some versions give a path-like object, others a StringIO
            if hasattr(data_src, "getvalue"):
                text = data_src.getvalue()
            else:
                with open(data_src, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            # slice header by the first line that contains Date or DATE
            lines = [ln for ln in text.splitlines() if ln.strip()]
            header_idx = next((i for i, ln in enumerate(lines) if ("Date" in ln or "DATE" in ln) and "," in ln), None)
            if header_idx is None:
                continue
            df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
            # find metric-like columns
            metric_cols = [c for c in df.columns if any(k in c.lower() for k in ["temp", "precip", "rain", "snow"]) and not c.lower().startswith("flag")]
            if not metric_cols:
                continue
            count = 0
            for c in metric_cols:
                vals = pd.to_numeric(df[c], errors="coerce")
                count += vals.notna().sum()
            total_valid += count
        except Exception:
            continue
    return total_valid

async def ecc_fetch_city_year(station_id: int, year: int) -> Optional[pd.DataFrame]:
    """
    Pull one year's daily CSV and normalize.
    """
    try:
        ec = ECHistorical(station_id=int(station_id), year=int(year), format="csv", timeframe=2)
        await ec.update()
        data_src = ec.station_data
        if data_src is None:
            return None

        if hasattr(data_src, "getvalue"):
            text = data_src.getvalue()
        else:
            with open(data_src, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        # slice to row with Date columns
        lines = [ln for ln in text.splitlines() if ln.strip()]
        header_idx = next((i for i, ln in enumerate(lines) if ("Date" in ln or "DATE" in ln) and "," in ln), None)
        if header_idx is None:
            return None
        df_raw = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))

        df_norm = _normalize_weather_columns(df_raw)
        return df_norm
    except Exception:
        return None

# ----------------------------
# GeoMet fallback
# ----------------------------

async def geomet_candidates(session: aiohttp.ClientSession, lat: float, lon: float, pad_deg: float = 0.6, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Query GeoMet climate-stations near (lat, lon) within bbox (pad_deg).
    """
    minx = lon - pad_deg
    maxx = lon + pad_deg
    miny = lat - pad_deg
    maxy = lat + pad_deg
    params = {"f": "json", "lang": "en", "limit": str(limit), "bbox": f"{minx},{miny},{maxx},{maxy}"}
    try:
        async with session.get(GEOMET_BASE, params=params, timeout=60) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception:
        return []
    return data.get("features", [])

def _geomet_has_daily_coverage(props: Dict[str, Any]) -> bool:
    d1 = props.get("DLY_FIRST_DATE") or props.get("dly_first_date")
    d2 = props.get("DLY_LAST_DATE")  or props.get("dly_last_date")
    if not d1 or not d2:
        return False
    y1 = _year_str_to_int(str(d1))
    y2 = _year_str_to_int(str(d2))
    return y1 <= 2024 and y2 >= 2025

def _score_station_name_for_quality(name: str, city: str) -> int:
    n = name.lower()
    score = 0
    if city.lower() in n: score += 3
    if any(k in n for k in ["intl", "airport", " a ", " ap ", "cda", "climate"]): score += 2
    if any(k in n for k in ["auto", " automatic"]): score += 1
    return score

async def select_station_for_city(session: aiohttp.ClientSession, city: str, lat: float, lon: float) -> Optional[Tuple[int, str, float]]:
    """
    Choose a station id/name for city using overrides → env_canada → GeoMet fallback (expanding radius).
    Returns (station_id, station_name, distance_km) or None.
    """
    # 0) Overrides
    if city.lower() in STATION_OVERRIDES:
        sid = STATION_OVERRIDES[city.lower()]
        days = await ecc_count_valid_days(sid)
        if days > 200:
            return sid, f"OVERRIDE_{sid}", np.nan

    # 1) env_canada helper (if available)
    if get_historical_stations is not None:
        for radius_km in (50, 100, 200):
            try:
                stations = await get_historical_stations([lat, lon], radius=radius_km, limit=20)
            except Exception:
                stations = None

            if isinstance(stations, dict) and stations:
                # stations dict sample provided by you
                rows = []
                for name, meta in stations.items():
                    sid = _safe_int(meta.get("id"))
                    if sid is None: continue
                    # Coverage windows come as 'YYYY-MM-DD|YYYY-MM-DD'
                    dly = meta.get("dlyRange", "") or ""
                    first, last = (dly.split("|") + ["",""])[:2]
                    score = 0
                    if first and last:
                        y1 = _year_str_to_int(first)
                        y2 = _year_str_to_int(last)
                        if y1 <= 2024 and y2 >= 2025: score += 3
                    if "airport" in name.lower() or "a'" in name.lower(): score += 2
                    prox = float(meta.get("proximity", 9999.0))
                    rows.append((sid, name, prox, score))
                if rows:
                    rows.sort(key=lambda x: (-x[3], x[2]))  # score desc, then proximity asc
                    for sid, name, prox, _ in rows[:8]:
                        days = await ecc_count_valid_days(sid)
                        if days > 200:
                            return sid, name, prox

    # 2) GeoMet fallback with expanding bbox
    for pad in (0.6, 1.0, 1.8, 2.5):
        feats = await geomet_candidates(session, lat, lon, pad_deg=pad, limit=800)
        if not feats:
            continue

        cands = []
        for f in feats:
            props = f.get("properties", {})
            geom  = f.get("geometry", {}) or {}
            coords = geom.get("coordinates") if geom else None
            if not coords or len(coords) < 2:
                continue
            sid = _safe_int(props.get("STN_ID"))
            if sid is None:
                continue
            stn_lon, stn_lat = coords[0], coords[1]
            dist_km = geodesic((lat, lon), (stn_lat, stn_lon)).km
            score = _score_station_name_for_quality(str(props.get("STATION_NAME","")), city)
            if _geomet_has_daily_coverage(props): score += 3
            cands.append((sid, props.get("STATION_NAME","?"), dist_km, score))

        if not cands:
            continue

        # rank, then validate via ECCC
        cands.sort(key=lambda x: (-x[3], x[2]))  # score desc, dist asc
        for sid, name, dist_km, _ in cands[:12]:
            days = await ecc_count_valid_days(sid)
            if days > 200:
                return sid, name, dist_km

    return None

# ----------------------------
# City driver
# ----------------------------

async def fetch_city_weather(session: aiohttp.ClientSession, city: str, city_id: int, lat: float, lon: float) -> Dict[str, Any]:
    """
    Resolve station & fetch 2024–2025 daily, normalize, write CSV.
    """
    row = {
        "city": city,
        "city_id": city_id,
        "station_id": None,
        "station_name": None,
        "dist_km": None,
        "n_days": 0,
        "ok": False,
        "path": None,
        "error": None
    }

    try:
        sel = await select_station_for_city(session, city, float(lat), float(lon))
        if not sel:
            row["error"] = "no candidates"
            return row
        sid, sname, dist_km = sel

        frames = []
        for yr in (2024, 2025):
            dfy = await ecc_fetch_city_year(sid, yr)
            if dfy is None or dfy.empty:
                continue
            frames.append(dfy)

        if not frames:
            row["error"] = "no valid frames"
            return row

        df_city = pd.concat(frames, ignore_index=True)
        # keep only in range
        df_city = df_city[(df_city["date"] >= pd.to_datetime(START_DATE)) &
                          (df_city["date"] <= pd.to_datetime(END_DATE))].copy()

        # numeric coercion; fix “DataFrame not a Series” by selecting first col if needed
        for col in ["temp_max","temp_min","temp_mean","precip_mm"]:
            if col in df_city.columns:
                series = df_city[col]
                if isinstance(series, pd.DataFrame):
                    # take the first non-duplicate column
                    series = series.iloc[:, 0]
                df_city[col] = pd.to_numeric(series, errors="coerce")

        # enrich metadata
        df_city["city_id"] = city_id
        df_city["city"] = city

        out_path = os.path.join(OUT_DIR, f"weather_{re.sub(r'[^a-z0-9_]+','_', city.lower().strip().replace(' ','_'))}.csv")
        df_city.to_csv(out_path, index=False)

        row.update({
            "station_id": sid,
            "station_name": sname,
            "dist_km": float(dist_km) if dist_km == dist_km else None,  # handle NaN
            "n_days": int(len(df_city)),
            "ok": True,
            "path": out_path
        })
        return row

    except Exception as e:
        row["error"] = str(e)
        return row
# Utility: Clean city names for geocoding
# ---------------------------------------------------
def clean_city_name(city_name: str) -> str:
    city_name = re.sub(r"[^a-zA-Z0-9\s-]", "", city_name)
    return city_name.strip()

# Step 1. Add lat/lon for cities
# ---------------------------------------------------
def add_lat_lon_to_cities(city_csv, out_csv):
    df = pd.read_csv(city_csv)
    if not os.path.exists(city_csv):
        raise FileNotFoundError(f"{city_csv} not found (expected columns: city_name, city_id).")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if not {"city_id", "city_name"}.issubset(df.columns):
        raise ValueError("city.csv must contain columns: city_id, city_name")

    geolocator = Nominatim(user_agent="tips_forecast_project")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    lats, lons = [], []
    for _, row in df.iterrows():
        raw_name = row["city_name"]
        safe_name = clean_city_name(raw_name)
        try:
            loc = geocode(f"{safe_name}, Canada")
            if loc:
                lats.append(loc.latitude)
                lons.append(loc.longitude)
                print(f"[ok] {raw_name}: ({loc.latitude:.4f}, {loc.longitude:.4f})")
            else:
                lats.append(None)
                lons.append(None)
                print(f"[warn] {raw_name}: no result")
        except Exception as e:
            print(f"[error] {raw_name}: {e}")
            lats.append(None)
            lons.append(None)

    df["lat"] = lats
    df["lon"] = lons
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with latitude/longitude for {len(df)} cities.")
    return out_csv

# ----------------------------
# Main
# ----------------------------

async def main():
    # Input city file must have: city, city_id, lat, lon
    csv_lat_long = add_lat_lon_to_cities(
        city_csv="/Users/niharikam/Documents/tips_forecast_clean/data/city.csv",
        out_csv="/Users/niharikam/Documents/tips_forecast_clean/data/city_with_latlon.csv",
    )

    cities = pd.read_csv(csv_lat_long)
    cities.columns = [c.strip().lower().replace(" ", "_") for c in cities.columns]
    need = {"city_name","city_id","lat","lon"}
    if not need.issubset(set(cities.columns)):
        raise ValueError(f"{cities} must have columns {need}, found {set(cities.columns)}")

    # run in parallel with small concurrency
    sem = asyncio.Semaphore(6)
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        results = []

        async def run_city(row):
            async with sem:
                return await fetch_city_weather(session, str(row["city_name"]).strip(), int(row["city_id"]), float(row["lat"]), float(row["lon"]))

        for _, r in cities.iterrows():
            if pd.isna(r["lat"]) or pd.isna(r["lon"]):
                continue
            tasks.append(run_city(r))

        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            c = res["city"]
            if res["ok"]:
                print(f"[ok] {c}: saved {res['n_days']} days → {res['path']}")
            else:
                print(f"[warn] {c}: {res.get('error','unknown error')}")

    # Write summaries
    df_sum = pd.DataFrame(results).sort_values(["ok","n_days"], ascending=[True, False])
    df_sum.to_csv(os.path.join(DATA_DIR, "weather_fetch_summary.csv"), index=False)

    station_map = df_sum[["city_id","city","station_id","station_name","dist_km","n_days","ok"]].copy()
    station_map.to_csv(os.path.join(DATA_DIR, "city_station_map.csv"), index=False)

    print("Saved weather_fetch_summary.csv and city_station_map.csv")

if __name__ == "__main__":
    asyncio.run(main())

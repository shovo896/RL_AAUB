"""
OpenSky Collision-Avoidance Dataset Builder
Run locally with internet access.

Output:
  opensky_collision_avoidance_dataset.csv

Install:
  pip install requests pandas numpy

Optional OpenSky auth:
  export OPENSKY_TOKEN="your_bearer_token"
"""

import os, time, itertools, math, requests
import pandas as pd
import numpy as np

API_URL = "https://opensky-network.org/api/states/all"

# Bangladesh / Dhaka-region bounding box. Change as needed.
PARAMS = {
    "lamin": 22.5,
    "lomin": 88.8,
    "lamax": 24.9,
    "lomax": 91.9,
}

HEADERS = {}
token = os.getenv("OPENSKY_TOKEN")
if token:
    HEADERS["Authorization"] = f"Bearer {token}"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def bucket_distance(d_km):
    if d_km < 5: return "near"
    if d_km < 20: return "medium"
    return "far"

def bucket_altitude(delta_m):
    if abs(delta_m) < 300: return "same"
    return "above" if delta_m > 0 else "below"

def bucket_closing(v_rel):
    if v_rel > 15: return "approaching"
    if v_rel < -15: return "moving_away"
    return "stable"

def recommended_action(distance_state, altitude_state, closing_state):
    if distance_state == "near" and altitude_state == "same":
        return "climb"
    if distance_state == "near" and altitude_state == "below":
        return "descend"
    if distance_state == "near" and altitude_state == "above":
        return "climb"
    if distance_state == "medium" and altitude_state == "same" and closing_state == "approaching":
        return "climb"
    return "maintain"

def reward(distance_state, altitude_state, action):
    if distance_state == "near" and altitude_state == "same":
        return -100
    if distance_state == "medium" and altitude_state == "same":
        return -20
    if action in ["climb", "descend"]:
        return -2
    return 5

def fetch_states():
    r = requests.get(API_URL, params=PARAMS, headers=HEADERS, timeout=30)
    r.raise_for_status()
    payload = r.json()
    cols = [
        "icao24","callsign","origin_country","time_position","last_contact",
        "longitude","latitude","baro_altitude","on_ground","velocity",
        "true_track","vertical_rate","sensors","geo_altitude","squawk",
        "spi","position_source","category"
    ]
    states = payload.get("states") or []
    df = pd.DataFrame(states, columns=cols[:len(states[0])] if states else cols)
    need = ["icao24","latitude","longitude","geo_altitude","velocity","true_track","vertical_rate"]
    df = df.dropna(subset=["icao24","latitude","longitude","geo_altitude","velocity"])
    return df[need].copy(), payload.get("time")

def build_pairs(df, snapshot_time, max_pair_distance_km=50):
    rows = []
    for a, b in itertools.combinations(df.to_dict("records"), 2):
        d_km = haversine_km(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
        if d_km > max_pair_distance_km:
            continue

        delta_alt = b["geo_altitude"] - a["geo_altitude"]
        rel_speed_proxy = abs((a.get("velocity") or 0) - (b.get("velocity") or 0))
        d_state = bucket_distance(d_km)
        a_state = bucket_altitude(delta_alt)
        c_state = bucket_closing(rel_speed_proxy)
        action = recommended_action(d_state, a_state, c_state)

        rows.append({
            "time": snapshot_time,
            "own_icao24": a["icao24"],
            "intruder_icao24": b["icao24"],
            "own_latitude": a["latitude"],
            "own_longitude": a["longitude"],
            "own_geo_altitude_m": a["geo_altitude"],
            "own_velocity_ms": a["velocity"],
            "own_heading_deg": a.get("true_track"),
            "own_vertical_rate_ms": a.get("vertical_rate"),
            "intruder_latitude": b["latitude"],
            "intruder_longitude": b["longitude"],
            "intruder_geo_altitude_m": b["geo_altitude"],
            "intruder_velocity_ms": b["velocity"],
            "intruder_heading_deg": b.get("true_track"),
            "intruder_vertical_rate_ms": b.get("vertical_rate"),
            "relative_distance_km": round(d_km, 3),
            "relative_altitude_m": round(delta_alt, 2),
            "relative_speed_proxy_ms": round(rel_speed_proxy, 2),
            "distance_state": d_state,
            "altitude_state": a_state,
            "closing_state": c_state,
            "mdp_state": f"{d_state}_{a_state}_{c_state}",
            "recommended_action": action,
            "reward": reward(d_state, a_state, action),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    all_pairs = []
    snapshots = 10000         # collect 10000
    sleep_seconds = 10     # wait between snapshots

    for i in range(snapshots):
        states, t = fetch_states()
        pairs = build_pairs(states, t)
        all_pairs.append(pairs)
        print(f"snapshot={i+1}, aircraft={len(states)}, pairs={len(pairs)}")
        if i < snapshots - 1:
            time.sleep(sleep_seconds)

    out = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    out.to_csv("opensky_collision_avoidance_dataset.csv", index=False)
    print("saved opensky_collision_avoidance_dataset.csv", out.shape)

import argparse
import json
import os
import time
import math
import itertools
from pathlib import Path

import requests
import pandas as pd

API_URL = "https://opensky-network.org/api/states/all"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"

# High-traffic Asia region
PARAMS = {
    "lamin": 1,
    "lomin": 95,
    "lamax": 45,
    "lomax": 145
}

TARGET_ROWS = 100_000
SLEEP_SECONDS = 90
RATE_LIMIT_SLEEP = 600
MAX_PAIR_DISTANCE_KM = 80
OUTPUT_FILE = BASE_DIR / "opensky_collision_avoidance_dataset.csv"
TOKEN_REFRESH_MARGIN_SECONDS = 30


class TokenManager:
    def __init__(self):
        self.token = os.getenv("OPENSKY_TOKEN")
        self.expires_at = float("inf") if self.token else 0
        self.client_id = None
        self.client_secret = None

        if not self.token:
            self._load_credentials()

    def _load_credentials(self):
        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                f"Missing {CREDENTIALS_FILE}. Add OpenSky clientId/clientSecret."
            )

        with CREDENTIALS_FILE.open(encoding="utf-8") as file:
            credentials = json.load(file)

        self.client_id = credentials.get("clientId") or credentials.get("client_id")
        self.client_secret = (
            credentials.get("clientSecret") or credentials.get("client_secret")
        )

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "credentials.json must contain clientId and clientSecret."
            )

    def invalidate(self):
        self.expires_at = 0

    def get_token(self):
        if self.token and time.monotonic() < self.expires_at:
            return self.token

        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        self.token = payload["access_token"]
        expires_in = payload.get("expires_in", 1800)
        self.expires_at = time.monotonic() + max(
            expires_in - TOKEN_REFRESH_MARGIN_SECONDS, 0
        )
        return self.token

    def headers(self):
        return {"Authorization": f"Bearer {self.get_token()}"}


TOKEN_MANAGER = TokenManager()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bucket_distance(d_km):
    if d_km < 5:
        return "near"
    if d_km < 20:
        return "medium"
    return "far"


def bucket_altitude(delta_m):
    if abs(delta_m) < 300:
        return "same"
    if delta_m > 0:
        return "above"
    return "below"


def bucket_closing(relative_speed):
    if relative_speed > 15:
        return "approaching"
    if relative_speed < -15:
        return "moving_away"
    return "stable"


def recommended_action(distance_state, altitude_state, closing_state):
    if distance_state == "near" and altitude_state == "same":
        return "climb"

    if distance_state == "near" and altitude_state == "below":
        return "descend"

    if distance_state == "near" and altitude_state == "above":
        return "climb"

    if (
        distance_state == "medium"
        and altitude_state == "same"
        and closing_state == "approaching"
    ):
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
    for attempt in range(2):
        response = requests.get(
            API_URL,
            params=PARAMS,
            headers=TOKEN_MANAGER.headers(),
            timeout=30,
        )

        if response.status_code != 401 or attempt == 1:
            break

        TOKEN_MANAGER.invalidate()

    response.raise_for_status()
    payload = response.json()

    states = payload.get("states", [])
    snapshot_time = payload.get("time")

    cols = [
        "icao24",
        "callsign",
        "origin_country",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "true_track",
        "vertical_rate",
        "sensors",
        "geo_altitude",
        "squawk",
        "spi",
        "position_source",
        "category"
    ]

    if not states:
        return pd.DataFrame(), snapshot_time

    normalized_states = [
        list(state[:len(cols)]) + [None] * max(0, len(cols) - len(state))
        for state in states
    ]
    df = pd.DataFrame(normalized_states, columns=cols)

    required_cols = [
        "icao24",
        "latitude",
        "longitude",
        "geo_altitude",
        "velocity",
        "true_track",
        "vertical_rate"
    ]

    df = df.dropna(
        subset=["icao24", "latitude", "longitude", "geo_altitude", "velocity"]
    )

    return df[required_cols].copy(), snapshot_time


def build_pairs(df, snapshot_time):
    rows = []
    aircraft = df.to_dict("records")

    for own, intruder in itertools.combinations(aircraft, 2):
        distance_km = haversine_km(
            own["latitude"],
            own["longitude"],
            intruder["latitude"],
            intruder["longitude"]
        )

        if distance_km > MAX_PAIR_DISTANCE_KM:
            continue

        relative_altitude = intruder["geo_altitude"] - own["geo_altitude"]

        own_speed = own["velocity"] if own["velocity"] else 0
        intruder_speed = intruder["velocity"] if intruder["velocity"] else 0

        relative_speed = abs(own_speed - intruder_speed)

        distance_state = bucket_distance(distance_km)
        altitude_state = bucket_altitude(relative_altitude)
        closing_state = bucket_closing(relative_speed)

        action = recommended_action(
            distance_state,
            altitude_state,
            closing_state
        )

        rows.append({
            "time": snapshot_time,

            "own_icao24": own["icao24"],
            "intruder_icao24": intruder["icao24"],

            "own_latitude": own["latitude"],
            "own_longitude": own["longitude"],
            "own_geo_altitude_m": own["geo_altitude"],
            "own_velocity_ms": own["velocity"],
            "own_heading_deg": own["true_track"],
            "own_vertical_rate_ms": own["vertical_rate"],

            "intruder_latitude": intruder["latitude"],
            "intruder_longitude": intruder["longitude"],
            "intruder_geo_altitude_m": intruder["geo_altitude"],
            "intruder_velocity_ms": intruder["velocity"],
            "intruder_heading_deg": intruder["true_track"],
            "intruder_vertical_rate_ms": intruder["vertical_rate"],

            "relative_distance_km": round(distance_km, 3),
            "relative_altitude_m": round(relative_altitude, 2),
            "relative_speed_proxy_ms": round(relative_speed, 2),

            "distance_state": distance_state,
            "altitude_state": altitude_state,
            "closing_state": closing_state,

            "mdp_state": f"{distance_state}_{altitude_state}_{closing_state}",
            "recommended_action": action,
            "reward": reward(distance_state, altitude_state, action)
        })

    return pd.DataFrame(rows)


def save_progress(all_pairs):
    if not all_pairs:
        return 0

    final_df = pd.concat(all_pairs, ignore_index=True)
    final_df = final_df.drop_duplicates(
        subset=[
            "time",
            "own_icao24",
            "intruder_icao24",
            "relative_distance_km",
            "relative_altitude_m"
        ]
    )

    final_df.to_csv(OUTPUT_FILE, index=False)
    return len(final_df)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an OpenSky collision-avoidance dataset."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing output dataset before collecting.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch and save one snapshot, then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.reset and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"Reset existing dataset: {OUTPUT_FILE}")

    all_pairs = []
    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE)
        all_pairs.append(existing_df)
        print(f"Resuming with {len(existing_df)} existing rows.")

    snapshot_count = 0

    while True:
        try:
            states_df, snapshot_time = fetch_states()

            if states_df.empty:
                print("No aircraft found. Waiting...")
                time.sleep(SLEEP_SECONDS)
                continue

            pairs_df = build_pairs(states_df, snapshot_time)

            if not pairs_df.empty:
                all_pairs.append(pairs_df)

            total_rows = save_progress(all_pairs)

            snapshot_count += 1

            print(
                f"snapshot={snapshot_count}, "
                f"aircraft={len(states_df)}, "
                f"pairs={len(pairs_df)}, "
                f"total_rows={total_rows}"
            )

            if total_rows >= TARGET_ROWS:
                print("Target reached.")
                break

            if args.once:
                print("Single snapshot complete.")
                break

            time.sleep(SLEEP_SECONDS)

        except requests.exceptions.HTTPError as e:
            print("HTTP Error:", e)

            if "429" in str(e):
                print("Rate limit hit. Sleeping 10 minutes...")
                time.sleep(RATE_LIMIT_SLEEP)
            else:
                print("Other HTTP error. Sleeping 2 minutes...")
                time.sleep(120)

        except Exception as e:
            print("Error:", e)
            print("Sleeping 2 minutes...")
            time.sleep(120)

    print(f"Dataset saved as: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

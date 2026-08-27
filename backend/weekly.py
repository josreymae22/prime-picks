import argparse
import os
import sys
import requests

PP_API_URL = os.getenv("PP_API_URL", "").rstrip("/")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")


def call(method, path, params=None):
    url = f"{PP_API_URL}{path}"

    try:
        if method == "POST":
            response = requests.post(url, params=params, timeout=120)
        else:
            response = requests.get(url, params=params, timeout=120)

        response.raise_for_status()

        print(f"OK {method} {path} -> {response.status_code}")
        return response.json()

    except Exception as e:
        print(f"FAILED {method} {path}: {e}")
        return None


def run_once():
    if not PP_API_URL:
        print("ERROR: PP_API_URL is missing")
        sys.exit(1)

    if not ADMIN_SECRET:
        print("ERROR: ADMIN_SECRET is missing")
        sys.exit(1)

    print("=== Prime Picks hourly update starting ===")
    print(f"API: {PP_API_URL}")

    # 1. Refresh NFL injuries
    call(
        "POST",
        "/injuries/refresh",
        {
            "league": "NFL",
            "secret": ADMIN_SECRET,
        },
    )

    # 2. Sync NFL roster/player moves
    call(
        "POST",
        "/roster/sync",
        {
            "league": "NFL",
            "secret": ADMIN_SECRET,
        },
    )

    # 3. Take latest NFL betting line snapshot
    call(
        "POST",
        "/movement/snapshot",
        {
            "league": "NFL",
            "secret": ADMIN_SECRET,
        },
    )

    # 4. Take CFB betting line snapshot
    call(
        "POST",
        "/movement/snapshot",
        {
            "league": "CFB",
            "secret": ADMIN_SECRET,
        },
    )

    print("=== Prime Picks hourly update complete ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one update cycle and exit",
    )

    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_once()


if __name__ == "__main__":
    main()

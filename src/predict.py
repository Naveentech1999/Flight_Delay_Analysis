"""Predict flight delay risk for a single flight."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one flight prediction."""
    parser = argparse.ArgumentParser(description="Predict whether one flight is likely to be delayed.")
    parser.add_argument("--model", type=Path, default=Path("models/flight_delay_model.joblib"), help="Path to a trained model file.")
    parser.add_argument("--airline", required=True, help="Airline or carrier code, such as AA or DL.")
    parser.add_argument("--origin", required=True, help="Origin airport code.")
    parser.add_argument("--destination", required=True, help="Destination airport code.")
    parser.add_argument("--scheduled-departure-hour", type=int, required=True, help="Scheduled departure hour from 0 to 23.")
    parser.add_argument("--day-of-week", type=int, required=True, help="Day of week from 1 to 7.")
    parser.add_argument("--month", type=int, required=True, help="Month from 1 to 12.")
    parser.add_argument("--distance", type=float, required=True, help="Flight distance in miles.")
    return parser.parse_args()


def main() -> None:
    """Load the model and print a delay prediction."""
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}. Run src/train_model.py first.")

    model = joblib.load(args.model)
    flight = pd.DataFrame(
        [
            {
                "airline": args.airline,
                "origin": args.origin,
                "destination": args.destination,
                "scheduled_departure_hour": args.scheduled_departure_hour,
                "day_of_week": args.day_of_week,
                "month": args.month,
                "distance": args.distance,
            }
        ]
    )
    prediction = int(model.predict(flight)[0])
    probability = float(model.predict_proba(flight)[0][1])

    print(f"Prediction: {'Delayed' if prediction else 'On time'}")
    print(f"Delay probability: {probability:.2f}")


if __name__ == "__main__":
    main()

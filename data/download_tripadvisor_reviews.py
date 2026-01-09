"""
Script to download TripAdvisor hotel reviews dataset from Hugging Face
and save it as JSON.
"""

import json
from datetime import datetime, date
from pathlib import Path
from datasets import load_dataset


def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def get_sentiment(label: int) -> str:
    """Convert numeric label to sentiment string."""
    if label <= 2:
        return "negative"
    elif label == 3:
        return "neutral"
    else:  # 4-5
        return "positive"


def main():
    # Define output path
    output_dir = Path(__file__).parent / "final_assignment"
    output_file = output_dir / "customer_surveys_hotels.json"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading TripAdvisor hotel reviews dataset from Hugging Face...")
    dataset = load_dataset("argilla/tripadvisor-hotel-reviews")

    # Convert to list of dictionaries
    # The dataset typically has a 'train' split
    if "train" in dataset:
        raw_data = dataset["train"].to_list()
    else:
        # If no train split, use the first available split
        split_name = list(dataset.keys())[0]
        raw_data = dataset[split_name].to_list()

    print(f"Downloaded {len(raw_data)} reviews")

    # Simplify to match customer_surveys.json format
    data = []
    for item in raw_data:
        label = int(item["prediction"][0]["label"])
        data.append({
            "id": item["id"],
            "review": item["text"],
            "customer_satisfaction_score": label,
            "survey_sentiment": get_sentiment(label),
        })

    # Save as JSON
    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_serializer)

    print(f"Successfully saved {len(data)} reviews to {output_file}")


if __name__ == "__main__":
    main()

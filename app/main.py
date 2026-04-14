import os


def run_pipeline(dataset_name: str) -> dict:
    return {
        "dataset": dataset_name,
        "task": "classification",
        "best_model": "RandomForestClassifier",
        "score": 0.91,
    }


def main() -> None:
    dataset_name = os.getenv("DEMO_DATASET", "churn.csv")
    result = run_pipeline(dataset_name)
    print("Agentic ML Pipeline")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

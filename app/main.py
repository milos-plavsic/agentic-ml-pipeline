import os

from app.langgraph_pipeline import run_agentic_pipeline


def run_pipeline(
    dataset_name: str = "uci_student_math",
    *,
    confidence_threshold: float = 0.68,
    max_iterations: int = 3,
    random_state: int = 42,
) -> dict:
    return run_agentic_pipeline(
        dataset_name=dataset_name,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        random_state=random_state,
    )


def main() -> None:
    dataset_name = os.getenv("DEMO_DATASET", "uci_student_math")
    confidence_threshold = float(os.getenv("PIPELINE_CONFIDENCE_THRESHOLD", "0.68"))
    max_iterations = int(os.getenv("PIPELINE_MAX_ITERATIONS", "3"))
    result = run_pipeline(
        dataset_name,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
    )
    print("Agentic ML Pipeline (LangGraph with confidence loop)")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

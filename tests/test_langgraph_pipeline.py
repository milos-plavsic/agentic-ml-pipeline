from app.langgraph_pipeline import run_agentic_pipeline


def test_confidence_retry_loop_runs_multiple_iterations_when_threshold_high() -> None:
    out = run_agentic_pipeline(
        dataset_name="uci_student_math",
        confidence_threshold=0.99,
        max_iterations=3,
        random_state=0,
    )
    assert out["iterations"] == 3
    assert out["loop_terminated_reason"] == "max_iterations_reached"
    assert len(out["iteration_history"]) == 3
    assert len(out["decision_log"]) == 3


def test_confidence_threshold_stops_early_when_met() -> None:
    out = run_agentic_pipeline(
        dataset_name="uci_student_math",
        confidence_threshold=0.1,
        max_iterations=3,
        random_state=0,
    )
    assert out["iterations"] == 1
    assert out["loop_terminated_reason"] == "confidence_threshold_reached"
    assert out["confidence_score"] >= out["confidence_threshold"]

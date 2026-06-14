"""Check official_test evaluation guardrails."""

from __future__ import annotations

from src.evaluate import validate_evaluation_request


def main() -> None:
    validate_evaluation_request({"split": "val", "official_final_eval": False})

    try:
        validate_evaluation_request({"split": "test", "official_final_eval": False})
    except ValueError:
        pass
    else:
        raise AssertionError("official_test evaluation was allowed without official flag")

    try:
        validate_evaluation_request(
            {
                "split": "test",
                "official_final_eval": True,
                "final_selection_lock": "results/protocol/nonexistent_lock_for_test.json",
            }
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("official_test evaluation was allowed without a lock file")


if __name__ == "__main__":
    main()

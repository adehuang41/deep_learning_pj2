"""Check validation-based checkpoint selection."""

from __future__ import annotations

from src.train import DEFAULT_CONFIG, _is_better_val, _protocol_guard


def main() -> None:
    assert DEFAULT_CONFIG["checkpoint_selection_split"] == "val"
    _protocol_guard({"train_split": "dev", "checkpoint_selection_split": "val"})
    assert _is_better_val(0.91, 0.5, 0.90, 0.4)
    assert _is_better_val(0.91, 0.3, 0.91, 0.4)
    assert not _is_better_val(0.90, 0.3, 0.91, 0.4)

    try:
        _protocol_guard({"train_split": "dev", "checkpoint_selection_split": "none"})
    except ValueError:
        pass
    else:
        raise AssertionError("dev training accepted non-validation checkpoint selection")


if __name__ == "__main__":
    main()

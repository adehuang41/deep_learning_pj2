"""Check that training code does not load official_test."""

from __future__ import annotations

import inspect

from src import train


def main() -> None:
    assert train.DEFAULT_CONFIG["use_test_during_training"] is False
    source = inspect.getsource(train.run_training)
    assert "include_test=False" in source
    assert "Training loader unexpectedly includes official_test" in source
    assert "loaders.test" in source


if __name__ == "__main__":
    main()

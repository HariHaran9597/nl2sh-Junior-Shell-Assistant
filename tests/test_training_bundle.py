from pathlib import Path

import pytest

from training.train_qlora import (
    BASE_MODEL_REVISION,
    DATASET_REVISION,
    prompt_completion,
    latest_checkpoint,
)


def test_training_uses_pinned_public_revisions():
    assert len(BASE_MODEL_REVISION) == 40
    assert len(DATASET_REVISION) == 40


def test_prompt_completion_maps_only_request_and_shell_target():
    row = prompt_completion({"nl": "  show files  ", "bash": "  ls -la  ", "extra": "ignored"})
    assert row == {
        "prompt": [{"role": "user", "content": "show files"}],
        "completion": [{"role": "assistant", "content": "ls -la"}],
    }


@pytest.mark.parametrize("row", [
    {"nl": "", "bash": "ls"},
    {"nl": "show files", "bash": "  "},
])
def test_prompt_completion_rejects_empty_pairs(row):
    with pytest.raises(ValueError, match="non-empty"):
        prompt_completion(row)


def test_latest_checkpoint_selects_highest_numeric_step(tmp_path):
    for name in ("checkpoint-100", "checkpoint-9", "checkpoint-25", "not-a-checkpoint"):
        (tmp_path / name).mkdir()
    assert Path(latest_checkpoint(tmp_path)).name == "checkpoint-100"


def test_latest_checkpoint_returns_none_when_absent(tmp_path):
    assert latest_checkpoint(tmp_path) is None

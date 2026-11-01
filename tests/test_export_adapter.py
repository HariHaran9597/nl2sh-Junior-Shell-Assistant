import json

import pytest

from training.export_adapter_gguf import read_adapter_info


def write_adapter(root, *, base="Qwen/Qwen2.5-0.5B-Instruct", peft_type="LORA", task_type="CAUSAL_LM", metadata=None):
    root.mkdir(parents=True)
    (root / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": base,
        "peft_type": peft_type,
        "task_type": task_type,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj"],
    }), encoding="utf-8")
    (root / "adapter_model.safetensors").write_bytes(b"test-adapter-payload")
    if metadata is not None:
        (root / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_read_adapter_info_uses_base_declared_by_artifact(tmp_path):
    adapter = tmp_path / "adapter"
    write_adapter(adapter, metadata={"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "rows": 10000})

    info = read_adapter_info(adapter)

    assert info["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert info["training_metadata"]["rows"] == 10000
    assert info["adapter_bytes"] == len(b"test-adapter-payload")
    assert len(info["adapter_sha256"]) == 64


def test_read_adapter_info_rejects_metadata_base_mismatch(tmp_path):
    adapter = tmp_path / "adapter"
    write_adapter(adapter, metadata={"base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct"})

    with pytest.raises(ValueError, match="Adapter/base mismatch"):
        read_adapter_info(adapter)


@pytest.mark.parametrize(("peft_type", "task_type"), [("ADALORA", "CAUSAL_LM"), ("LORA", "SEQ_CLS")])
def test_read_adapter_info_rejects_unsupported_adapter_type(tmp_path, peft_type, task_type):
    adapter = tmp_path / "adapter"
    write_adapter(adapter, peft_type=peft_type, task_type=task_type)

    with pytest.raises(ValueError, match="Only causal-language-model LoRA"):
        read_adapter_info(adapter)


def test_read_adapter_info_requires_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        read_adapter_info(adapter)

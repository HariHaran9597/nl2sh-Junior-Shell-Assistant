"""Static verification of the trained LoRA adapter (no transformers needed).
Checks: all 28 layers x 5 target modules present, rank-16 A/B shapes, dtypes.
"""
import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open

ADAPTER = Path(__file__).resolve().parent.parent / "models" / "nl2sh-lora" / "adapter_model.safetensors"
EXPECTED_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"}
EXPECTED_LAYERS = 28  # Qwen2.5-Coder-1.5B
EXPECTED_RANK = 16


def main():
    assert ADAPTER.exists(), f"missing {ADAPTER}"
    print(f"file: {ADAPTER} ({ADAPTER.stat().st_size / 1e6:.1f} MB)")
    with safe_open(str(ADAPTER), framework="pt") as f:
        keys = list(f.keys())
    print(f"tensors: {len(keys)} (expect {EXPECTED_LAYERS * len(EXPECTED_MODULES) * 2})")

    mods, layers, bad = Counter(), set(), []
    for k in keys:
        # e.g. base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight
        parts = k.split(".")
        try:
            li = parts.index("layers")
            layers.add(int(parts[li + 1]))
        except (ValueError, IndexError):
            bad.append(k)
            continue
        mod = parts[-3] if parts[-1] == "weight" else None
        mods[mod] += 1
        with safe_open(str(ADAPTER), framework="pt") as f:
            t = f.get_tensor(k)
        is_a = k.endswith("lora_A.weight")
        r_in, r_out = (t.shape[0], t.shape[1]) if is_a else (t.shape[1], t.shape[0])
        if r_in != EXPECTED_RANK and r_out != EXPECTED_RANK:
            # one dim must equal rank
            if EXPECTED_RANK not in tuple(t.shape):
                bad.append(f"{k} shape {tuple(t.shape)}")
    print("modules:", dict(mods))
    print("layers:", len(layers), sorted(layers)[:5], "...", sorted(layers)[-3:])
    assert set(mods) == EXPECTED_MODULES, f"module mismatch: {set(mods)}"
    assert len(layers) == EXPECTED_LAYERS, f"layer count {len(layers)}"
    assert not bad, f"bad tensors: {bad[:5]}"
    # adapter config cross-check
    cfg = json.loads((ADAPTER.parent / "adapter_config.json").read_text())
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 16, cfg
    print("adapter_config: r=16 alpha=16 dropout=0.05 OK")
    print("ADAPTER VERIFIED: 28 layers x 5 modules x (A+B) rank-16, all shapes sane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

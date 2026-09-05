"""Simulate: train saves checkpoint-4000 + tar -> session wipe -> recovery cell restores.
Uses the REAL recovery-cell source from D:/Downloads/nl2sh_FIXED.ipynb.
"""
import glob
import json
import os
import tarfile
import tempfile

NB = "D:/Downloads/nl2sh_FIXED.ipynb"
nb = json.loads(open(NB, encoding="utf-8").read())
rec_cells = [c for c in nb["cells"]
             if c["cell_type"] == "code" and "READY - re-run the TRAIN cell" in c["source"]]
assert len(rec_cells) == 1, f"expected 1 recovery cell, found {len(rec_cells)}"
rec_src = rec_cells[0]["source"]
print("recovery cell found in notebook:", len(rec_src), "chars")

tmp = tempfile.mkdtemp(prefix="sim-kaggle-").replace("\\", "/")
OUT = os.path.join(tmp, "nl2sh-lora").replace("\\", "/")
os.makedirs(OUT)

# 1. Fake what the TRAIN cell produced: checkpoint dir + BackupCallback tar
ckpt = os.path.join(OUT, "checkpoint-4000")
os.makedirs(ckpt)
for f in ["adapter_model.safetensors", "optimizer.pt", "trainer_state.json",
          "tokenizer_config.json", "training_args.bin"]:
    open(os.path.join(ckpt, f), "w").write("fake")
tar_path = os.path.join(tmp, "nl2sh-ckpt-4000.tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(ckpt, arcname=os.path.basename(ckpt))
print("1. simulated train output: dir + tar exist:",
      os.path.isdir(ckpt), os.path.isfile(tar_path))

# 2. Simulate session wipe: everything under OUT gone, tar moved to working dir
import shutil
work = os.path.join(tmp, "working").replace("\\", "/")
os.makedirs(work)
shutil.move(tar_path, os.path.join(work, "nl2sh-ckpt-4000.tar.gz"))
shutil.rmtree(OUT)
os.makedirs(OUT)
print("2. simulated wipe: dirs:", glob.glob(OUT + "/checkpoint-*") or "NONE")

# 3. Run the REAL recovery cell with OUT/working pointed at temp paths
patched = rec_src.replace("/kaggle/working/nl2sh-lora", OUT)
patched = patched.replace("/kaggle/working/nl2sh-ckpt-", work + "/nl2sh-ckpt-")
ns = {"__name__": "__test__"}
exec(compile(patched, "recovery_cell", "exec"), ns)

# 4. Assert restore worked
restored = os.path.join(OUT, "checkpoint-4000")
assert os.path.isdir(restored), "RESTORE FAILED"
files = sorted(os.listdir(restored))
assert "adapter_model.safetensors" in files, files
print("3. RESTORE OK:", restored, files)
print("SIMULATION PASSED - checkpoint-4000 recovered from tar, train cell would resume from step 4000")

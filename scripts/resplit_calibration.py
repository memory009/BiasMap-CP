"""
Step 0: Re-split cal.jsonl (18,016) into 3 leakage-free sub-splits.

  diag_cal   (50%, ~9,008) — Mondrian CP calibration + BiasMap diagnosis
  repair_val (25%, ~4,504) — CVaR repair early stopping / HP tuning
  recal      (25%, ~4,504) — Post-repair recalibration (fresh split CP)

train.jsonl and test.jsonl are NOT touched.
OOD splits are NOT touched.

Also re-splits cached model outputs in results/sprint1/ so existing
inference results can be reused without re-running models.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
SPLITS_DIR = Path("data/splits")
RESULTS_DIR = Path("results/sprint1")
MODELS = ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct",
          "llava-1.5-7b-hf", "blip2-opt-2.7b"]
DATASETS = ["vsr", "gqa", "whatsup"]


def resplit_calibration():
    random.seed(SEED)

    # ---- 1. Load & stratified split cal.jsonl ----
    cal_samples = []
    with open(SPLITS_DIR / "cal.jsonl") as f:
        for line in f:
            cal_samples.append(json.loads(line))
    print(f"Original cal.jsonl: {len(cal_samples)} samples")

    by_relation = defaultdict(list)
    for s in cal_samples:
        by_relation[s["relation_type"]].append(s)

    diag_cal, repair_val, recal = [], [], []
    for rel, samples in sorted(by_relation.items()):
        random.shuffle(samples)
        n = len(samples)
        n_diag = max(1, round(n * 0.5))
        n_repair = max(1, round(n * 0.25))
        diag_cal.extend(samples[:n_diag])
        repair_val.extend(samples[n_diag:n_diag + n_repair])
        recal.extend(samples[n_diag + n_repair:])

    new_splits = {"diag_cal": diag_cal, "repair_val": repair_val, "recal": recal}
    for name, data in new_splits.items():
        out = SPLITS_DIR / f"{name}.jsonl"
        with open(out, "w") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} samples -> {out}")

    # Sanity: no overlap, union == cal
    all_ids = set()
    for data in new_splits.values():
        ids = {s["id"] for s in data}
        assert len(ids & all_ids) == 0, "ID overlap between sub-splits!"
        all_ids |= ids
    original_ids = {s["id"] for s in cal_samples}
    assert all_ids == original_ids, (
        f"Union mismatch: {len(all_ids)} vs {len(original_ids)}")
    print(f"  ✅ No overlap, union == original cal ({len(all_ids)} IDs)")

    # ---- 2. Build ID -> sub-split lookup ----
    id_to_split = {}
    for name, data in new_splits.items():
        for s in data:
            id_to_split[s["id"]] = name

    # ---- 3. Re-split cached model outputs ----
    for model in MODELS:
        for dataset in DATASETS:
            ds_dir = RESULTS_DIR / model / dataset

            # Re-split cal.jsonl (model outputs)
            cal_out = ds_dir / "cal.jsonl"
            if not cal_out.exists():
                continue

            buckets = {"diag_cal": [], "repair_val": [], "recal": []}
            with open(cal_out) as f:
                for line in f:
                    rec = json.loads(line)
                    sid = rec["sample_id"]
                    target = id_to_split.get(sid)
                    if target:
                        rec["split"] = target
                        buckets[target].append(rec)

            for split_name, records in buckets.items():
                p = ds_dir / f"{split_name}.jsonl"
                with open(p, "w") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            total = sum(len(v) for v in buckets.values())
            print(f"  {model}/{dataset}: "
                  f"diag_cal={len(buckets['diag_cal'])}, "
                  f"repair_val={len(buckets['repair_val'])}, "
                  f"recal={len(buckets['recal'])}  (total {total})")

            # Re-split cal_logits.jsonl (if exists)
            logits_file = ds_dir / "cal_logits.jsonl"
            if logits_file.exists():
                logit_buckets = {"diag_cal": [], "repair_val": [], "recal": []}
                with open(logits_file) as f:
                    for line in f:
                        rec = json.loads(line)
                        sid = rec.get("sample_id", rec.get("id"))
                        target = id_to_split.get(sid)
                        if target:
                            logit_buckets[target].append(rec)
                for split_name, records in logit_buckets.items():
                    p = ds_dir / f"{split_name}_logits.jsonl"
                    with open(p, "w") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 4. Verify 5-way no-overlap ----
    print("\n=== Final 5-way overlap check ===")
    split_names = ["train", "diag_cal", "repair_val", "recal", "test"]
    split_ids = {}
    for s in split_names:
        path = SPLITS_DIR / f"{s}.jsonl"
        ids = set()
        with open(path) as f:
            for line in f:
                ids.add(json.loads(line)["id"])
        split_ids[s] = ids
        print(f"  {s}: {len(ids)} samples")

    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = split_ids[a] & split_ids[b]
            assert len(overlap) == 0, f"OVERLAP {a} ∩ {b}: {len(overlap)}"
    total = sum(len(v) for v in split_ids.values())
    print(f"  ✅ No overlap between any of 5 splits (total {total})")


if __name__ == "__main__":
    resplit_calibration()

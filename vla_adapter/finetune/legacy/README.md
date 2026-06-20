# legacy/ — superseded bucket builders

These are the original **two-stage** parking-bucket pipeline, kept for
reference / reproducibility. They are no longer the active path.

| file | role |
|---|---|
| `create_parking_buckets.py`    | stage 1 (v1): classify each frame into phase buckets by route geometry |
| `create_parking_buckets_v2.py` | early split (only pre_turn / turn_execution / swing_out) |
| `create_parking_buckets_v3.py` | stage 2 (v3): split every phase bucket by freeroll/ped/door + speed-split turn_execution_ped |

**Replaced by** `../create_parking_buckets_simple.py`, a single-pass builder
that reads each measurement once and emits the same buckets. It was verified
**byte-identical** (same 37 keys, same per-bucket path lists, same pkl MD5) to
`create_parking_buckets.py` → `create_parking_buckets_v3.py` run back-to-back on
`run_topdown_001`. The phase thresholds (`classify`) and the freeroll/ped/door
`group()` are copied verbatim, so the buckets the BEST parking model
(`2026_06_06_10_21_16_parking_ft_v2`, epoch 9) trained on are reproduced exactly.

Prefer the single-pass builder for all new runs. These remain only so the old
behavior can be re-derived if ever needed.

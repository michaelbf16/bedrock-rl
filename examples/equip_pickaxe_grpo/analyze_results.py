"""Recompute the controls and paired pass@3 sensitivity on the saved dev set."""
from collections import Counter, defaultdict
from itertools import product
import json
from math import comb
from pathlib import Path

from bedrock_rl.env.catalog import resolve_items
from bedrock_rl.env.task import Task


def pass_rows(records):
    """Collapse exactly three successes/failures per frozen prompt."""
    grouped = defaultdict(dict)
    for record in records:
        key = (record["row_index"], record["seed"],
               record["init_yaw"], record["init_pitch"])
        sample = record["sample_index"]
        if record["broken"] or sample in grouped[key]:
            raise ValueError(f"broken or duplicate pass@3 trial for {key}")
        grouped[key][sample] = record["success"]
    if any(set(samples) != {0, 1, 2} for samples in grouped.values()):
        raise ValueError("each pass@3 prompt needs exactly samples 0, 1, 2")
    return {key: any(samples.values()) for key, samples in grouped.items()}


def bootstrap_interval(strata, rows):
    """Exact paired percentile bootstrap, retaining all rows in each seed."""
    # Convolution gives the distribution of the sum over resampled worlds.
    # Each stratum retains its observed count, with replacement within it.
    distribution = Counter({0: 1})
    for values in strata:
        counts = Counter(values)
        for _ in values:
            updated = Counter()
            for total, ways in distribution.items():
                for delta, frequency in counts.items():
                    updated[total + delta] += ways * frequency
            distribution = updated
    size = sum(distribution.values())
    interval = []
    for percentile in (25, 975):
        cumulative = 0
        for total in sorted(distribution):
            cumulative += distribution[total]
            if cumulative * 1000 >= percentile * size:
                interval.append(total / rows)
                break
    return interval


def analyze():
    directory = Path(__file__).resolve().parent
    curves = directory / "curves"

    def read(label):
        return json.loads((curves / f"dev_{label}.records.json").read_text())

    base, final = (pass_rows(read(f"{label}.pass3")) for label in ("base", "step10"))
    if base.keys() != final.keys():
        raise ValueError("pass@3 records do not match on row, seed and pose")
    deltas = Counter()
    transitions = Counter()
    counts = Counter()
    for key, before in base.items():
        seed = key[1]
        counts[seed] += 1
        deltas[seed] += int(final[key]) - int(before)
        transitions[before, final[key]] += 1
    if len(set(counts.values())) != 1:
        raise ValueError("the fixed-denominator bootstrap needs equal rows per seed")
    improved = transitions[False, True]
    lost = transitions[True, False]
    discordant = improved + lost
    row_p = min(1, 2 * sum(comb(discordant, i) for i in range(min(improved, lost) + 1))
                / 2 ** discordant)

    nonzero = [x for x in deltas.values() if x]
    observed = abs(sum(nonzero))
    extreme = sum(abs(sum(sign * value for sign, value in zip(signs, nonzero)))
                  >= observed for signs in product((-1, 1), repeat=len(nonzero)))

    task = Task.load(directory / "task.yaml")
    iron = resolve_items(["iron_pickaxe"])[0]
    slot_strata = defaultdict(list)
    slot_counts = Counter()
    for seed, delta in deltas.items():
        inventory = task.initial_state.inventory_slots(seed)
        slot = next(i + 1 for i, (item, _, _) in inventory.items() if item == iron)
        slot_strata[slot].append(delta)
        slot_counts[slot] += counts[seed]
    greedy = read("step10")
    keys = Counter(json.loads(row["actions"])[0]["k"] for row in greedy)

    return {
        "rows": len(base), "unique_worlds": len(counts),
        "rows_per_world": next(iter(counts.values())),
        "deterministic_successes": {
            label: sum(row["success"] for row in read(label))
            for label in ("base", "step5", "step10")},
        "step10_first_keys": dict(sorted(keys.items())),
        "rows_by_target_slot": dict(sorted(slot_counts.items())),
        "pass3_successes": [sum(base.values()), sum(final.values())],
        "paired_improvements": improved, "paired_regressions": lost,
        "row_exact_mcnemar_p": row_p,
        "nonzero_paired_deltas_by_world": {k: v for k, v in sorted(deltas.items()) if v},
        "whole_world_sign_flip_p": extreme / 2 ** len(nonzero),
        "paired_gain": sum(deltas.values()) / len(base),
        "whole_world_percentile_ci95": bootstrap_interval([list(deltas.values())], len(base)),
        "slot_stratified_percentile_ci95": bootstrap_interval(slot_strata.values(), len(base)),
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))

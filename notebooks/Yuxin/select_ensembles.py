"""Select at most four equal-weight members within one family on development folds."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_experiments import source_hashes, write_json
from tune_models import OUT, TREE_TARGET, code_signature, frames


def select(phase):
    data = json.loads((OUT / f"{phase}_trials.json").read_text())
    assert data["signature"] == code_signature() and data["data_hashes"] == source_hashes()
    complete = sorted([t for t in data["trials"] if t["state"] == "COMPLETE"], key=lambda t:t["value"])
    assert complete
    folds = frames()
    ratios = {}
    for trial in complete:
        ratios[trial["trial"]] = []
        for i, (fold, _, _, _, _) in enumerate(folds):
            with np.load(trial["cache_files"][i], allow_pickle=False) as cached:
                np.testing.assert_array_equal(cached["row_index"], fold.valid_idx)
                ratios[trial["trial"]].append(cached["ratio"].copy())

    def evaluate(numbers):
        scores = {}
        for i, (fold, a, b, _, _) in enumerate(folds):
            ratio = np.mean([ratios[n][i] for n in numbers], axis=0)
            level = float(a.loc[a.RENT_APPROVAL_DATE.eq(a.RENT_APPROVAL_DATE.max()), "MONTHLY_RENT"].median())
            prediction = np.maximum(ratio*level, 1.)
            scores[fold.definition.name] = float(np.sqrt(np.mean((b.MONTHLY_RENT.to_numpy()-prediction)**2)))
        return {"members": numbers, "n_members": len(numbers), "fold_rmse": scores,
                "mean_rmse": float(np.mean(list(scores.values())))}

    maximum = min(4, len(complete))
    candidates = [dict(evaluate([t["trial"] for t in complete[:count]]), method="top_k")
                  for count in range(1, maximum+1)]
    chosen = [complete[0]["trial"]]
    forward_evaluations = 0
    for count in range(2, maximum+1):
        additions = [evaluate(chosen+[t["trial"]]) for t in complete if t["trial"] not in chosen]
        forward_evaluations += len(additions)
        candidate = min(additions, key=lambda c:c["mean_rmse"])
        chosen = candidate["members"]
        candidates.append(dict(candidate, method="greedy_forward"))
    winner = min(candidates, key=lambda c:c["mean_rmse"])
    files = []
    for i, (fold, a, _, _, _) in enumerate(folds):
        ratio = np.mean([ratios[n][i] for n in winner["members"]], axis=0)
        level = float(a.loc[a.RENT_APPROVAL_DATE.eq(a.RENT_APPROVAL_DATE.max()), "MONTHLY_RENT"].median())
        file = Path(complete[0]["cache_files"][i]).parent / f"{phase}_selected_{i}.npz"
        np.savez_compressed(file, prediction=np.maximum(ratio*level, 1.), ratio=ratio, row_index=fold.valid_idx)
        files.append(str(file))
    lookup = {t["trial"]: t for t in complete}
    params = {"members": [lookup[n]["recipe"] for n in winner["members"]],
              "encoder": "carry", "random_state": 5228}
    selected = {**winner, "cache_files": files, "parameters": params, "class": "ensemble", "phase": phase,
                "trial": winner["method"]+"_"+str(winner["n_members"]), "signature": code_signature(),
                "data_hashes": data["data_hashes"], "beats_best_tree": winner["mean_rmse"] < TREE_TARGET,
                "selection": "best K=1..4 equal-weight top-K or greedy-forward candidates within one family",
                "selection_bias": "ranking and K selected on public development folds, not independent test",
                "candidates": candidates, "forward_candidate_evaluations": forward_evaluations}
    write_json(OUT / f"selected_{phase}.json", selected)
    print(json.dumps({"phase": phase, "selected": winner, "beats_tree": selected["beats_best_tree"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["carry", "transformer"], required=True)
    select(parser.parse_args().phase)

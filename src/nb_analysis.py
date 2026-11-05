import json, glob, os, re
import pandas as pd
from math_verify import parse, verify
from loopdiag import last_boxed, boxed_spans, loop_onset, split_final

def is_correct(pred, gold):
    if pred is None:
        return False
    try:
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return False

def classify(r):
    reasoning, final = split_final(r["text"], r["thinking"])
    pred = last_boxed(final)
    correct = is_correct(pred, r["answer"])
    onset = loop_onset(r["token_ids"])
    if correct:
        cat = "correct"
    elif r["finish_reason"] == "length":
        cat = "truncated_loop" if onset is not None else "truncated_noloop"
    elif pred is None:
        cat = "no_answer"
    else:
        cat = "wrong"
    # Did a correct answer already appear somewhere before the model went off track?
    cands = boxed_spans(reasoning) + re.findall(r"answer is[:\s]*\$?([^\n$]{1,40})", reasoning)
    seen = (not correct) and any(is_correct(c.strip(" .$"), r["answer"]) for c in cands[-20:])
    return dict(condition=r["condition"], unique_id=r["unique_id"], level=r["level"],
                subject=r["subject"], n_tokens=len(r["token_ids"]), finish=r["finish_reason"],
                pred=pred, correct=correct, category=cat, loop_onset=onset,
                correct_seen_earlier=seen)

rows = [json.loads(l) for p in sorted(glob.glob(f"{OUT}/gen_*.jsonl")) for l in open(p)]
df = pd.DataFrame([classify(r) for r in rows])

probes = [json.loads(l) for p in sorted(glob.glob(f"{OUT}/probe_*.jsonl")) for l in open(p)]
if probes:
    pr = pd.DataFrame(probes)
    df = df.merge(pr[["condition", "unique_id", "probe_pred"]], on=["condition", "unique_id"], how="left")
    gold = {r["unique_id"]: r["answer"] for r in rows}
    df["probe_correct"] = [is_correct(p, gold[u]) if isinstance(p, str) else False
                           for p, u in zip(df.probe_pred, df.unique_id)]
else:
    df["probe_correct"] = False
df["correct_after_probe"] = df.correct | df.probe_correct

summary = df.groupby("condition").agg(
    n=("correct", "size"),
    accuracy=("correct", "mean"),
    acc_after_forced_answer=("correct_after_probe", "mean"),
    truncated=("finish", lambda s: (s == "length").mean()),
    looping=("loop_onset", lambda s: s.notna().mean()),
    correct_seen_earlier=("correct_seen_earlier", "mean"),
    mean_tokens=("n_tokens", "mean"),
)
cats = pd.crosstab(df.condition, df.category, normalize="index")
display((summary * [1, 100, 100, 100, 100, 100, 1]).round(1).rename(columns=lambda c: c if c in ("n", "mean_tokens") else c + " %"))
display((cats * 100).round(1))
df.to_csv(f"{OUT}/per_problem.csv", index=False)


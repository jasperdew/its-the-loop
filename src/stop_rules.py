"""Step 1: simulate simple stopping rules offline on finished generations (no GPU).

For every trace we walk through the answers the model writes while reasoning (\\boxed{...} or
"the answer is ...") and ask: if we had stopped at token t and taken the latest answer so far,
would it have been correct, and how many tokens would we have spent?

Rules:
  none         the model's own output (baseline)
  loop_stop    stop at loop onset (distinct 20-gram ratio < 0.5), take the latest answer so far
  stable_k     stop as soon as the same answer has appeared k times, take that answer
  combo_k      whichever of loop_stop / stable_k fires first

Usage: python src/stop_rules.py results/<run>/ [--tokenizer Qwen/Qwen3.5-0.8B]
"""
import argparse, glob, json, os, re, sys
import pandas as pd
from math_verify import parse, verify
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from loopdiag import boxed_spans, last_boxed, loop_onset, split_final  # copied from notebook 01

ANSWER_IS = re.compile(r"answer is[:\s]*\$?([^\n$]{1,40}?)(?=\.\s|\.$|,|;|\$|\n|<|$)")


def is_correct(pred, gold):
    if pred is None:
        return False
    try:
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return False


def norm(a):
    a = a.strip(" .$")
    for s in ("\\left", "\\right", "\\!", "\\,", " ", "\\displaystyle"):
        a = a.replace(s, "")
    return a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


def candidates(text):
    """(char_end, answer) for every answer the model states, in order of appearance."""
    out = []
    i = 0
    while (i := text.find("\\boxed", i)) >= 0:
        span = boxed_spans(text[i:])
        if span:
            out.append((i + len("\\boxed{") + len(span[0]) + 1, span[0]))
        i += 6
    out += [(m.end(), m.group(1)) for m in ANSWER_IS.finditer(text)]
    return sorted(out)


def simulate(r, tok, ks=(2, 3)):
    text, ids = r["text"], r["token_ids"]
    _, final = split_final(text, r["thinking"])
    n = len(ids)
    res = {"none": (r["finish_reason"] == "stop" and is_correct(last_boxed(final), r["answer"]), n)}

    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    char2tok = lambda c: next((j for j, (_, e) in enumerate(enc["offset_mapping"]) if e >= c), n)
    cands = [(char2tok(c), a) for c, a in candidates(text)]

    onset = loop_onset(ids)
    def latest_before(t):
        prior = [a for p, a in cands if p <= t]
        return prior[-1] if prior else None
    t_loop = onset if onset is not None else None
    res["loop_stop"] = ((is_correct(latest_before(t_loop), r["answer"]), t_loop) if t_loop is not None
                        else res["none"])

    for k in ks:
        seen, t_stable, a_stable = {}, None, None
        for p, a in cands:
            seen[norm(a)] = seen.get(norm(a), 0) + 1
            if seen[norm(a)] >= k:
                t_stable, a_stable = p, a
                break
        res[f"stable_{k}"] = ((is_correct(a_stable, r["answer"]), t_stable) if t_stable is not None
                              else res["none"])
        firsts = [t for t in (t_loop, t_stable) if t is not None]
        if not firsts:
            res[f"combo_{k}"] = res["none"]
        elif t_stable is not None and t_stable == min(firsts):
            res[f"combo_{k}"] = res[f"stable_{k}"]
        else:
            res[f"combo_{k}"] = res["loop_stop"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = []
    for path in sorted(glob.glob(os.path.join(a.run_dir, "gen_*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            for rule, (ok, t) in simulate(r, tok).items():
                rows.append(dict(condition=r["condition"], unique_id=r["unique_id"], rule=rule,
                                 correct=ok, tokens=t))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.run_dir, "stop_rules_per_problem.csv"), index=False)
    t = df.groupby(["condition", "rule"]).agg(accuracy=("correct", "mean"), mean_tokens=("tokens", "mean"))
    base = t.xs("none", level="rule").mean_tokens
    t["tokens_saved_%"] = [100 * (1 - mt / base[c]) for (c, _), mt in zip(t.index, t.mean_tokens)]
    t["accuracy"] *= 100
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()

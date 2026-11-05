"""Step 2: checkpoint probes.

Every EVERY tokens of each finished trace, close the reasoning and ask for the boxed answer
(greedy, 48 tokens). This gives, per trace, the answer the model would have given had we stopped
at token t. The best stopping time per trace is the upper bound for any stopping policy.
Prefix caching makes this cheap: all checkpoints of one trace share their prefix.

Usage: python ckptprobe.py --model TAG --out results
"""
import argparse, glob, json, os
from loopdiag import _llm, _sampling, last_boxed
from loopsurvey import MODELS

EVERY = 512


def run(tag, out, max_tokens, seed, dtype, pattern=None):
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt
    m = MODELS[tag]
    trc = m.get("trust_remote_code", False)
    tok = AutoTokenizer.from_pretrained(m["repo"], trust_remote_code=trc)
    llm = _llm(m["repo"], max_tokens, seed, dtype, trc)
    sp = _sampling(dict(temperature=0.0), 48, seed, stop=["}$", "}\n", "$\n"])
    suffix = "The final answer is $\\boxed{"
    for path in sorted(glob.glob(os.path.join(out, pattern or f"gen_{tag}__*.jsonl"))):
        rows = [json.loads(l) for l in open(path)]
        keys, reqs = [], []
        for r in rows:
            p_ids = tok(r["prompt"], add_special_tokens=False).input_ids
            n = len(r["token_ids"])
            for t in sorted(set(list(range(EVERY, n, EVERY)) + [n])):
                ctx = r["token_ids"][:t]
                closed = "</think>" in tok.decode(ctx)
                close = "\n</think>\n\n" if r["thinking"] and not closed else "\n\n"
                ids = p_ids + ctx + tok(close + suffix, add_special_tokens=False).input_ids
                keys.append((r["unique_id"], t))
                reqs.append(TokensPrompt(prompt_token_ids=ids))
        outs = llm.generate(reqs, sp)
        name = os.path.basename(path)[len("gen_"):-len(".jsonl")]
        with open(os.path.join(out, f"ckpt_{name}.jsonl"), "w") as f:
            for (uid, t), o in zip(keys, outs):
                gen = o.outputs[0].text
                pred = last_boxed("\\boxed{" + gen + ("}" if "}" not in gen else ""))
                f.write(json.dumps(dict(unique_id=uid, t=t, pred=pred)) + "\n")
        print(f"[{name}] {len(reqs)} checkpoint probes", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--glob", default=None)  # trace files to probe, relative to --out
    a = ap.parse_args()
    run(a.model, a.out, a.max_tokens, a.seed, a.dtype, a.glob)

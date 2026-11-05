"""Loop survey: generation + forced-answer probe for several small models on a MATH-500 subset.

One subprocess per model (python loopsurvey.py --model TAG) so GPU memory is freed between models.
Writes results/gen_{TAG}__{COND}.jsonl and results/probe_{TAG}__{COND}.jsonl; the analysis code
from notebook 01 reads these unchanged because the condition field is "{TAG}__{COND}".
"""
import argparse, json, os, time
from loopdiag import _llm, _sampling, split_final, last_boxed, loop_onset, PROMPT_SUFFIX

# thinking: the model reasons inside <think>...</think> before answering.
# chat_kwargs: passed to apply_chat_template (e.g. enable_thinking for Qwen3/3.5).
# rec: sampling recommended on the model card for reasoning/math.
# Picked from the 2026-11 inventory of the strongest open models <=2.5B (scores are vendor-reported).
QWEN_THINK = dict(enable_thinking=True)
MODELS = {
    # rec = exact MIT "Weight-Space Programming" eval settings, so our baseline matches their 7.9%
    "qwen3.5-0.8b": dict(repo="Qwen/Qwen3.5-0.8B", thinking=True, chat_kwargs=QWEN_THINK,
                         rec=dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)),
    "qwen3.5-2b":   dict(repo="Qwen/Qwen3.5-2B", thinking=True, chat_kwargs=QWEN_THINK,
                         rec=dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)),
    "minicpm5-1b":  dict(repo="openbmb/MiniCPM5-1B", thinking=True, chat_kwargs=QWEN_THINK, trust_remote_code=True,
                         rec=dict(temperature=0.9, top_p=0.95)),
    "minicpm5-2b":  dict(repo="openbmb/MiniCPM5-2B", thinking=True, chat_kwargs=QWEN_THINK, trust_remote_code=True,
                         rec=dict(temperature=1.0, top_p=0.95)),
    "justrl-1.5b":  dict(repo="hbx/JustRL-Nemotron-1.5B", thinking=True,
                         rec=dict(temperature=0.7, top_p=0.9)),
    "r1-distill-1.5b": dict(repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", thinking=True,
                            rec=dict(temperature=0.6, top_p=0.95)),
}

# rec  = recommended sampling, plain prompt
# crit = same sampling, plus a written stopping criterion in the prompt (no other change)
CONDS = ["rec", "crit"]
CRITERION = ("\nStopping rule for your reasoning: once you have a candidate answer, verify it once. "
             "If the check agrees, stop thinking and give the final answer. "
             "Only keep reasoning if you found a specific, concrete error.")


def run(tag, n_step, max_tokens, seed, out, dtype):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt
    m = MODELS[tag]
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(0, 500, n_step))
    tok = AutoTokenizer.from_pretrained(m["repo"], trust_remote_code=m.get("trust_remote_code", False))
    llm = _llm(m["repo"], max_tokens, seed, dtype, m.get("trust_remote_code", False))
    probe_sp = _sampling(dict(temperature=0.0), 48, seed, stop=["}$", "}\n", "$\n"])
    for cond in CONDS:
        c = m["rec"]
        suffix = PROMPT_SUFFIX + (CRITERION if cond == "crit" else "")
        name = f"{tag}__{cond}"
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": ex["problem"] + suffix}],
            tokenize=False, add_generation_prompt=True, **m.get("chat_kwargs", {})) for ex in ds]
        t0 = time.time()
        outs = llm.generate(prompts, _sampling(c, max_tokens, seed))
        dt = time.time() - t0
        rows = []
        for ex, p, o in zip(ds, prompts, outs):
            g = o.outputs[0]
            rows.append(dict(condition=name, thinking=m["thinking"], unique_id=ex["unique_id"],
                             level=ex["level"], subject=ex["subject"], answer=ex["answer"],
                             prompt=p, text=g.text, token_ids=list(g.token_ids),
                             finish_reason=g.finish_reason))
        with open(os.path.join(out, f"gen_{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        n_tok = sum(len(r["token_ids"]) for r in rows)
        print(f"[{name}] {len(rows)} problems, {n_tok:,} tokens, {dt/60:.1f} min", flush=True)

        # Forced-answer probe on outputs without a final answer (same logic as notebook 01).
        todo, reqs = [], []
        for r in rows:
            _, final = split_final(r["text"], r["thinking"])
            if last_boxed(final) is not None and r["finish_reason"] == "stop":
                continue
            onset = loop_onset(r["token_ids"])
            ctx = r["token_ids"][:onset] if onset is not None else r["token_ids"]
            close = "\n</think>\n\n" if r["thinking"] and "</think>" not in tok.decode(ctx) else "\n\n"
            ids = tok(r["prompt"], add_special_tokens=False).input_ids + ctx + \
                tok(close + "The final answer is $\\boxed{", add_special_tokens=False).input_ids
            todo.append((r, onset))
            reqs.append(TokensPrompt(prompt_token_ids=ids))
        pouts = llm.generate(reqs, probe_sp) if reqs else []
        with open(os.path.join(out, f"probe_{name}.jsonl"), "w") as f:
            for (r, onset), o in zip(todo, pouts):
                gen = o.outputs[0].text
                pred = last_boxed("\\boxed{" + gen + ("}" if "}" not in gen else ""))
                f.write(json.dumps(dict(condition=name, unique_id=r["unique_id"], cut_at=onset,
                                        probe_raw=gen, probe_pred=pred)) + "\n")
        print(f"[{name}] probed {len(todo)} outputs without a final answer", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n-step", type=int, default=5)  # every 5th problem -> 100 problems
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dtype", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run(a.model, a.n_step, a.max_tokens, a.seed, a.out, a.dtype)

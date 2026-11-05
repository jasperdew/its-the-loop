"""Loop diagnosis for small reasoning models on MATH-500.

Generation runs in a subprocess (python loopdiag.py ...) so the notebook kernel
never imports torch/vLLM. Analysis helpers are importable without a GPU.
"""
import argparse, json, os, re, time

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# Sampling settings: "rec" = Qwen3.5 model-card recommendation for text tasks.
CONDITIONS = {
    "nothink_greedy": dict(thinking=False, temperature=0.0),
    "think_greedy":   dict(thinking=True,  temperature=0.0),
    "nothink_rec":    dict(thinking=False, temperature=1.0, top_p=1.0,  top_k=20, presence_penalty=2.0),
    "think_rec":      dict(thinking=True,  temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5),
}

# ---------- analysis helpers (no GPU needed) ----------

def boxed_spans(s):
    """All \\boxed{...} contents in order, with balanced braces."""
    out, i = [], 0
    while True:
        i = s.find("\\boxed", i)
        if i < 0:
            return out
        j = s.find("{", i)
        if j < 0:
            return out
        depth = 0
        for k in range(j, len(s)):
            if s[k] == "{":
                depth += 1
            elif s[k] == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[j + 1:k])
                    break
        i = j + 1


def last_boxed(s):
    b = boxed_spans(s)
    return b[-1] if b else None


def distinct_ratio(ids, n=20):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def loop_onset(ids, win=512, step=128, n=20, thr=0.5):
    """First token index where a window becomes mostly repeated n-grams, else None."""
    for start in range(0, max(1, len(ids) - win + 1), step):
        if len(ids) >= win and distinct_ratio(ids[start:start + win], n) < thr:
            return start
    return None


def split_final(text, thinking):
    """(reasoning, final_region). In thinking mode the answer must come after </think>."""
    think, sep, after = text.partition("</think>")
    if sep:
        return think, after
    return (text, "") if thinking else ("", text)


# ---------- generation ----------

def _llm(model, max_tokens, seed, dtype="auto", trust_remote_code=False):
    from vllm import LLM
    if dtype == "auto":  # T4 (sm75) has no bf16; L4/A100 do
        import torch
        dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
    print(f"vLLM dtype={dtype}", flush=True)
    return LLM(model=model, dtype=dtype, max_model_len=max_tokens + 2048,
               gpu_memory_utilization=0.90, seed=seed,
               trust_remote_code=trust_remote_code,
               limit_mm_per_prompt={"image": 0, "video": 0})


def _sampling(c, max_tokens, seed, **extra):
    from vllm import SamplingParams
    kw = {k: c[k] for k in ("temperature", "top_p", "top_k", "presence_penalty") if k in c}
    return SamplingParams(max_tokens=max_tokens, seed=seed, **kw, **extra)


def generate(args):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(args.n))
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = _llm(args.model, args.max_tokens, args.seed, args.dtype)
    for name in args.conditions.split(","):
        c = CONDITIONS[name]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": ex["problem"] + PROMPT_SUFFIX}],
            tokenize=False, add_generation_prompt=True, enable_thinking=c["thinking"]) for ex in ds]
        t0 = time.time()
        outs = llm.generate(prompts, _sampling(c, args.max_tokens, args.seed))
        dt = time.time() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        path = os.path.join(args.out, f"gen_{name}.jsonl")
        with open(path, "w") as f:
            for ex, p, o in zip(ds, prompts, outs):
                g = o.outputs[0]
                f.write(json.dumps(dict(
                    condition=name, thinking=c["thinking"], unique_id=ex["unique_id"],
                    level=ex["level"], subject=ex["subject"], answer=ex["answer"],
                    prompt=p, text=g.text, token_ids=list(g.token_ids),
                    finish_reason=g.finish_reason)) + "\n")
        print(f"[{name}] {len(outs)} problems, {n_tok:,} tokens, {dt/60:.1f} min "
              f"({n_tok/dt:,.0f} tok/s) -> {path}", flush=True)


def probe(args):
    """Forced-answer probe: cut each non-final output at loop onset (or at its end),
    close the reasoning and ask for the boxed answer. Measures accuracy locked in loops."""
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = _llm(args.model, args.max_tokens, args.seed, args.dtype)
    sp = _sampling(dict(temperature=0.0), 48, args.seed, stop=["}$", "}\n", "$\n"])
    for name in args.conditions.split(","):
        rows = [json.loads(l) for l in open(os.path.join(args.out, f"gen_{name}.jsonl"))]
        todo, reqs = [], []
        for r in rows:
            reasoning, final = split_final(r["text"], r["thinking"])
            if last_boxed(final) is not None and r["finish_reason"] == "stop":
                continue  # model gave a final answer itself; nothing to rescue
            ids = r["token_ids"]
            onset = loop_onset(ids)
            ctx = ids[:onset] if onset is not None else ids
            close = "\n</think>\n\n" if r["thinking"] and "</think>" not in tok.decode(ctx) else "\n\n"
            suffix = close + "The final answer is $\\boxed{"
            prompt_ids = tok(r["prompt"], add_special_tokens=False).input_ids + ctx + \
                tok(suffix, add_special_tokens=False).input_ids
            todo.append((r, onset))
            reqs.append(TokensPrompt(prompt_token_ids=prompt_ids))
        outs = llm.generate(reqs, sp) if reqs else []
        path = os.path.join(args.out, f"probe_{name}.jsonl")
        with open(path, "w") as f:
            for (r, onset), o in zip(todo, outs):
                gen = o.outputs[0].text
                pred = last_boxed("\\boxed{" + gen + ("}" if "}" not in gen else ""))
                f.write(json.dumps(dict(condition=name, unique_id=r["unique_id"],
                                        cut_at=onset, probe_raw=gen, probe_pred=pred)) + "\n")
        print(f"[{name}] probed {len(todo)} non-final outputs -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "probe"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dtype", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    generate(a) if a.mode == "generate" else probe(a)

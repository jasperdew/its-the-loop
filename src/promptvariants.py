"""Text-only loop control for small reasoning models: three concepts, all without training.

A  terse reasoning   instruction up front to reason in short notes (more reasoning per token)
B  stopping rule     instruction up front to stop once the answer checks out
C  injected nudges   generation runs in 512-token chunks; at the right moment one short line is
                     inserted into the model's own reasoning, then generation continues:
                       C1 loop    when a repetition loop is detected
                       C2 answer  when the model writes a new candidate answer (mirrors it back)
                       C3 period  every 1024 tokens

Writes results/gen_{variant}.jsonl in the format of notebook 01 (token_ids include injected text).
Usage: python promptvariants.py --out results
"""
import argparse, json, os, re, time
from loopdiag import _llm, _sampling, PROMPT_SUFFIX, boxed_spans, loop_onset

# Up-front instructions, kept as short as possible: every token is paid on every problem.
STATIC = {
    "base":         "",
    "terse1":       "\nThink in terse notes: equations, no prose, never restate.",
    "terse2":       "\nReason tersely. Short steps. No repetition.",
    "stop1":        "\nVerify once, then stop.",
    "stop2":        "\nStop thinking once your answer checks out.",
    "terse_stop1":  "\nThink in terse notes. Verify once, then stop.",
    "terse_stop2":  "\nReason tersely. Stop once your answer checks out.",
}
DYNAMIC = ["inject_loop", "inject_answer", "inject_period"]
CHUNK = 512

# Exact sampling of the MIT Weight-Space Programming evaluation of Qwen3.5-0.8B.
MIT_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)

ANSWER_IS = re.compile(r"answer is[:\s]*\$?([^\n$]{1,40}?)(?=\.\s|\.$|,|;|\$|\n|<|$)")


def new_answers(text):
    return [a.strip(" .$") for a in boxed_spans(text) + ANSWER_IS.findall(text)]


def record(v, ex, prompt, ids, finish, n_instr, tok):
    return dict(condition=v, thinking=True, unique_id=ex["unique_id"], level=ex["level"],
                subject=ex["subject"], answer=ex["answer"], prompt=prompt, text=tok.decode(ids),
                token_ids=ids, finish_reason=finish, instruction_tokens=n_instr)


def run_static(llm, tok, ds, names, a):
    prompts, meta = [], []
    for v in names:
        for ex in ds:
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": ex["problem"] + PROMPT_SUFFIX + STATIC[v]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True))
            meta.append((v, ex))
    outs = llm.generate(prompts, _sampling(MIT_SAMPLING, a.max_tokens, a.seed))
    n_instr = {v: len(tok(STATIC[v], add_special_tokens=False).input_ids) for v in names}
    return [record(v, ex, p, list(o.outputs[0].token_ids), o.outputs[0].finish_reason, n_instr[v], tok)
            for (v, ex), p, o in zip(meta, prompts, outs)]


def nudge(v, chunk_text, gen_ids, state):
    """Line to insert after this chunk, or None."""
    if v == "inject_loop":
        if len(gen_ids) - state["last"] >= 1024 and loop_onset(gen_ids[-1024:]) is not None:
            return "\nI'm going in circles. Commit to my best answer.\n"
    elif v == "inject_answer":
        fresh = [x for x in new_answers(chunk_text) if x and x not in state["seen"]]
        if fresh:
            state["seen"].update(fresh)
            return f"\n(Answer: {fresh[-1]}. Stop unless I find an error.)\n"
    elif v == "inject_period":
        if len(gen_ids) - state["last"] >= 1024:
            return "\n(Check: do I have the answer? If yes, stop.)\n"
    return None


def run_dynamic(llm, tok, ds, names, a):
    from vllm.inputs import TokensPrompt
    jobs = []
    for v in names:
        for ex in ds:
            p = tok.apply_chat_template([{"role": "user", "content": ex["problem"] + PROMPT_SUFFIX}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=True)
            jobs.append(dict(v=v, ex=ex, prompt=p, p_ids=tok(p, add_special_tokens=False).input_ids,
                             gen=[], done=None, n_instr=0, state=dict(last=0, seen=set())))
    rnd = 0
    while active := [j for j in jobs if j["done"] is None]:
        reqs = [TokensPrompt(prompt_token_ids=j["p_ids"] + j["gen"]) for j in active]
        sps = [_sampling(MIT_SAMPLING, min(CHUNK, a.max_tokens - len(j["gen"])), a.seed + rnd) for j in active]
        for j, o in zip(active, llm.generate(reqs, sps, use_tqdm=False)):
            out = o.outputs[0]
            j["gen"] += list(out.token_ids)
            if out.finish_reason == "stop":
                j["done"] = "stop"
            elif len(j["gen"]) >= a.max_tokens:
                j["done"] = "length"
            elif "</think>" not in tok.decode(j["gen"]):  # only nudge while still thinking
                line = nudge(j["v"], out.text, j["gen"], j["state"])
                if line:
                    ids = tok(line, add_special_tokens=False).input_ids
                    j["gen"] += ids
                    j["n_instr"] += len(ids)
                    j["state"]["last"] = len(j["gen"])
        rnd += 1
        print(f"round {rnd}: {len(active)} active", flush=True)
    return [record(j["v"], j["ex"], j["prompt"], j["gen"], j["done"], j["n_instr"], tok) for j in jobs]


def main(a):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(0, 500, a.n_step))
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = _llm(a.model, a.max_tokens, a.seed, a.dtype)
    names = a.variants.split(",")
    rows = []
    for kind, sel, fn in (("static", [v for v in names if v in STATIC], run_static),
                          ("dynamic", [v for v in names if v in DYNAMIC], run_dynamic)):
        if sel:
            t0 = time.time()
            got = fn(llm, tok, ds, sel, a)
            rows += got
            for v in sel:
                with open(os.path.join(a.out, f"gen_{v}.jsonl"), "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in got if r["condition"] == v)
            print(f"{kind}: {len(got)} generations in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--variants", default=",".join(list(STATIC) + DYNAMIC))
    ap.add_argument("--n-step", type=int, default=5)  # every 5th problem -> 100 problems
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dtype", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    main(a)

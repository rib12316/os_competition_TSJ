#!/usr/bin/env python3
"""F1 C8 before/after 吞吐基准（受控并发负载，隔离 C8 效应）。

固定 workload（num_prompts × input_len × output_len × concurrency），只换 KV dtype
（bf16 baseline vs int8 C8），比聚合 output tok/s、e2e 延迟、成功率。与上游 PR #7474
验证 C8 用的 random_bench 同型。显存（KV 容量）从引擎启动日志读（更干净）。

用法：
  python scripts/bench_c8.py --url http://127.0.0.1:8001/v1 --model probe \
      --num-prompts 48 --input-len 512 --output-len 128 --concurrency 16
"""
from __future__ import annotations
import argparse, json, statistics, time, urllib.request, concurrent.futures as cf

# 长得像文本的填充（内容不影响吞吐；KV 按 token 数填充）
_FILL = ("The quick brown fox jumps over the lazy dog. "
         "In an e-commerce system, returns are processed within 30 days. ") * 64


def _request(url, model, prompt, out_len):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": out_len, "temperature": 0.7, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(url + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    dt = time.time() - t0
    tok = r.get("usage", {}).get("completion_tokens", 0)
    return dt, tok, (r.get("choices", [{}])[0].get("finish_reason") if r.get("choices") else "err")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=48)
    ap.add_argument("--input-len", type=int, default=512, help="近似 prompt token 数")
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    # 造 prompt：重复填充到约 input_len 词（粗略 ≈ token）
    words = _FILL.split()
    prompt = " ".join((words * (args.input_len // len(words) + 1))[:args.input_len])

    prompts = [prompt] * args.num_prompts
    print(f"[bench] {args.num_prompts} prompts × in~{args.input_len}tok × out{args.output_len}tok, "
          f"concurrency={args.concurrency}", flush=True)

    lats, toks, ok = [], 0, 0
    t_start = time.time()
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        futs = [pool.submit(_request, args.url, args.model, p, args.output_len) for p in prompts]
        for f in cf.as_completed(futs):
            try:
                dt, tok, fr = f.result()
                lats.append(dt); toks += tok
                if fr != "error":
                    ok += 1
            except Exception as e:  # noqa: BLE001
                lats.append(300.0)
                print(f"[bench] req failed: {e}", flush=True)
    wall = time.time() - t_start

    lats.sort()
    def pct(p):
        return lats[min(len(lats) - 1, int(len(lats) * p))]
    print("---- RESULT ----")
    print(f"completed_ok    : {ok}/{args.num_prompts}")
    print(f"wall_time_s     : {wall:.2f}")
    print(f"output_tokens   : {toks}")
    print(f"agg_output_tps  : {toks / wall:.1f} tok/s   <- 主吞吐指标")
    print(f"e2e_latency_p50 : {pct(0.50):.2f}s")
    print(f"e2e_latency_p95 : {pct(0.95):.2f}s")
    # 机器可解析行
    print(f"METRIC\tcompleted_ok\t{ok}")
    print(f"METRIC\twall_time_s\t{wall:.2f}")
    print(f"METRIC\toutput_tokens\t{toks}")
    print(f"METRIC\tagg_output_tps\t{toks / wall:.1f}")
    print(f"METRIC\te2e_p50\t{pct(0.50):.2f}")
    print(f"METRIC\te2e_p95\t{pct(0.95):.2f}")


if __name__ == "__main__":
    main()

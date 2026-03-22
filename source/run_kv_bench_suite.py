import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


TASK_REPEAT = {
    "gov_report": 3,
    "lgbench": 2,
}

TASK_MAX_GEN = {
    "gov_report": 512,
    "lgbench": 16384,
}

MODEL_ALIASES = {
    "llama3.1-8b-chat-8k": "llama-3.1-chat-8b",
    "llama-3.1-8b-chat-8k": "llama-3.1-chat-8b",
    "llama-3.1-chat-8b": "llama-3.1-chat-8b",
    "qwen-2.5-7b": "qwen-2.5-chat-7b",
    "qwen-2.5-chat-7b": "qwen-2.5-chat-7b",
}


def parse_csv_str(v: str):
    return [x.strip() for x in v.split(",") if x.strip()]


def parse_csv_int(v: str):
    out = []
    for x in parse_csv_str(v):
        out.append(int(x))
    return out


def resolve_model_names(user_models, model2path):
    resolved = []
    for m in user_models:
        key = MODEL_ALIASES.get(m, m)
        if key not in model2path:
            raise ValueError(
                f"Model '{m}' (resolved='{key}') not found in config/model2path.json. "
                f"Available keys: {sorted(model2path.keys())}"
            )
        resolved.append(key)
    return resolved


def build_plan(models, tasks, bszs):
    impls = ["freekv", "echokv"]
    plan = []
    for model in models:
        for task in tasks:
            if task not in TASK_REPEAT:
                raise ValueError(
                    f"Unsupported task '{task}'. Supported: {sorted(TASK_REPEAT.keys())}"
                )
            reps = TASK_REPEAT[task]
            max_gen = TASK_MAX_GEN[task]
            for bsz in bszs:
                for impl in impls:
                    for run_idx in range(1, reps + 1):
                        plan.append(
                            {
                                "model": model,
                                "task": task,
                                "bsz": int(bsz),
                                "impl": impl,
                                "run_idx": run_idx,
                                "reps": reps,
                                "max_gen": int(max_gen),
                            }
                        )
    return plan


def run_one(cmd, cwd: Path, log_path: Path, dry_run: bool):
    text_cmd = " ".join(cmd)
    print(f"[Run] {text_cmd}")
    if dry_run:
        return 0

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    elapsed_s = time.perf_counter() - start

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {text_cmd}\n")
        f.write(f"\n[exit_code] {proc.returncode}\n")
        f.write(f"[elapsed_s] {elapsed_s:.3f}\n\n")
        f.write("[stdout]\n")
        f.write(proc.stdout or "")
        f.write("\n[stderr]\n")
        f.write(proc.stderr or "")

    if proc.returncode != 0:
        print(f"[Fail] exit={proc.returncode}, log={log_path}")
    else:
        print(f"[Done] {elapsed_s:.2f}s, log={log_path}")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch benchmark runner for FreeKV(topk) vs EchoKV(token). "
            "gov_report repeats=3, lgbench repeats=2."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default="llama3.1-8b-chat-8k,qwen-2.5-7b",
        help="Comma-separated model keys or aliases.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gov_report,lgbench",
        help="Comma-separated tasks. Supported: gov_report,lgbench",
    )
    parser.add_argument(
        "--bszs",
        type=str,
        default="1,2,4",
        help="Comma-separated repeat_bsz values.",
    )
    parser.add_argument(
        "--data_idx",
        type=int,
        default=0,
        help="Single sample index for pred.py (default: 0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup rounds for each run.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32000,
        help="Input max length for pred.py.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable path.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional tag for this batch run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands only, do not execute.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    source_dir = script_path.parent
    project_root = source_dir.parent
    model_cfg_path = project_root / "config" / "model2path.json"
    pred_script = project_root / "source" / "pred.py"

    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Not found: {model_cfg_path}")
    if not pred_script.exists():
        raise FileNotFoundError(f"Not found: {pred_script}")

    with model_cfg_path.open("r", encoding="utf-8") as f:
        model2path = json.load(f)

    user_models = parse_csv_str(args.models)
    models = resolve_model_names(user_models, model2path)
    tasks = parse_csv_str(args.tasks)
    bszs = parse_csv_int(args.bszs)

    plan = build_plan(models, tasks, bszs)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.run_tag.strip() if args.run_tag else "default"
    run_root = project_root / "tmp_res" / "bench_suite" / f"{ts}_{tag}"
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "pred_script": str(pred_script),
        "models": models,
        "tasks": tasks,
        "bszs": bszs,
        "plan_size": len(plan),
        "temperature": args.temperature,
        "warmup": args.warmup,
        "max_length": args.max_length,
        "data_idx": args.data_idx,
        "dry_run": bool(args.dry_run),
    }
    with (run_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[Suite] total runs: {len(plan)}")
    print(f"[Suite] run root : {run_root}")

    failed = []
    for idx, item in enumerate(plan, start=1):
        model = item["model"]
        task = item["task"]
        bsz = item["bsz"]
        impl = item["impl"]
        run_idx = item["run_idx"]
        max_gen = item["max_gen"]

        cmd = [
            args.python,
            str(pred_script),
            "--model",
            model,
            "--dataset",
            task,
            "--repeat_bsz",
            str(bsz),
            "--max_gen",
            str(max_gen),
            "--max_length",
            str(args.max_length),
            "--temperature",
            str(args.temperature),
            "--warmup",
            str(args.warmup),
            "--data_idx",
            str(args.data_idx),
        ]
        if impl == "echokv":
            cmd.extend(["--sel_policy", "echokv_token"])
        else:
            cmd.extend(["--sel_policy", "topk"])

        run_name = (
            f"{idx:03d}_model={model}_task={task}_impl={impl}_bsz={bsz}_run={run_idx}"
        )
        log_path = logs_dir / f"{run_name}.log"

        print(
            f"\n[Suite] ({idx}/{len(plan)}) model={model} task={task} "
            f"impl={impl} bsz={bsz} run={run_idx}/{item['reps']}"
        )
        code = run_one(cmd, cwd=project_root, log_path=log_path, dry_run=args.dry_run)
        if code != 0:
            failed.append(
                {
                    "idx": idx,
                    "model": model,
                    "task": task,
                    "impl": impl,
                    "bsz": bsz,
                    "run_idx": run_idx,
                    "log": str(log_path),
                    "exit_code": code,
                }
            )

    summary = {
        "total_runs": len(plan),
        "failed_runs": len(failed),
        "failed": failed,
    }
    with (run_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Suite] completed")
    print(f"[Suite] failed: {len(failed)} / {len(plan)}")
    print(f"[Suite] summary: {run_root / 'summary.json'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()


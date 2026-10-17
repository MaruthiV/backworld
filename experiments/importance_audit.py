import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter


def sampler_counters(sampler):
    counters = {
        "policy_evaluations": sampler.policy_evaluations,
        "executed_transitions": sampler.transitions,
    }
    if hasattr(sampler, "proposals"):
        counters.update(
            proposal_cache_entries=len(sampler.proposals),
            feasibility_evaluations=len(sampler.bounds),
            sampled_steps=sampler.sampled_steps,
        )
    else:
        counters.update(
            policy_cache_entries=len(sampler.policy_cache),
            suffix_cache_entries=len(sampler.suffix_cache),
        )
    return counters


def timed_estimates(world, weights, horizon, method, budgets, seed):
    import numpy as np

    from backworld.estimators import ExactSuffix
    from backworld.importance import FeasibleImportance

    started = perf_counter()
    sampler = (
        FeasibleImportance(world, weights, horizon) if method == "feasible_importance"
        else ExactSuffix(world, weights, horizon, 0 if method == "reinforce" else 2)
    )
    rng = np.random.default_rng(seed)
    initialization_seconds = perf_counter() - started
    gradient_sum = np.zeros_like(weights)
    gradient_squared_sum = 0.0
    value_sum = 0.0
    nonzero = 0
    samples = 0
    last_completed_seconds = 0.0
    counters = sampler_counters(sampler)
    checkpoints = []
    boundary = 0
    while boundary < len(budgets):
        value, gradient = sampler.estimate(rng)
        next_gradient_sum = gradient_sum + gradient
        next_gradient_squared_sum = gradient_squared_sum + float(np.sum(gradient ** 2))
        next_value_sum = value_sum + value
        next_nonzero = nonzero + bool(np.any(gradient != 0))
        next_counters = sampler_counters(sampler)
        completed_seconds = perf_counter() - started
        while boundary < len(budgets) and completed_seconds > budgets[boundary]:
            checkpoints.append({
                "budget_seconds": budgets[boundary],
                "samples": samples,
                "nonzero_updates": nonzero,
                "last_included_completion_seconds": last_completed_seconds,
                "crossing_call_completion_seconds": completed_seconds,
                "crossing_call_overshoot_seconds": completed_seconds - budgets[boundary],
                "counters": counters,
                "mean_value": value_sum / samples if samples else None,
                "mean_gradient": (gradient_sum / samples).tolist() if samples else None,
                "mean_gradient_squared_norm": gradient_squared_sum / samples if samples else None,
            })
            boundary += 1
        gradient_sum = next_gradient_sum
        gradient_squared_sum = next_gradient_squared_sum
        value_sum = next_value_sum
        nonzero = next_nonzero
        samples += 1
        last_completed_seconds = completed_seconds
        counters = next_counters
    return {
        "method": method, "initialization_seconds": initialization_seconds,
        "actual_seconds_including_final_crossing_call": perf_counter() - started,
        "total_samples_including_final_crossing_call": samples,
        "total_counters_including_final_crossing_call": counters,
        "checkpoints": checkpoints,
    }


def run_case(args):
    import numpy as np
    import torch

    from backworld import OrderWorld, SoftmaxPolicy, compile_world
    from backworld.graph import GraphLimitError
    from backworld.policy import initial_weights

    torch.set_num_threads(1)
    warmup_started = perf_counter()
    warmup = compile_world(OrderWorld(target_orders=(True, False)), 3, prune_impossible=True)
    warmup.expected_return(SoftmaxPolicy(initial_weights())).backward()
    warmup_seconds = perf_counter() - warmup_started
    count = args.orders[0]
    shared = args.variants[0] == "shared"
    seed = args.seeds[0]
    world = OrderWorld(
        shared_inventory=shared, replenishments=count - 1 if shared else 0,
        target_orders=(True,) * count,
    )
    horizon = 4 * count
    weights = initial_weights(seed, world)
    result = {
        "orders": count, "variant": args.variants[0], "policy_seed": seed,
        "world": asdict(world), "horizon": horizon, "warmup_seconds": warmup_seconds,
    }
    started = perf_counter()
    try:
        graph = compile_world(
            world, horizon, args.max_nodes, args.construction_seconds, prune_impossible=True,
        )
    except GraphLimitError as error:
        return {
            **result, "status": "reference_resource_limit", "limit": error.limit,
            "partial_graph_nodes": error.nodes, "construction_seconds": error.elapsed_seconds,
            "runs": [],
        }
    construction_seconds = perf_counter() - started
    started = perf_counter()
    policy = SoftmaxPolicy(weights)
    policy_initialization_seconds = perf_counter() - started
    durations = []
    for _ in range(4):
        started = perf_counter()
        policy.zero_grad(set_to_none=True)
        value = graph.expected_return(policy)
        value.backward()
        durations.append(perf_counter() - started)
    probability = float(value.detach())
    reference = policy.weights.grad.detach().numpy().copy()
    reference_squared_norm = float(np.sum(reference ** 2))
    if not np.isfinite(probability) or probability <= 0 or reference_squared_norm <= 0 or not np.isfinite(reference).all():
        raise RuntimeError("reference must have finite nonzero success probability and gradient")
    result["exact"] = {
        "status": "complete", "success_probability": probability,
        "gradient": reference.tolist(), "gradient_norm": float(np.sqrt(reference_squared_norm)),
        "construction_seconds": construction_seconds,
        "policy_initialization_seconds": policy_initialization_seconds,
        "first_gradient_seconds": durations[0], "repeat_gradient_seconds": durations[1:],
        "median_repeat_gradient_seconds": statistics.median(durations[1:]),
        "graph_nodes": graph.node_count, "graph_edges": graph.edge_count,
        "pruned_edges": graph.pruned_edges, "feasibility_evaluations": graph.feasibility_evaluations,
        "budget_availability_from_measured_costs": [{
            "budget_seconds": budget,
            "cold_result_available": construction_seconds + policy_initialization_seconds + durations[0] <= budget,
            "reused_graph_result_available": statistics.median(durations[1:]) <= budget,
        } for budget in args.budgets],
    }
    del graph, policy, warmup, value
    methods = ["reinforce", "exact_suffix_2", "feasible_importance"]
    runs = []
    for repetition in range(args.repetitions):
        offset = repetition % len(methods)
        for method in methods[offset:] + methods[:offset]:
            stream = np.random.SeedSequence([seed, count, int(shared), repetition, methods.index(method), 2026])
            try:
                run = timed_estimates(world, weights, horizon, method, args.budgets, stream)
            except FloatingPointError as error:
                runs.append({
                    "method": method, "repetition": repetition,
                    "status": "numerical_failure", "error": str(error),
                })
                continue
            run.update(repetition=repetition, status="complete", stream_entropy=stream.entropy)
            for checkpoint in run["checkpoints"]:
                if not checkpoint["samples"]:
                    checkpoint.update(relative_squared_gradient_error=None, relative_single_estimate_mse=None)
                    continue
                mean = np.asarray(checkpoint["mean_gradient"])
                checkpoint["relative_squared_gradient_error"] = float(
                    np.sum((mean - reference) ** 2) / reference_squared_norm,
                )
                mse = (
                    checkpoint["mean_gradient_squared_norm"] - 2 * float(np.sum(mean * reference))
                    + reference_squared_norm
                )
                checkpoint["relative_single_estimate_mse"] = max(0.0, mse / reference_squared_norm)
                checkpoint["relative_value_error"] = abs(checkpoint["mean_value"] - probability) / probability
            runs.append(run)
    status = "numerical_failure" if any(run["status"] == "numerical_failure" for run in runs) else "complete"
    result.update(status=status, runs=runs)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_worker_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--variants", nargs="+", choices=["shared", "separate"], default=["shared", "separate"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.25])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--construction-seconds", type=float, default=15.0)
    parser.add_argument("--case-seconds", type=float, default=45.0)
    parser.add_argument("--total-seconds", type=float, default=240.0)
    parser.add_argument("--output", type=Path, default=Path("results/importance-audit.json"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    limits = [*args.budgets, args.construction_seconds, args.case_seconds, args.total_seconds]
    if (
        min(args.orders) < 1 or min(args.seeds) < 0 or args.repetitions < 1 or args.max_nodes < 1
        or not all(isfinite(limit) and limit > 0 for limit in limits)
    ):
        parser.error("counts, budgets, and time limits must be positive; seeds must be nonnegative")
    args.budgets = sorted(set(args.budgets))
    if args.worker:
        if any(len(values) != 1 for values in (args.orders, args.variants, args.seeds)):
            parser.error("a worker accepts exactly one case")
        print(json.dumps(run_case(args), allow_nan=False))
        return
    source_files = sorted(Path("src/backworld").glob("*.py")) + [Path(__file__), Path("pyproject.toml")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "bounded equal-wall-time frozen-policy baseline pilot, not a learning comparison",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": version("numpy"), "torch": version("torch"), "torch_threads": 1,
        },
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    for variant in args.variants:
        for count in sorted(set(args.orders)):
            for seed in args.seeds:
                identity = {"orders": count, "variant": variant, "policy_seed": seed}
                remaining = args.total_seconds - (perf_counter() - started)
                if remaining <= 0:
                    report["cases"].append({**identity, "status": "skipped", "reason": "supervisor_time_limit"})
                else:
                    command = [
                        sys.executable, "-m", "experiments.importance_audit", "--worker",
                        "--orders", str(count), "--variants", variant, "--seeds", str(seed),
                        "--budgets", *map(str, args.budgets), "--repetitions", str(args.repetitions),
                        "--max-nodes", str(args.max_nodes),
                        "--construction-seconds", str(args.construction_seconds),
                    ]
                    print(f"Profiling {count} orders ({variant}), seed {seed}", flush=True)
                    case_started = perf_counter()
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=min(args.case_seconds, remaining))
                        if completed.returncode:
                            case = {**identity, "status": "error", "stderr": completed.stderr[-4000:]}
                        else:
                            case = json.loads(completed.stdout)
                    except subprocess.TimeoutExpired:
                        case = {**identity, "status": "case_time_limit"}
                    case["worker_wall_seconds_including_startup"] = perf_counter() - case_started
                    report["cases"].append(case)
                    print(f"  {case['status']}", flush=True)
                report["supervisor_seconds"] = perf_counter() - started
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved frozen-policy comparison to {args.output}", flush=True)
    if any(case["status"] in ("error", "numerical_failure") for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

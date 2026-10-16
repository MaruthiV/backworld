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

def run_case(args):
    import numpy as np
    import torch

    from backworld import OrderWorld, SoftmaxPolicy, compile_world
    from backworld.estimators import ExactSuffix
    from backworld.graph import GraphLimitError
    from backworld.planning import shortest_plan
    from backworld.policy import initial_weights

    torch.set_num_threads(1)
    warmup_started = perf_counter()
    warmup = compile_world(
        OrderWorld(target_orders=(True, False)), 3,
        prune_impossible=args.prune_impossible,
    )
    warmup.expected_return(SoftmaxPolicy(initial_weights())).backward()
    warmup_seconds = perf_counter() - warmup_started

    count = args.orders[0]
    shared = args.variants[0] == "shared"
    world = OrderWorld(
        shared_inventory=shared, replenishments=count - 1 if shared else 0,
        target_orders=(True,) * count,
    )
    horizon = 4 * count
    weights = initial_weights(args.seed, world)
    result = {
        "orders": count,
        "variant": args.variants[0],
        "world": asdict(world),
        "horizon": horizon,
        "action_count": world.action_count,
        "parameter_count": weights.size,
        "prune_impossible": args.prune_impossible,
        "warmup_seconds": warmup_seconds,
    }
    exact_gradient = None
    try:
        started = perf_counter()
        graph = compile_world(
            world, horizon, args.max_nodes, args.construction_seconds,
            prune_impossible=args.prune_impossible,
        )
        construction_seconds = perf_counter() - started
        policy = SoftmaxPolicy(weights)
        durations = []
        values = []
        for _ in range(args.repeats + 1):
            policy.zero_grad(set_to_none=True)
            started = perf_counter()
            value = graph.expected_return(policy)
            value.backward()
            durations.append(perf_counter() - started)
            values.append(float(value.detach()))
        exact_gradient = policy.weights.grad.detach().numpy().copy()
        if not np.isfinite(exact_gradient).all() or not np.isfinite(values).all():
            raise RuntimeError("nonfinite exact calculation")
        if max(values) - min(values) > 1e-12:
            raise RuntimeError("repeated calculation changed the frozen policy value")
        tensors = (*graph.transitions, *graph.features, graph.rewards)
        result["exact"] = {
            "status": "complete", "graph_nodes": graph.node_count,
            "graph_edges": graph.edge_count,
            "pruned_edges": graph.pruned_edges,
            "feasibility_evaluations": graph.feasibility_evaluations,
            "layer_sizes": [len(layer) for layer in graph.layers],
            "graph_tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "construction_seconds": construction_seconds,
            "first_forward_backward_seconds": durations[0],
            "repeat_forward_backward_seconds": durations[1:],
            "median_repeat_forward_backward_seconds": statistics.median(durations[1:]),
            "success_probability": values[-1],
            "gradient_norm": float(np.linalg.norm(exact_gradient)),
        }
        del graph, policy, tensors, value
    except GraphLimitError as error:
        result["exact"] = {
            "status": "resource_limit", "limit": error.limit,
            "partial_graph_nodes": error.nodes, "construction_seconds": error.elapsed_seconds,
            "success_probability": None,
        }

    estimates = []
    for depth in (0, 2):
        started = perf_counter()
        estimator = ExactSuffix(world, weights, horizon, depth)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, count, int(shared), depth]))
        nonzero = 0
        squared_error = 0.0
        mean = np.zeros_like(weights)
        for sample in range(args.samples):
            _, derivative = estimator.estimate(rng)
            if sample == 0:
                first_seconds = perf_counter() - started
            mean += derivative / args.samples
            nonzero += bool(np.any(derivative != 0))
            if exact_gradient is not None:
                squared_error += float(np.sum((derivative - exact_gradient) ** 2)) / args.samples
        seconds = perf_counter() - started
        estimates.append({
            "exact_steps": depth, "samples": args.samples,
            "total_seconds": seconds, "first_estimate_seconds": first_seconds,
            "nonzero_updates": nonzero,
            "policy_evaluations": estimator.policy_evaluations,
            "executed_transitions": estimator.transitions,
            "suffix_cache_entries": len(estimator.suffix_cache),
            "mean_gradient_norm": float(np.linalg.norm(mean)),
            "single_estimate_gradient_mse": squared_error if exact_gradient is not None else None,
        })
        del estimator
    result["fixed_suffix"] = estimates

    started = perf_counter()
    plan = shortest_plan(world, horizon, args.max_nodes, args.planner_seconds)
    if plan.actions is not None:
        state = world.initial_state()
        for action in plan.actions + (world.action_count - 1,) * (horizon - len(plan.actions)):
            state = world.step(state, action)
        if not world.reward(state):
            raise RuntimeError("planner result does not replay to a terminal success")
    result["planner"] = {
        **asdict(plan), "plan_length": len(plan.actions) if plan.actions is not None else None,
        "seconds_including_replay": perf_counter() - started,
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_worker_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--variants", nargs="+", choices=["shared", "separate"], default=["shared", "separate"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--construction-seconds", type=float, default=15.0)
    parser.add_argument("--planner-seconds", type=float, default=10.0)
    parser.add_argument("--case-seconds", type=float, default=45.0)
    parser.add_argument("--total-seconds", type=float, default=240.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--prune-impossible", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/scaling-audit.json"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    limits = [args.construction_seconds, args.planner_seconds, args.case_seconds, args.total_seconds]
    if (
        min(args.orders) < 1 or args.max_nodes < 1 or args.repeats < 1 or args.samples < 1
        or not all(isfinite(limit) and limit > 0 for limit in limits)
    ):
        parser.error("counts and finite time limits must be positive")
    if args.worker:
        if len(args.orders) != 1 or len(args.variants) != 1:
            parser.error("a worker accepts exactly one case")
        print(json.dumps(run_case(args), allow_nan=False))
        return

    source_files = sorted(Path("src/backworld").glob("*.py")) + [Path(__file__), Path("pyproject.toml")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "bounded structural cost pilot, not an equal-time learning comparison",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": version("numpy"), "torch": version("torch"), "torch_threads": 1,
        },
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    for variant in args.variants:
        stop_reason = None
        for count in sorted(set(args.orders)):
            remaining = args.total_seconds - (perf_counter() - started)
            if stop_reason or remaining <= 0:
                report["cases"].append({
                    "orders": count, "variant": variant, "status": "skipped",
                    "reason": stop_reason or "supervisor_time_limit",
                })
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                continue
            command = [
                sys.executable, "-m", "experiments.scaling_audit", "--worker",
                "--orders", str(count), "--variants", variant, "--seed", str(args.seed),
                "--max-nodes", str(args.max_nodes),
                "--construction-seconds", str(args.construction_seconds),
                "--planner-seconds", str(args.planner_seconds),
                "--repeats", str(args.repeats), "--samples", str(args.samples),
            ]
            if args.prune_impossible:
                command.append("--prune-impossible")
            print(f"Profiling {count} orders ({variant})", flush=True)
            case_started = perf_counter()
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=min(args.case_seconds, remaining))
                if completed.returncode:
                    case = {"orders": count, "variant": variant, "status": "error", "stderr": completed.stderr[-4000:]}
                    stop_reason = "previous_worker_error"
                else:
                    case = json.loads(completed.stdout)
                    case["status"] = "complete"
                    if case["exact"]["status"] == "resource_limit":
                        stop_reason = "previous_resource_limit"
            except subprocess.TimeoutExpired:
                case = {"orders": count, "variant": variant, "status": "case_time_limit"}
                stop_reason = "previous_case_time_limit"
            case["worker_wall_seconds_including_startup"] = perf_counter() - case_started
            report["cases"].append(case)
            report["supervisor_seconds"] = perf_counter() - started
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"  {case['status']}", flush=True)
    print(f"Saved bounded scaling results to {args.output}", flush=True)
    if any(case["status"] == "error" for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

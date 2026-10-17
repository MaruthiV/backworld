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
import subprocess
import sys
from time import perf_counter


def run_case(args):
    import numpy as np
    import torch

    from backworld import OrderWorld, SoftmaxPolicy, compile_world
    from backworld.frontier import (
        FrontierLimitError, build_frontier_truth, frontier_moments, search_exact_sets,
    )
    from backworld.graph import GraphLimitError
    from backworld.oracle import exact_forward
    from backworld.policy import initial_weights

    torch.set_num_threads(1)
    count, variant, seed = args.orders[0], args.variants[0], args.seeds[0]
    world = OrderWorld(
        shared_inventory=variant == "shared",
        replenishments=count - 1 if variant == "shared" else 0,
        target_orders=(True,) * count,
    )
    horizon = 4 * count
    weights = initial_weights(seed, world)
    result = {
        "orders": count, "variant": variant, "policy_seed": seed,
        "world": asdict(world), "horizon": horizon, "depth": args.depth,
    }
    started = perf_counter()
    try:
        graph = compile_world(
            world, horizon, args.max_nodes, args.construction_seconds, prune_impossible=True,
        )
    except GraphLimitError as error:
        return {
            **result, "status": "reference_resource_limit", "limit": error.limit,
            "partial_graph_nodes": error.nodes, "seconds": error.elapsed_seconds,
        }
    construction_seconds = perf_counter() - started
    started = perf_counter()
    policy = SoftmaxPolicy(weights)
    value = graph.expected_return(policy)
    value.backward()
    torch_seconds = perf_counter() - started
    torch_value = float(value.detach())
    torch_gradient = policy.weights.grad.detach().numpy().ravel().copy()
    started = perf_counter()
    reference = exact_forward(world, weights, horizon)
    forward_seconds = perf_counter() - started
    reference_gradient = reference.gradient.ravel()
    norm_squared = float(np.dot(reference_gradient, reference_gradient))
    if reference.value <= 0 or norm_squared <= 0:
        raise RuntimeError("screen requires a positive value and nonzero reference gradient")
    started = perf_counter()
    truth = build_frontier_truth(world, weights, graph, args.depth)
    setup_seconds = perf_counter() - started
    prefix_value = float(np.sum(truth.masses * truth.values))
    prefix_gradient = truth.contributions.sum(axis=0)
    checks = {
        "torch_value_relative_error": abs(torch_value - reference.value) / reference.value,
        "prefix_value_relative_error": abs(prefix_value - reference.value) / reference.value,
        "root_value_relative_error": abs(truth.root_value - reference.value) / reference.value,
        "torch_gradient_relative_squared_error": float(np.sum((torch_gradient - reference_gradient) ** 2)) / norm_squared,
        "prefix_gradient_relative_squared_error": float(np.sum((prefix_gradient - reference_gradient) ** 2)) / norm_squared,
        "root_gradient_relative_squared_error": float(np.sum((truth.root_gradient - reference_gradient) ** 2)) / norm_squared,
    }
    if not np.isfinite(list(checks.values())).all() or any(
        error > (1e-10 if "value" in key else 1e-20) for key, error in checks.items()
    ):
        raise RuntimeError(f"independent reference mismatch: {checks}")
    result["reference"] = {
        "value": reference.value, "gradient": reference.gradient.tolist(),
        "gradient_norm_squared": norm_squared, "graph_nodes": graph.node_count,
        "layer_nodes": [len(layer) for layer in graph.layers],
        "construction_seconds": construction_seconds,
        "first_torch_gradient_seconds": torch_seconds,
        "independent_forward_seconds": forward_seconds,
        "independent_forward_states": reference.state_evaluations,
        "frontier_truth_seconds": setup_seconds,
        "checks": checks,
    }
    result["frontier"] = {
        "states": [asdict(state) for state in truth.states],
        "masses": truth.masses.tolist(), "total_live_mass": float(truth.masses.sum()),
        "values": truth.values.tolist(),
        "contribution_norms": np.linalg.norm(truth.contributions, axis=1).tolist(),
        "closure_nodes": [bits.bit_count() for bits in truth.closures],
        "prefix_nodes_including_cut": truth.exact_nodes(()),
    }
    budgets = tuple(int(fraction * graph.node_count) for fraction in (0.25, 0.5))
    result["proposals"] = {}
    for proposal in ("feasible", "original"):
        stage = "moments"
        started = perf_counter()
        moment_seconds = None
        try:
            moments = frontier_moments(truth, proposal, args.moment_seconds)
            moment_seconds = perf_counter() - started
            stage = "subset_search"
            search = search_exact_sets(
                truth, moments, budgets, args.max_frontier, args.search_seconds,
            )
            result["proposals"][proposal] = {
                **search, "moment_seconds": moment_seconds,
                "guided_policy_evaluations": moments.policy_evaluations,
                "frontier_full_score_second_moments": moments.full_score_second_moments.tolist(),
            }
        except FrontierLimitError as error:
            result["proposals"][proposal] = {
                "status": "resource_limit", "stage": stage, "limit": error.limit,
                "limit_stage_seconds": error.elapsed_seconds,
                "moment_seconds": moment_seconds,
                "total_seconds": perf_counter() - started,
            }
    result["status"] = (
        "complete" if all(run["status"] == "complete" for run in result["proposals"].values())
        else "partial_resource_limit"
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_worker_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--variants", nargs="+", choices=["shared", "separate"], default=["shared", "separate"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--max-frontier", type=int, default=18)
    parser.add_argument("--construction-seconds", type=float, default=15.0)
    parser.add_argument("--moment-seconds", type=float, default=10.0)
    parser.add_argument("--search-seconds", type=float, default=10.0)
    parser.add_argument("--case-seconds", type=float, default=45.0)
    parser.add_argument("--total-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("results/frontier-ceiling.json"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    limits = [args.construction_seconds, args.moment_seconds, args.search_seconds, args.case_seconds, args.total_seconds]
    if (
        min(args.orders) < 1 or min(args.seeds) < 0 or args.max_nodes < 1 or args.max_frontier < 0
        or not 0 <= args.depth <= 4 * min(args.orders)
        or not all(isfinite(limit) and limit > 0 for limit in limits)
    ):
        parser.error("case sizes, cut depth, and finite time limits must be valid")
    if args.worker:
        if any(len(values) != 1 for values in (args.orders, args.variants, args.seeds)):
            parser.error("a worker accepts exactly one case")
        print(json.dumps(run_case(args), allow_nan=False))
        return
    sources = sorted(Path("src/backworld").glob("*.py")) + [Path(__file__), Path("pyproject.toml")]
    previous = [Path("results/importance-fixed-count.json"), Path("results/frontier-tests.xml")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "evaluation-only exact-subset ceiling, not a deployable method or training result",
        "node_convention": "all pruned state/time keys including initial, terminal, and every live cut key; shared suffix union counted once",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": version("numpy"), "torch": version("torch"), "torch_threads": 1,
        },
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "previous_artifact_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in previous if path.exists()},
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
                    case = {**identity, "status": "skipped", "reason": "supervisor_time_limit"}
                else:
                    command = [
                        sys.executable, "-m", "experiments.frontier_ceiling", "--worker",
                        "--orders", str(count), "--variants", variant, "--seeds", str(seed),
                        "--depth", str(args.depth), "--max-nodes", str(args.max_nodes),
                        "--max-frontier", str(args.max_frontier),
                        "--construction-seconds", str(args.construction_seconds),
                        "--moment-seconds", str(args.moment_seconds), "--search-seconds", str(args.search_seconds),
                    ]
                    print(f"Checking frontier ceiling: {count} orders ({variant}), seed {seed}", flush=True)
                    case_started = perf_counter()
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=min(args.case_seconds, remaining))
                        case = json.loads(completed.stdout) if completed.returncode == 0 else {
                            **identity, "status": "error", "stderr": completed.stderr[-4000:],
                        }
                    except subprocess.TimeoutExpired:
                        case = {**identity, "status": "case_time_limit"}
                    case["worker_wall_seconds_including_startup"] = perf_counter() - case_started
                    print(f"  {case['status']}", flush=True)
                report["cases"].append(case)
                report["supervisor_seconds"] = perf_counter() - started
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved frontier ceiling to {args.output}", flush=True)
    if any(case["status"] == "error" for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

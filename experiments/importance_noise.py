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


def sample_fixed_counts(world, weights, horizon, counts, seed):
    import numpy as np

    from backworld.importance import FeasibleImportance
    from experiments.importance_audit import sampler_counters

    started = perf_counter()
    sampler = FeasibleImportance(world, weights, horizon)
    rng = np.random.default_rng(seed)
    initialization_seconds = perf_counter() - started
    gradient_sum = np.zeros_like(weights)
    values = []
    squared_norms = []
    log_weights = []
    nonzero = 0
    checkpoints = []
    for sample in range(1, counts[-1] + 1):
        value, gradient = sampler.estimate(rng)
        gradient_sum += gradient
        values.append(value)
        squared_norms.append(float(np.sum(gradient ** 2)))
        log_weights.append(sampler.last_log_weight)
        nonzero += bool(np.any(gradient != 0))
        if sample not in counts:
            continue
        value_array = np.asarray(values)
        norm_array = np.asarray(squared_norms)
        finite_logs = np.asarray([weight for weight in log_weights if np.isfinite(weight)])
        mean = gradient_sum / sample
        value_sum = float(value_array.sum())
        value_squared_sum = float(np.sum(value_array ** 2))
        norm_sum = float(norm_array.sum())
        top_count = max(1, int(np.ceil(sample * 0.01)))
        checkpoint = {
            "samples": sample, "nonzero_updates": nonzero,
            "mean_value": value_sum / sample, "mean_gradient": mean.tolist(),
            "mean_value_squared": value_squared_sum / sample,
            "mean_gradient_squared_norm": norm_sum / sample,
            "empirical_value_variance": max(0.0, (value_squared_sum - value_sum ** 2 / sample) / (sample - 1)),
            "empirical_gradient_variance_trace": max(0.0, (norm_sum - sample * float(np.sum(mean ** 2))) / (sample - 1)),
            "corrected_value_quantiles_0_50_90_99_100": np.quantile(value_array, [0, 0.5, 0.9, 0.99, 1]).tolist(),
            "log_weight_quantiles_0_50_90_99_100": np.quantile(finite_logs, [0, 0.5, 0.9, 0.99, 1]).tolist() if finite_logs.size else None,
            "nonfinite_log_weights": sample - finite_logs.size,
            "largest_value_share": float(value_array.max()) / value_sum if value_sum else None,
            "top_one_percent_value_share": float(np.sort(value_array)[-top_count:].sum()) / value_sum if value_sum else None,
            "largest_squared_gradient_share": float(norm_array.max()) / norm_sum if norm_sum else None,
            "top_one_percent_squared_gradient_share": float(np.sort(norm_array)[-top_count:].sum()) / norm_sum if norm_sum else None,
            "observed_value_effective_sample_size": value_sum ** 2 / value_squared_sum if value_squared_sum else None,
            "counters": sampler_counters(sampler),
        }
        checkpoint["elapsed_seconds"] = perf_counter() - started
        checkpoints.append(checkpoint)
    return {
        "initialization_seconds": initialization_seconds,
        "total_seconds": perf_counter() - started,
        "checkpoints": checkpoints,
    }


def run_case(args):
    import numpy as np
    import torch

    from backworld import OrderWorld, SoftmaxPolicy, compile_world
    from backworld.graph import GraphLimitError
    from backworld.oracle import OracleLimitError, exact_importance_moments
    from backworld.policy import initial_weights

    torch.set_num_threads(1)
    started = perf_counter()
    warmup = compile_world(OrderWorld(target_orders=(True, False)), 3, prune_impossible=True)
    warmup.expected_return(SoftmaxPolicy(initial_weights())).backward()
    warmup_seconds = perf_counter() - started
    count, variant, seed = args.orders[0], args.variants[0], args.seeds[0]
    shared = variant == "shared"
    world = OrderWorld(
        shared_inventory=shared, replenishments=count - 1 if shared else 0,
        target_orders=(True,) * count,
    )
    horizon = 4 * count
    weights = initial_weights(seed, world)
    result = {
        "orders": count, "variant": variant, "policy_seed": seed,
        "world": asdict(world), "horizon": horizon, "warmup_seconds": warmup_seconds,
    }
    started = perf_counter()
    try:
        graph = compile_world(world, horizon, args.max_nodes, args.construction_seconds, prune_impossible=True)
    except GraphLimitError as error:
        return {
            **result, "status": "reference_resource_limit", "limit": error.limit,
            "partial_graph_nodes": error.nodes, "elapsed_seconds": error.elapsed_seconds,
            "runs": [],
        }
    construction_seconds = perf_counter() - started
    started = perf_counter()
    policy = SoftmaxPolicy(weights)
    value = graph.expected_return(policy)
    value.backward()
    evaluation_seconds = perf_counter() - started
    probability = float(value.detach())
    reference = policy.weights.grad.detach().numpy().copy()
    reference_norm_squared = float(np.sum(reference ** 2))
    if not np.isfinite(probability) or probability <= 0 or reference_norm_squared <= 0 or not np.isfinite(reference).all():
        raise RuntimeError("reference must have finite nonzero probability and gradient")
    result["reference"] = {
        "success_probability": probability, "gradient": reference.tolist(),
        "gradient_norm": float(np.sqrt(reference_norm_squared)),
        "construction_seconds": construction_seconds,
        "initialization_and_gradient_seconds": evaluation_seconds,
        "graph_nodes": graph.node_count,
    }
    del graph, policy, value, warmup
    true_variance = None
    if count > args.moment_max_orders:
        result["moments"] = {"status": "skipped", "reason": "predeclared_order_scope"}
    else:
        started = perf_counter()
        try:
            moments = exact_importance_moments(
                world, weights, horizon, args.moment_max_nodes, args.moment_seconds,
            )
            mean_error = float(np.sum((moments.mean - reference) ** 2)) / reference_norm_squared
            value_error = abs(moments.value - probability) / probability
            if mean_error > 1e-20 or value_error > 1e-10:
                raise RuntimeError("importance moment mean disagrees with the exact reference")
            true_variance = moments.variance_trace
            result["moments"] = {
                "status": "complete", "seconds": perf_counter() - started,
                "value": moments.value, "mean_gradient": moments.mean.tolist(),
                "value_second_moment": moments.value_second_moment,
                "gradient_second_moment": moments.gradient_second_moment,
                "value_variance": moments.value_variance,
                "gradient_variance_trace": true_variance,
                "relative_value_variance": moments.value_variance / probability ** 2,
                "relative_gradient_variance": true_variance / reference_norm_squared,
                "mean_relative_squared_error": mean_error,
                "value_relative_error": value_error,
                "proposal_success_probability": moments.proposal_success_probability,
                "layered_states": moments.layered_states,
                "policy_evaluations": moments.policy_evaluations,
                "executed_transitions": moments.transitions,
                "feasibility_evaluations": moments.feasibility_evaluations,
            }
        except OracleLimitError as error:
            result["moments"] = {
                "status": "resource_limit", "limit": error.limit,
                "partial_layered_states": error.nodes, "seconds": error.elapsed_seconds,
            }
    runs = []
    for repetition in range(args.repetitions):
        stream = np.random.SeedSequence([seed, count, int(shared), repetition, 20261016, 5])
        try:
            run = sample_fixed_counts(world, weights, horizon, args.counts, stream)
        except FloatingPointError as error:
            runs.append({"status": "numerical_failure", "repetition": repetition, "error": str(error)})
            continue
        run.update(status="complete", repetition=repetition, stream_entropy=stream.entropy)
        for checkpoint in run["checkpoints"]:
            mean = np.asarray(checkpoint["mean_gradient"])
            checkpoint.update(
                relative_squared_gradient_error=float(np.sum((mean - reference) ** 2)) / reference_norm_squared,
                value_ratio=checkpoint["mean_value"] / probability,
                relative_single_estimate_mse=max(0.0, (
                    checkpoint["mean_gradient_squared_norm"] - 2 * float(np.sum(mean * reference))
                    + reference_norm_squared
                ) / reference_norm_squared),
                expected_relative_squared_gradient_error=(
                    true_variance / (checkpoint["samples"] * reference_norm_squared)
                    if true_variance is not None else None
                ),
                empirical_to_true_gradient_variance_ratio=(
                    checkpoint["empirical_gradient_variance_trace"] / true_variance
                    if true_variance else None
                ),
            )
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
    parser.add_argument("--counts", type=int, nargs="+", default=[512, 4096])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--construction-seconds", type=float, default=15.0)
    parser.add_argument("--moment-max-orders", type=int, default=4)
    parser.add_argument("--moment-max-nodes", type=int, default=5_000)
    parser.add_argument("--moment-seconds", type=float, default=10.0)
    parser.add_argument("--case-seconds", type=float, default=45.0)
    parser.add_argument("--total-seconds", type=float, default=240.0)
    parser.add_argument("--output", type=Path, default=Path("results/importance-fixed-count.json"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    limits = [args.construction_seconds, args.moment_seconds, args.case_seconds, args.total_seconds]
    if (
        min(args.orders) < 1 or min(args.seeds) < 0 or min(args.counts) < 2
        or min(args.repetitions, args.max_nodes, args.moment_max_orders, args.moment_max_nodes) < 1
        or not all(isfinite(limit) and limit > 0 for limit in limits)
    ):
        parser.error("counts and finite time limits must be positive; sample counts at least two and seeds nonnegative")
    args.counts = sorted(set(args.counts))
    if args.worker:
        if any(len(values) != 1 for values in (args.orders, args.variants, args.seeds)):
            parser.error("a worker accepts exactly one case")
        print(json.dumps(run_case(args), allow_nan=False))
        return
    source_files = sorted(Path("src/backworld").glob("*.py")) + [
        Path(__file__), Path("experiments/importance_audit.py"), Path("pyproject.toml"),
    ]
    previous = [Path("results/importance-audit.json"), Path("results/importance-summary.json")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "fixed-count importance-noise diagnostic, not a training comparison",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": version("numpy"), "torch": version("torch"), "torch_threads": 1,
        },
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
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
                    report["cases"].append({**identity, "status": "skipped", "reason": "supervisor_time_limit"})
                else:
                    command = [
                        sys.executable, "-m", "experiments.importance_noise", "--worker",
                        "--orders", str(count), "--variants", variant, "--seeds", str(seed),
                        "--counts", *map(str, args.counts), "--repetitions", str(args.repetitions),
                        "--max-nodes", str(args.max_nodes), "--construction-seconds", str(args.construction_seconds),
                        "--moment-max-orders", str(args.moment_max_orders), "--moment-max-nodes", str(args.moment_max_nodes),
                        "--moment-seconds", str(args.moment_seconds),
                    ]
                    print(f"Checking fixed counts: {count} orders ({variant}), seed {seed}", flush=True)
                    case_started = perf_counter()
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, timeout=min(args.case_seconds, remaining))
                        case = json.loads(completed.stdout) if completed.returncode == 0 else {
                            **identity, "status": "error", "stderr": completed.stderr[-4000:],
                        }
                    except subprocess.TimeoutExpired:
                        case = {**identity, "status": "case_time_limit"}
                    case["worker_wall_seconds_including_startup"] = perf_counter() - case_started
                    report["cases"].append(case)
                    print(f"  {case['status']}", flush=True)
                report["supervisor_seconds"] = perf_counter() - started
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved fixed-count diagnostic to {args.output}", flush=True)
    if any(case["status"] in ("error", "numerical_failure") for case in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import tracemalloc

import numpy as np
import torch

from backworld import OrderWorld, SoftmaxPolicy, compile_world
from backworld.estimators import ExactSuffix
from backworld.oracle import exact_forward, exact_suffix_moments
from backworld.policy import initial_weights


def run_case(world, horizon, seed, samples):
    weights = initial_weights(seed)
    started = perf_counter()
    graph = compile_world(world, horizon)
    compilation_seconds = perf_counter() - started

    started = perf_counter()
    reference = exact_forward(world, weights, horizon)
    oracle_seconds = perf_counter() - started

    policy = SoftmaxPolicy(weights)
    started = perf_counter()
    value = graph.expected_return(policy)
    value.backward()
    backward_seconds = perf_counter() - started
    gradient = policy.weights.grad.detach().numpy()
    value_error = abs(float(value.detach()) - reference.value)
    gradient_error = float(np.max(np.abs(gradient - reference.gradient)))
    if value_error > 1e-11 or gradient_error > 1e-11:
        raise RuntimeError("compiled calculation disagrees with the independent oracle")

    estimates = []
    for depth in sorted({0, 1, 2, 3, horizon} & set(range(horizon + 1))):
        moment_started = perf_counter()
        moments = exact_suffix_moments(world, weights, horizon, depth)
        moment_seconds = perf_counter() - moment_started
        if np.max(np.abs(moments.mean - reference.gradient)) > 1e-11:
            raise RuntimeError("exact estimator mean disagrees with the independent oracle")
        rng = np.random.default_rng(np.random.SeedSequence([seed, depth, 2026]))
        estimator = ExactSuffix(world, weights, horizon, depth)
        gradients = []
        values = []
        tracemalloc.start()
        started = perf_counter()
        for sample in range(samples):
            estimate, derivative = estimator.estimate(rng)
            if sample == 0:
                first_estimate_seconds = perf_counter() - started
            values.append(estimate)
            gradients.append(derivative.reshape(-1))
        seconds = perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gradients = np.stack(gradients)
        errors = gradients - reference.gradient.reshape(-1)
        estimates.append({
            "exact_steps": depth,
            "samples": samples,
            "mean_value_estimate": float(np.mean(values)),
            "single_estimate_gradient_mse": float(np.mean(np.sum(errors ** 2, axis=1))),
            "mean_gradient_error_norm": float(np.linalg.norm(errors.mean(axis=0))),
            "gradient_sample_variance_trace": float(gradients.var(axis=0, ddof=1).sum()),
            "true_gradient_variance_trace": moments.variance_trace,
            "moment_oracle_seconds_excluded_from_estimator_timing": moment_seconds,
            "nonzero_value_estimates": int(np.count_nonzero(values)),
            "total_seconds_with_allocation_tracing": seconds,
            "first_estimate_seconds_with_allocation_tracing": first_estimate_seconds,
            "peak_python_allocation_bytes": peak_bytes,
            "policy_evaluations": estimator.policy_evaluations,
            "executed_transitions": estimator.transitions,
            "suffix_cache_entries": len(estimator.suffix_cache),
        })

    return {
        "world": asdict(world),
        "horizon": horizon,
        "policy_seed": seed,
        "baseline": 0.0,
        "layer_sizes": [len(layer) for layer in graph.layers],
        "graph_nodes": graph.node_count,
        "graph_edges": graph.edge_count,
        "compilation_seconds": compilation_seconds,
        "oracle_seconds": oracle_seconds,
        "torch_forward_backward_seconds": backward_seconds,
        "exact_success_probability": reference.value,
        "exact_gradient_norm": float(np.linalg.norm(reference.gradient)),
        "probability_of_no_sampled_successes": float((1 - reference.value) ** samples),
        "compiled_value_absolute_error": value_error,
        "compiled_gradient_max_absolute_error": gradient_error,
        "suffix_estimates": estimates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/gradient-audit.json"))
    args = parser.parse_args()
    if args.samples < 2 or args.horizon < 1:
        parser.error("samples must be at least two and horizon must be positive")
    torch.set_num_threads(1)
    cases = []
    for seed in args.seeds:
        for shared in (True, False):
            world = OrderWorld(shared_inventory=shared, replenishments=int(shared))
            cases.append(run_case(world, args.horizon, seed, args.samples))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "correctness and equal-sample cost diagnostic, not a training comparison",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("src/backworld").glob("*.py")) + [Path(__file__), Path("pyproject.toml")]
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(cases)} checked cases to {args.output}")
    for case in cases:
        print(
            f"seed={case['policy_seed']} shared={case['world']['shared_inventory']} "
            f"nodes={case['graph_nodes']} success={case['exact_success_probability']:.8f} "
            f"gradient_error={case['compiled_gradient_max_absolute_error']:.2e}"
        )


if __name__ == "__main__":
    main()

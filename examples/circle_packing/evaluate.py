#!/usr/bin/env python3
"""
Standalone evaluator for circle packing example (n=26).
No external dependencies on evolve_anything package.
"""

import os
import json
import argparse
import importlib.util
import numpy as np
from typing import Tuple, Optional, Dict, Any


# =============================================================================
# Generic utilities - can be copied to any evaluation script
# =============================================================================


def load_python_module(program_path: str):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_results(
    results_dir: str,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None,
):
    """Save metrics.json and correct.json to the results directory."""
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error}, f, indent=2)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# =============================================================================
# Circle packing specific evaluation logic
# =============================================================================


def validate_packing(
    centers: np.ndarray, radii: np.ndarray, reported_sum: float, atol=1e-6
) -> Tuple[bool, Optional[str]]:
    """Validate circle packing result."""
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26

    if centers.shape != (n_expected, 2):
        return (
            False,
            f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}",
        )
    if radii.shape != (n_expected,):
        return (
            False,
            f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}",
        )

    if np.any(radii < 0):
        negative_indices = np.where(radii < 0)[0]
        return False, f"Negative radii found at indices: {negative_indices}"

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        return (
            False,
            f"Sum of radii ({np.sum(radii):.6f}) doesn't match reported ({reported_sum:.6f})",
        )

    # Check circles are inside unit square
    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        if x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol:
            return (
                False,
                f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square",
            )

    # Check circles don't overlap
    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                return (
                    False,
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, Sum radii: {radii[i] + radii[j]:.4f}",
                )

    return True, None


def generate_visualization(
    centers: np.ndarray,
    radii: np.ndarray,
    results_dir: str,
    is_valid: bool,
):
    """Generate a visualization of the circle packing result."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1A1A2E")
        ax.set_facecolor("#1A1A2E")

        # Draw unit square boundary
        ax.plot(
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            color="#4ECDC4",
            linewidth=2,
            linestyle="--",
        )

        # Create circle patches with gradient colors based on radius
        circles = []
        colors = []
        for i in range(len(radii)):
            circle = Circle(centers[i], radii[i])
            circles.append(circle)
            colors.append(radii[i])

        # Create collection with viridis colormap
        pc = PatchCollection(
            circles, cmap="viridis", alpha=0.8, edgecolor="white", linewidth=0.5
        )
        pc.set_array(np.array(colors))
        ax.add_collection(pc)

        # Add colorbar
        cbar = plt.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Radius", color="#EAEAEA", fontsize=10)
        cbar.ax.yaxis.set_tick_params(color="#EAEAEA")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#EAEAEA")

        # Set aspect and limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

        # Title with score
        total_radius = np.sum(radii)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        ax.set_title(
            f"Circle Packing (n={len(radii)})\nSum of Radii: {total_radius:.4f} {status}",
            color="#EAEAEA",
            fontsize=14,
            fontweight="bold",
        )

        # Style axes
        ax.tick_params(colors="#8E8E93")
        for spine in ax.spines.values():
            spine.set_color("#4ECDC4")
            spine.set_alpha(0.3)

        ax.set_xlabel("x", color="#EAEAEA")
        ax.set_ylabel("y", color="#EAEAEA")

        # Save the figure
        viz_path = os.path.join(results_dir, "visualization.png")
        plt.savefig(
            viz_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="#1A1A2E",
            edgecolor="none",
        )
        plt.close(fig)

        print(f"Visualization saved to: {viz_path}")
        return viz_path

    except ImportError as e:
        print(
            f"Warning: Could not generate visualization (matplotlib not available): {e}"
        )
        return None
    except Exception as e:
        print(f"Warning: Error generating visualization: {e}")
        return None


def evaluate(program_path: str, results_dir: str):
    """Run the circle packing evaluation."""
    print(f"Evaluating: {program_path}")
    print(f"Results dir: {results_dir}")

    try:
        # Load the program and get the function
        module = load_python_module(program_path)
        if not hasattr(module, "run_packing"):
            raise AttributeError("Program must define 'run_packing' function")

        # Run the experiment
        centers, radii, reported_sum = module.run_packing()

        # Validate
        is_valid, error_msg = validate_packing(centers, radii, reported_sum)

        # Compute metrics
        metrics = {
            "combined_score": float(reported_sum) if is_valid else 0.0,
            "public": {
                "num_circles": int(centers.shape[0]),
                "sum_of_radii": float(np.sum(radii)),
            },
            "private": {
                "reported_sum": float(reported_sum),
            },
        }

        # Save extra data
        np.savez(
            os.path.join(results_dir, "extra.npz"),
            centers=centers,
            radii=radii,
            reported_sum=reported_sum,
        )

        # Generate visualization
        generate_visualization(centers, radii, results_dir, is_valid)

        save_results(results_dir, metrics, is_valid, error_msg)

        if is_valid:
            print(f"SUCCESS: Sum of radii = {reported_sum:.6f}")
        else:
            print(f"FAILED: {error_msg}")

    except Exception as e:
        print(f"ERROR: {e}")
        metrics = {"combined_score": 0.0, "error": str(e)}
        save_results(results_dir, metrics, correct=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circle packing evaluator")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    evaluate(args.program_path, args.results_dir)

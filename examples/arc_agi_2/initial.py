import numpy as np

# EVOLVE-BLOCK-START
def transform(input_grid: np.ndarray) -> np.ndarray:
    """Transform the input grid according to the discovered pattern.

    Args:
        input_grid: 2D numpy array with integer values 0-9.

    Returns:
        Transformed 2D numpy array with integer values 0-9.
    """
    return input_grid.copy()
# EVOLVE-BLOCK-END


def run_transform(input_grid):
    """Entry point called by the evaluator. Do not modify."""
    return transform(np.array(input_grid, dtype=int)).tolist()

import random
from typing import List, Optional

from evolve_anything.database import Program
from .prompts_base import perf_str


CROSS_SYS_FORMAT = """
You are given multiple code scripts implementing the same algorithm.
You are tasked with generating a new code snippet that combines these code scripts in a way that is more efficient.
I.e. perform crossover between the code scripts.
Provide the complete new program code.
You MUST repond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the code you are proposing.
</DESCRIPTION>

<CODE>
```{language}
# The new rewritten program here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code. Do not change the code outside of these markers.
* Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
* Make sure the file still runs after your changes.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards."""


CROSS_ITER_MSG = """# Current program

Here is the current program we are trying to improve (you will need to propose a new program with the same inputs and outputs as the original program, but with improved internal implementation):

```{language}
{code_content}
```

Here are the performance metrics of the program:

{performance_metrics}{text_feedback_section}

# Task

Perform a cross-over between the code script above and the one below. Aim to combine the best parts of both code implementations that improves the score.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
"""


def get_cross_component(
    archive_inspirations: List[Program],
    top_k_inspirations: List[Program],
    language: str = "python",
    parent: Optional[Program] = None,
) -> str:
    all_inspirations = archive_inspirations + top_k_inspirations

    if not all_inspirations:
        return ""

    selected_inspiration = None

    # Compute embedding distance for diversity if parent is provided
    if parent and parent.embedding and any(p.embedding for p in all_inspirations):
        try:
            import numpy as np

            parent_emb = np.array(parent.embedding)
            max_dist = -1.0

            # Filter inspirations that have embeddings
            valid_inspirations = [p for p in all_inspirations if p.embedding]

            if valid_inspirations:
                for insp in valid_inspirations:
                    insp_emb = np.array(insp.embedding)

                    # Cosine distance = 1 - cosine_similarity
                    # We want to maximize distance (dissimilarity)
                    norm_p = np.linalg.norm(parent_emb)
                    norm_i = np.linalg.norm(insp_emb)

                    if norm_p > 0 and norm_i > 0:
                        cos_sim = np.dot(parent_emb, insp_emb) / (norm_p * norm_i)
                        dist = 1.0 - cos_sim
                    else:
                        dist = 1.0  # Default to max distance if norm is 0

                    if dist > max_dist:
                        max_dist = dist
                        selected_inspiration = insp
        except ImportError:
            pass
        except Exception:
            # Fallback to random if numpy fails or other error
            pass

    # Fallback to random choice if selection failed or no parent provided
    if selected_inspiration is None:
        selected_inspiration = random.choice(all_inspirations)

    crossover_inspiration = "# Crossover Inspiration Programs\n"
    crossover_inspiration += f"```{language}\n{selected_inspiration.code}\n```\n\n"
    crossover_inspiration += f"Performance metrics: {perf_str(selected_inspiration.combined_score, selected_inspiration.public_metrics)}\n\n"

    return crossover_inspiration

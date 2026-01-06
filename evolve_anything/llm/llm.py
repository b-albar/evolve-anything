"""
LLM Client module.

Provides a unified LLM client class for querying language models.
"""

import logging
import asyncio
from typing import Dict, List, Union, Optional, Type
import re
import json
from pydantic import BaseModel

from .query import sample_model_kwargs, query, query_async
from .models import QueryResult
from .dynamic_sampling import BanditBase, FixedSampler

MAX_RETRIES = 3

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client for querying language models.
    """

    def __init__(
        self,
        model_names: Union[List[str], str] = "openai/gpt-4o-mini",
        model_selection: Optional[BanditBase] = None,
        temperatures: Union[float, List[float]] = 0.75,
        max_tokens: Union[int, List[int]] = 4096,
        reasoning_efforts: Union[str, List[str]] = "auto",
        model_sample_probs: Optional[List[float]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the LLM client.

        Args:
            model_names: Model name(s) to query (OpenRouter format, e.g., "openai/gpt-4o")
            model_selection: Optional bandit for model selection
            temperatures: Temperature value(s) for sampling
            max_tokens: Maximum output tokens
            reasoning_efforts: Reasoning effort level(s)
            model_sample_probs: Probability weights for model sampling
            output_model: Optional Pydantic model for structured output
            verbose: Whether to enable verbose logging
        """
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        if isinstance(model_names, str):
            model_names = [model_names]
        self.model_names = model_names
        if not isinstance(model_selection, BanditBase):
            assert model_selection is None
            model_selection = FixedSampler(
                n_arms=len(model_names),
                prior_probs=model_sample_probs,
            )
        self.llm_selection = model_selection
        self.reasoning_efforts = reasoning_efforts
        self.model_sample_probs = model_sample_probs
        self.output_model = output_model
        self.structured_output = output_model is not None
        self.verbose = verbose

        # Per-model token tracking
        self._token_usage: Dict[str, Dict[str, int]] = {}

    def _track_tokens(self, result: QueryResult) -> None:
        """Track token usage per model."""
        if result is None or not hasattr(result, "model_name"):
            return
        model = result.model_name
        if model not in self._token_usage:
            self._token_usage[model] = {"input_tokens": 0, "output_tokens": 0}
        self._token_usage[model]["input_tokens"] += result.input_tokens
        self._token_usage[model]["output_tokens"] += result.output_tokens

    def get_token_usage(self) -> Dict[str, Dict[str, int]]:
        """Get per-model token usage.

        Returns:
            Dict mapping model names to token counts:
            {model_name: {'input_tokens': X, 'output_tokens': Y}}
        """
        return self._token_usage.copy()

    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
        self._token_usage = {}

    def get_kwargs(self) -> Dict:
        """Get sampled kwargs for a query."""
        posterior = self.llm_selection.posterior()
        if self.verbose:
            lines = ["==> SAMPLING:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<30} {prob:>8.4f}")
            logger.info("\n".join(lines))
        return sample_model_kwargs(
            model_names=self.model_names,
            temperatures=self.temperatures,
            max_tokens=self.max_tokens,
            reasoning_efforts=self.reasoning_efforts,
            model_sample_probs=posterior,
        )

    def query(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict] = [],
        llm_kwargs: Optional[Dict] = None,
    ) -> Optional[QueryResult]:
        """
        Execute a single synchronous query to the LLM.

        Args:
            msg: The message to query the LLM with
            system_msg: The system message to query the LLM with
            msg_history: Previous message history
            llm_kwargs: Additional LLM parameters

        Returns:
            QueryResult or None if all retries failed
        """
        if llm_kwargs is None:
            llm_kwargs = sample_model_kwargs(
                model_names=self.model_names,
                temperatures=self.temperatures,
                max_tokens=self.max_tokens,
                reasoning_efforts=self.reasoning_efforts,
                model_sample_probs=self.model_sample_probs,
            )
        if self.verbose:
            logger.info(f"==> QUERYING: {list(llm_kwargs.values())}")

        # Get posterior probabilities and create model_posteriors dict
        posterior = self.llm_selection.posterior()
        model_posteriors = dict(zip(self.model_names, posterior))
        model_posteriors = {k: float(v) for k, v in model_posteriors.items()}

        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = query(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **llm_kwargs,
                )
                self._track_tokens(result)
                if self.verbose and hasattr(result, "input_tokens"):
                    total_tokens = result.input_tokens + result.output_tokens
                    logger.info(f"==> QUERY: Tokens used: {total_tokens}")
                return result
            except Exception as e:
                logger.error(f"{try_count + 1}/{MAX_RETRIES} Error in query: {str(e)}")
                try_count += 1
        return None

    async def _query_async_with_retry(
        self,
        msg: str,
        system_msg: str,
        msg_history: List[Dict],
        llm_kwargs: Dict,
        model_posteriors: Optional[Dict[str, float]] = None,
    ) -> Optional[QueryResult]:
        """Execute an async query with retries."""
        try_count = 0
        while try_count < MAX_RETRIES:
            try:
                result = await query_async(
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    output_model=self.output_model,
                    model_posteriors=model_posteriors,
                    **llm_kwargs,
                )
                return result
            except Exception as e:
                logger.error(
                    f"{try_count + 1}/{MAX_RETRIES} Error in async query: {str(e)}"
                )
                try_count += 1
                if try_count < MAX_RETRIES:
                    await asyncio.sleep(1)
        return None

    async def batch_query_async(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
        llm_kwargs: Optional[List[Dict]] = None,
    ) -> List[QueryResult]:
        """
        Batch query the LLM using true async I/O.

        Args:
            num_samples: Number of samples to generate
            msg: The message(s) to query the LLM with
            system_msg: The system message(s) to query the LLM with
            msg_history: Previous conversation history
            llm_kwargs: Optional list of kwargs dictionaries for each query

        Returns:
            List of QueryResult objects
        """
        # Normalize inputs to lists
        if isinstance(msg, str):
            msg = [msg] * num_samples
        if isinstance(system_msg, str):
            system_msg = [system_msg] * num_samples
        if len(msg_history) == 0:
            msg_history = [[]] * num_samples
        elif isinstance(msg_history[0], dict):
            msg_history = [msg_history] * num_samples

        # Sample kwargs if not provided
        if llm_kwargs is None:
            posterior = self.llm_selection.posterior(samples=num_samples)
            if self.verbose:
                lines = [f"==> SAMPLING {num_samples} SAMPLES:"]
                for name, prob in zip(self.model_names, posterior):
                    lines.append(f"  {name:<30} {prob:>8.4f}")
                logger.info("\n".join(lines))

            llm_kwargs = [
                sample_model_kwargs(
                    model_names=self.model_names,
                    temperatures=self.temperatures,
                    max_tokens=self.max_tokens,
                    reasoning_efforts=self.reasoning_efforts,
                    model_sample_probs=posterior,
                )
                for _ in range(num_samples)
            ]
            model_posteriors = dict(zip(self.model_names, posterior))
            model_posteriors = {k: float(v) for k, v in model_posteriors.items()}
        else:
            model_posteriors = None

        # Create async tasks
        tasks = [
            self._query_async_with_retry(
                msg=msg[i],
                system_msg=system_msg[i],
                msg_history=msg_history[i],
                llm_kwargs=llm_kwargs[i],
                model_posteriors=model_posteriors,
            )
            for i in range(num_samples)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch query task {i}: {str(result)}")
            elif result is not None:
                final_results.append(result)
                self._track_tokens(result)

        if self.verbose:
            total_tokens = sum(
                r.input_tokens + r.output_tokens
                for r in final_results
                if hasattr(r, "input_tokens")
            )
            logger.info(f"==> BATCH: Total tokens used: {total_tokens}")

        return final_results

    def batch_kwargs_query(
        self,
        num_samples: int,
        msg: Union[str, List[str]],
        system_msg: Union[str, List[str]],
        msg_history: Union[List[Dict], List[List[Dict]]] = [],
    ) -> List[QueryResult]:
        """
        Batch query the LLM with auto-sampled kwargs.

        Uses async I/O internally for efficiency, with sync wrapper.

        Args:
            num_samples: Number of samples to generate
            msg: The message to query the LLM with
            system_msg: The system message to query the LLM with
            msg_history: Previous conversation history

        Returns:
            List of QueryResult objects
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Already in async context - cannot use asyncio.run
            # Fall back to creating tasks in existing loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.batch_query_async(num_samples, msg, system_msg, msg_history),
                )
                return future.result()
        else:
            # No running loop - use asyncio.run
            return asyncio.run(
                self.batch_query_async(num_samples, msg, system_msg, msg_history)
            )


def extract_between(
    content: str,
    start: str = "<json>",
    end: str = "</json>",
    return_dict: bool = True,
    fallback: bool = False,
) -> Optional[Union[str, dict]]:
    """
    Extract text from between start and end tags.

    Args:
        content: The input string containing tagged content
        start: Start tag
        end: End tag
        return_dict: Whether to parse as JSON and return dict
        fallback: Whether to try extracting from ``` blocks as fallback

    Returns:
        Extracted content as string or dict, or "none" if not found
    """
    match = re.search(f"{start}\\s*(.*?)\\s*{end}", content, re.DOTALL)
    if match:
        matched_str = match.group(1).strip()
        if return_dict:
            return json.loads(matched_str)
        else:
            return matched_str

    # Extracts any block between ``` and ```
    if fallback:
        match = re.search("```\\s*(.*?)\\s*```", content, re.DOTALL)
        if match:
            matched_str = match.group(1).strip()
            if return_dict:
                return json.loads(matched_str)
            else:
                return matched_str
    return "none"

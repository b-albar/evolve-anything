from typing import Dict, List, Optional


class QueryResult:
    """
    Result of an LLM query.

    Contains the response content and metadata about the query.
    """

    def __init__(
        self,
        content: str,
        msg: str,
        system_msg: str,
        new_msg_history: List[Dict],
        model_name: str,
        kwargs: Dict,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thought: str = "",
        model_posteriors: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the QueryResult.

        Args:
            content: The response content from the LLM
            msg: The original user message
            system_msg: The system prompt used
            new_msg_history: Updated message history including this exchange
            model_name: The model that was queried
            kwargs: The kwargs used for the query
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            thought: Any reasoning/thinking content from the model
            model_posteriors: Optional model posteriors for logging
        """
        self.content = content
        self.msg = msg
        self.system_msg = system_msg
        self.new_msg_history = new_msg_history
        self.model_name = model_name
        self.kwargs = kwargs
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thought = thought
        self.model_posteriors = model_posteriors or {}

    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "msg": self.msg,
            "system_msg": self.system_msg,
            "new_msg_history": self.new_msg_history,
            "model_name": self.model_name,
            "kwargs": self.kwargs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thought": self.thought,
            "model_posteriors": self.model_posteriors,
        }

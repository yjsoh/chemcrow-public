import time
import logging
import functools
from typing import Any, Callable
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import BaseTool

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("chemcrow")
logger.addHandler(logging.FileHandler("chemcrow.log", "w", encoding="utf-8"))


original_run = BaseTool._run


# class PromptAndResponseManager(BaseCallbackManager):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.logger = logging.getLogger("chemcrow")

#     def on_llm_start(self, serialized, prompts, **kwargs):
#         for handler in self.handlers:
#             handler.on_llm_start(serialized, prompts, **kwargs)

#     def on_llm_end(self, response, **kwargs):
#         for handler in self.handlers:
#             handler.on_llm_end(response, **kwargs)


class PromptAndResponseLogger(BaseCallbackHandler):
    already_tried = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("chemcrow")

    def on_llm_start(self, serialized, prompts, **kwargs):
        for i, prompt in enumerate(prompts):
            if prompt in self.already_tried:  # skip duplicates
                prompts[i] = (
                    "<EOS>"  # TODO: get it from the tokenizer instead of hardcoding.
                )
                continue
            self.logger.info("[LLM PROMPT]:\n%s\n", prompt)
            self.already_tried.add(prompt)

    def on_llm_end(self, response, **kwargs):
        self.logger.info("[LLM RESPONSE]:\n%s\n", response.generations[0][0].text)

    def on_tool_start(self, serialized, input_args, **kwargs):
        self.logger.info("[TOOL START]:\n%s\n", input_args)

    def on_tool_end(self, output, **kwargs):
        self.logger.info("[TOOL END]:\n%s\n", output)


def log_tool_execution(original_run: Callable) -> Callable:
    """
    Decorator to log tool execution.
    """

    @functools.wraps(original_run)
    def wrapper(self: BaseTool, *args: Any, **kwargs: Any) -> Any:
        # Log tool start
        logger.info(f"Starting tool execution: {self.name}")
        logger.info(f"Tool description: {self.description}")
        logger.info(f"Tool input: {args[0] if args else kwargs}")

        rng = nvtx.start_range(message="my_message", color="blue")

        try:
            # Execute the original run method
            start = time.time()
            result = original_run(self, *args, **kwargs)
            elapsed = time.time() - start

            # Log successful execution
            logger.info(f"Tool {self.name} completed successfully in {elapsed:.2f}s")
            logger.info(f"Tool output: {result}")
            nvtx.end_range(rng)

            return result
        except Exception as e:
            # Log any errors
            logger.error(f"Tool {self.name} failed with error: {str(e)}")
            nvtx.end_range(rng)
            raise

    return wrapper


def patch_base_tool():
    """
    Monkey patch the BaseTool._run method to add logging.
    """
    BaseTool._run = log_tool_execution(BaseTool._run)
    logger.info("Time: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Successfully patched BaseTool._run with logging functionality")

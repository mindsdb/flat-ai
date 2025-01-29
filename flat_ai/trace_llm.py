import json
import logging
import sys
import time
from io import StringIO

import openai

# Create a custom logging level for OpenAI trace
OPENAI_TRACE = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(OPENAI_TRACE, "OPENAI_TRACE")


def openai_trace(self, message, *args, **kwargs):
    if self.isEnabledFor(OPENAI_TRACE):
        self._log(OPENAI_TRACE, message, args, **kwargs)


logging.Logger.openai_trace = openai_trace

# Create a logger instance
logger = logging.getLogger(__name__)

# Set the logger level to OPENAI_TRACE
logger.setLevel(OPENAI_TRACE)

# Disable propagation to prevent logs from being printed to the console
logger.propagate = False


def configure_logging(log_destination=None):
    global logger
    print(f"log_destination: {log_destination}")
    if log_destination is None:
        # If no destination is provided, do nothing (logs are not shown or saved)
        return

    if isinstance(log_destination, str):
        # If a string is provided, assume it's a file name
        handler = logging.FileHandler(log_destination, mode="w")
    elif isinstance(log_destination, StringIO):
        # If a StringIO object is provided, use it as a memory file
        handler = logging.StreamHandler(log_destination)
    else:
        raise ValueError(
            "Invalid log_destination. Must be a file name (str) or StringIO object."
        )

    handler.setLevel(OPENAI_TRACE)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Remove any existing handlers
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # Add the new handler
    logger.addHandler(handler)


class MyOpenAI(sys.modules["openai"].OpenAI):
    def _process_json_values(self, obj):
        if isinstance(obj, dict):
            return {key: self._process_json_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._process_json_values(item) for item in obj]
        elif isinstance(obj, str) and self._is_valid_json(obj):
            return json.loads(obj)
        return obj

    def post(self, *args, **kwargs):
        # Generate a unique ID for this call
        call_id = f"{int(time.time())}"
        start_time = time.time()
        logger.openai_trace(f"\n\n=========\nGenerated unique call ID: {call_id}\n=========\n")
        # Log the function call
        logger.openai_trace(f"[{call_id}] Calling OpenAI post method")
        logger.openai_trace(
            f"[{call_id}] Arguments: {json.dumps(args, indent=2, ensure_ascii=True)}"
        )

        # Process kwargs to handle JSON strings recursively
        processed_kwargs = self._process_json_values(kwargs)
        logger.openai_trace(
            f"[{call_id}] Keyword arguments: {json.dumps(processed_kwargs, indent=2, default=str, ensure_ascii=True)}"
        )

        # Call the original post method with original kwargs
        result = super().post(*args, **kwargs)

        # Check if this is a streaming response
        is_streaming = isinstance(kwargs.get("body"), dict) and kwargs["body"].get(
            "stream", False
        )

        # If the result is an openai.Stream, wrap it to log chunks as they're iterated
        if isinstance(result, openai.Stream):
            original_stream = result

            def logging_stream():
                buffer = ""
                for chunk in original_stream:
                    string = str(chunk.choices[0].delta.content)
                    buffer += string
                    if chunk.choices[0].delta.content is None:
                        logger.openai_trace(f"[{call_id}] Response Stream: {buffer}")
                    yield chunk

            result = logging_stream()

            # For streaming responses, we only log timing metrics
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.openai_trace(f"[{call_id}] Timing metrics:")
            logger.openai_trace(f"[{call_id}] - Total time: {elapsed_time:.2f} seconds")

            return result

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Extract relevant fields from ChatCompletion object
        if isinstance(result, openai.types.CreateEmbeddingResponse):
            return result

        result_dict = {
            "id": result.id,
            "model": result.model,
            "choices": [
                {
                    "finish_reason": choice.finish_reason,
                    "message": {
                        "content": (
                            json.loads(choice.message.content)
                            if isinstance(choice.message.content, str)
                            and (
                                isinstance(kwargs.get("body"), dict)
                                and isinstance(
                                    kwargs["body"].get("response_format"), dict
                                )
                                and kwargs["body"]["response_format"].get("type")
                                == "json_object"
                                or self._is_valid_json(choice.message.content)
                            )
                            else choice.message.content
                        ),
                        "role": choice.message.role,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in choice.message.tool_calls
                        ]
                        if choice.message.tool_calls
                        else None,
                    },
                }
                for choice in result.choices
            ],
            "usage": {
                "completion_tokens": result.usage.completion_tokens,
                "prompt_tokens": result.usage.prompt_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        }

        # Calculate and log timing metrics
        tokens_per_second = (
            result.usage.total_tokens / elapsed_time if elapsed_time > 0 else 0
        )
        logger.openai_trace(f"[{call_id}] Timing metrics:")
        logger.openai_trace(f"[{call_id}] - Total time: {elapsed_time:.2f} seconds")
        logger.openai_trace(f"[{call_id}] - Tokens per second: {tokens_per_second:.2f}")
        logger.openai_trace(
            f"[{call_id}] - Result: {json.dumps(result_dict, indent=2, ensure_ascii=True)}"
        )

        return result

    def _is_valid_json(self, string):
        if not isinstance(string, str):
            return False
        try:
            json.loads(string)
            return True
        except json.JSONDecodeError:
            return False


# sys.modules['openai'].OpenAI = MyOpenAI

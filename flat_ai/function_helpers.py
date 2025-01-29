from typing import Callable, Dict, Any, Optional, List
import inspect
import re
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

# Helper functions and classes for handling function calls and OpenAI function descriptions

def create_openai_function_description(func: Callable) -> Dict[str, Any]:
    """
    Takes a function and returns an OpenAI function description.
    This converts a Python function into a format that OpenAI's API can understand and use.

    Args:
        func (Callable): The function to create a description for.

    Returns:
        Dict[str, Any]: A dictionary containing the OpenAI function description with name,
                       description, and parameters schema.
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)

    # Initialize the basic structure of the function description
    function_description = {
        "name": func.__name__,
        "description": docstring.split("\n")[0] if docstring else "",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    # Process each parameter of the function
    for param_name, param in signature.parameters.items():
        param_info = {
            "description": ""  # Initialize description
        }

        # Handle type annotations and convert Python types to JSON Schema types
        if param.annotation != inspect.Parameter.empty:
            if hasattr(param.annotation, "__origin__"):
                # Handle generic types like List and Dict
                if param.annotation.__origin__ == list:
                    param_info["type"] = "array"
                    if hasattr(param.annotation, "__args__"):
                        inner_type = param.annotation.__args__[0]
                        # Map Python types to JSON Schema types for array items
                        if inner_type == str:
                            param_info["items"] = {"type": "string"}
                        elif inner_type == int:
                            param_info["items"] = {"type": "integer"}
                        elif inner_type == float:
                            param_info["items"] = {"type": "number"}
                        elif inner_type == bool:
                            param_info["items"] = {"type": "boolean"}
                elif param.annotation.__origin__ == dict:
                    param_info["type"] = "object"
            else:
                # Handle simple types
                if param.annotation == str:
                    param_info["type"] = "string"
                elif param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
        else:
            # Default to string if no type info available
            param_info["type"] = "string"

        # Handle default values and required parameters
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default
        else:
            function_description["parameters"]["required"].append(param_name)

        # Extract parameter descriptions from the function's docstring
        if docstring:
            param_pattern = re.compile(rf"{param_name}(\s*\([^)]*\))?:\s*(.*)")
            param_matches = [
                param_pattern.match(line.strip()) for line in docstring.split("\n")
            ]
            param_lines = [match.group(2) for match in param_matches if match]
            if param_lines:
                param_desc = param_lines[0].strip()
                param_info["description"] = param_desc

        function_description["parameters"]["properties"][param_name] = param_info

    return {"type": "function", "function": function_description}


class FunctionCall(BaseModel):
    """
    Represents a single function call with its arguments and result.
    Inherits from Pydantic BaseModel for automatic validation.
    """
    function: Any  # The actual function to be called
    arguments: Dict[str, Any] = Field(default_factory=dict)  # Arguments to pass to the function
    result: Optional[Any] = None  # Stores the result after function execution

    model_config = {
        "arbitrary_types_allowed": True  # Allows any Python type to be stored
    }

    def validate_arguments(self):
        """
        Validates that the provided arguments match the function's signature.
        Raises ValueError if arguments are invalid.
        """
        sig = inspect.signature(self.function)
        try:
            # Try binding the arguments to the function signature
            sig.bind(**self.arguments)
        except TypeError as e:
            raise ValueError(f"Invalid arguments for function {self.function.__name__}: {str(e)}")

class FunctionsToCall():
    """
    Manages a collection of function calls and executes them in parallel.
    """
    def __init__(self):
        self.functions: list[FunctionCall] = []  # List to store function calls

    def __call__(self):
        """
        Executes all stored functions in parallel using a ThreadPoolExecutor.
        Validates arguments before execution and stores results.

        Returns:
            List of results from function calls, including any exceptions that occurred.
        """
        # Validate all function arguments before executing
        for func_call in self.functions:
            func_call.validate_arguments()

        with ThreadPoolExecutor() as executor:
            futures = []
            # Submit all functions to the executor
            for func_call in self.functions:
                futures.append(
                    executor.submit(func_call.function, **func_call.arguments)
                )
            
            # Store results in the FunctionCall objects
            for func_call, future in zip(self.functions, futures):
                try:
                    func_call.result = future.result()
                except Exception as e:
                    func_call.result = e  # Store exception as result if function fails
            
        return [func_call.result for func_call in self.functions]
    
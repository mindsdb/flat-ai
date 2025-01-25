"""
🤖 F.l.a.t. (Frameworkless LLM Agent... Thing)! 🤖

Look, we get it. You're tired of AI libraries that are more complex than your ex's emotional baggage.
Enter FlatAI - the AI wrapper that's flatter than your first pancake attempt!

This bad boy wraps around OpenAI's API like a warm tortilla around your favorite burrito fillings,
making it easier to digest and way less likely to cause mental indigestion.

Key Features:
- 🎯 Simple: So simple, your rubber duck could probably use it
- 🔄 Retries: Because even AI needs a second chance (or three)
- 🧠 Context Management: Like a brain, but one you can actually control
- 🎲 Function Picking: Let AI choose your functions like your mom chooses your clothes
- 📝 Object Generation: Creates objects faster than your cat creates chaos
- 🔄 Logic Blocks: if/else, loops, switch cases - all the Python goodies you know and love
- 🎭 Dynamic Flow: Control your AI's behavior with familiar programming patterns

License: MIT (Because sharing is caring, and lawyers are expensive)

Author: Your Friendly Neighborhood AI Wrangler
"""


import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union, Iterable
import openai
from pydantic import BaseModel, Field
from typing import Literal
import time
from flat_ai.trace_llm import MyOpenAI

openai.OpenAI = MyOpenAI
class Boolean(BaseModel):
    result: bool = Field(description="The true/false result based on the question and context")

class FlatAI:
    def __init__(self, client: Optional[openai.OpenAI] = None, model: str = "gpt-4", retries: int = 3, base_url: str = "https://api.openai.com/v1", api_key: Optional[str] = None):
        if client:
            self.client = client
        elif api_key:
            self.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        else:
            raise ValueError("Must provide either client or api_key")
            
        self.model = model
        self.retries = retries
        self._context = {}

    def _retry_on_error(self, func: Callable, *args, **kwargs) -> Any:
        """Helper method to retry operations on failure"""
        last_exception = None
        for attempt in range(self.retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
                raise Exception(f"Operation failed after {self.retries} attempts. Last error: {str(last_exception)}")

    def set_context(self, **kwargs):
        """Set the context for future LLM interactions"""
        self._context = kwargs

    def add_context(self, **kwargs): 
        """Add additional context while preserving existing context"""
        self._context.update(kwargs)

    def clear_context(self):
        """Clear all context"""
        self._context = {}

    def delete_from_context(self, *keys):
        """Remove specific keys from context"""
        for key in keys:
            self._context.pop(key, None)

    def _build_messages(self, *message_parts, **kwargs) -> List[Dict[str, str]]:
        """Build message list with context as system message if present"""
        messages = []
        
        if self._context:
            context_dict = {}
            for key, value in self._context.items():
                if isinstance(value, BaseModel):
                    context_dict[key] = json.loads(value.model_dump_json())
                else:
                    context_dict[key] = str(value)
            messages.append({"role": "system", "content": json.dumps(context_dict, indent=2)})

        if kwargs:
            extra_context_dict = {}
            for key, value in kwargs.items():
                if isinstance(value, BaseModel):
                    extra_context_dict[key] = json.loads(value.model_dump_json())
                else:
                    extra_context_dict[key] = str(value)
            messages.append({"role": "system", "content": json.dumps(extra_context_dict, indent=2)})

        messages.extend(message_parts)
        return messages

    def is_true(self, _question: str, **kwargs) -> bool:
        class IsItTrue(BaseModel):
            is_it_true: bool
        """Ask a yes/no question and get a boolean response"""
        ret = self.generate_object(IsItTrue, question=_question, **kwargs)
        return ret.is_it_true

    def classify(self, options: Dict[str, str], **kwargs) -> str:
        """Get a key from provided options based on context"""
        class Classification(BaseModel):
            choice: str = Field(description="The selected classification key", enum=list(options.keys()))

        def _execute():
            if not options:
                raise ValueError("Options dictionary cannot be empty")
            
            result = self.generate_object(Classification, options=options)
            return result.choice
            
        return self._retry_on_error(_execute)

    def generate_object(self, schema_class: Type[BaseModel | Any], **kwargs) -> Any:
        """Generate an object matching the provided schema"""
        def _execute():
            # Handle typing generics (List, Dict, etc)
            if hasattr(schema_class, "__origin__"):
                if schema_class.__origin__ is list:
                    item_type = schema_class.__args__[0]
                    if issubclass(item_type, BaseModel):
                        schema = {
                            "type": "array",
                            "items": item_type.model_json_schema()
                        }
                    else:
                        schema = {
                            "type": "array",
                            "items": {
                                "type": {
                                    str: "string",
                                    int: "integer",
                                    float: "number",
                                    bool: "boolean"
                                }.get(item_type, "string")
                            }
                        }
                else:
                    raise ValueError(f"Unsupported generic type: {schema_class}")
            # Handle basic Python types
            elif schema_class in (list, dict, str, int, float, bool):
                schema = {
                    "type": {
                        list: "array",
                        dict: "object",
                        str: "string",
                        int: "integer",
                        float: "number",
                        bool: "boolean"
                    }.get(schema_class, "string")
                }
                if schema["type"] == "array":
                    schema["items"] = {"type": "string"}  # Default to string items
            # Handle Pydantic models
            elif isinstance(schema_class, type) and issubclass(schema_class, BaseModel):
                schema = schema_class.model_json_schema()
            else:
                raise ValueError(f"Unsupported schema type: {schema_class}")

            messages = self._build_messages(
                {"role": "user", "content": "Based on the provided context and information, generate a complete and accurate object that precisely matches the schema. Use all relevant details to populate the fields with meaningful, appropriate values that best represent the data."},
                **kwargs
            )

            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object", "schema": schema},
                messages=messages
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Handle list of Pydantic models
            if (hasattr(schema_class, "__origin__") and 
                schema_class.__origin__ is list and 
                issubclass(schema_class.__args__[0], BaseModel)):
                return [schema_class.__args__[0].model_validate(item) for item in result]
            # Handle single Pydantic model
            elif isinstance(schema_class, type) and issubclass(schema_class, BaseModel):
                return schema_class.model_validate(result)
            
            return result
        return self._retry_on_error(_execute)

    def call_function(self, func: Callable, **kwargs) -> Any:
        """Call a function with AI-determined arguments"""
        func, args = self.pick_a_function("", [func], **kwargs)
        return func(**args)

    def pick_a_function(self, instructions: str, functions: List[Callable], **kwargs) -> tuple[Callable, Dict]:
        """Pick appropriate function and arguments based on instructions"""
        def _execute():
            tools = [create_openai_function_description(func) for func in functions]

            messages = self._build_messages(
                {"role": "system", "content": instructions},
                {"role": "user", "content": "Based on all the provided context and information, analyze and select the most appropriate function from the available options. Then, determine and specify the optimal parameters for that function to achieve the intended outcome."},
                **kwargs
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools
            )

            tool_call = response.choices[0].message.tool_calls[0]
            chosen_func = next(f for f in functions if f.__name__ == tool_call.function.name)
            
            args = json.loads(tool_call.function.arguments, strict=False)
            # Convert string lists back to actual lists
            for key, value in args.items():
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    try:
                        args[key] = json.loads(value, strict=False)
                    except json.JSONDecodeError:
                        pass

            return chosen_func, args
        return self._retry_on_error(_execute)

    def get_string(self, prompt: str, **kwargs) -> str:
        """Get a simple string response from the LLM"""
        def _execute():
            messages = self._build_messages(
                {"role": "user", "content": prompt},
                **kwargs
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        return self._retry_on_error(_execute)

    def get_stream(self, prompt: str, **kwargs) -> Iterable[str]:
        """Get a streaming response from the LLM"""
        def _execute():
            messages = self._build_messages(
                {"role": "user", "content": prompt},
                **kwargs
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        return self._retry_on_error(_execute)

def create_openai_function_description(func: Callable) -> Dict[str, Any]:
    """
    Takes a function and returns an OpenAI function description.

    Args:
        func (Callable): The function to create a description for.

    Returns:
        Dict[str, Any]: A dictionary containing the OpenAI function description.
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)

    function_description = {
        "name": func.__name__,
        "description": docstring.split("\n")[0] if docstring else "",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    for param_name, param in signature.parameters.items():
        param_info = {
            "description": ""  # Initialize description
        }

        # Try to get type from type annotation first
        if param.annotation != inspect.Parameter.empty:
            if hasattr(param.annotation, '__origin__'):
                if param.annotation.__origin__ == list:
                    param_info["type"] = "array"
                    if hasattr(param.annotation, '__args__'):
                        inner_type = param.annotation.__args__[0]
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

        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default
        else:
            function_description["parameters"]["required"].append(param_name)

        # Extract parameter description from docstring
        if docstring:
            param_pattern = re.compile(rf"{param_name}(\s*\([^)]*\))?:\s*(.*)")
            param_matches = [param_pattern.match(line.strip()) for line in docstring.split("\n")]
            param_lines = [match.group(2) for match in param_matches if match]
            if param_lines:
                param_desc = param_lines[0].strip()
                param_info["description"] = param_desc

        function_description["parameters"]["properties"][param_name] = param_info

    return {"type": "function", "function": function_description}








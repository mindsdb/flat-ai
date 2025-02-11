from pydantic import BaseModel, Field
from typing import List

INSTRUCTION_MESSAGE = "Based on the provided context and information, generate a complete and accurate Python code that precisely matches the USERs request. Use all relevant details to populate the code with meaningful, appropriate values that best represent the data."

class PythonCodeObject(BaseModel):
    """
    Represents a code object with a name and code.
    """
    code_notes: str
    raw_code: str

    def __str__(self):
        return f"PythonCodeObject <obj.code_notes>, <obj.raw_code>"

    def __repr__(self):
        return self.__str__()

    
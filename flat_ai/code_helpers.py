from pydantic import BaseModel, Field
from typing import List

class PythonCodeObject(BaseModel):
    """
    Represents a code object with a name and code.
    """
    code_notes: str
    raw_code: str

    def __str__(self):
        return f"{self.raw_code}"

    def __repr__(self):
        return self.__str__()

    
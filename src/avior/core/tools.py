"""Tools the LLM can call.

A `Tool` is a named operation the LLM may invoke during a run.  It declares the
shape of its arguments as a Pydantic model and runs them in `execute`.

The LLM is shown the tool's name, description, and the JSON schema of its
arguments model.  When the LLM asks to call the tool, the arguments it sends are
validated and coerced through that same model before `execute` runs - so
`execute` always receives a typed, validated arguments object, never a raw dict.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class Tool[ArgsT: BaseModel, ResultT](ABC):
    """A named operation the LLM can invoke, with typed arguments.

    Subclass it, set `name`, `description`, and `args_model`, and implement
    `execute`.  `args_model` is the single source of truth for both the schema
    sent to the LLM and the validation/coercion of the arguments it returns.

    The two type parameters make a single tool subclass type-safe: `execute`
    takes that subclass's `args_model` instance and returns its own result
    type.  A collection of different tools cannot keep these parameters,
    though - each tool has its own pair, and Python cannot express "a tool
    with some arguments model and some result type" (it has no existential
    types).  So a mixed tool collection is typed `Tool[Any, Any]`, and the
    per-tool types are re-established at runtime: the runner validates the
    incoming arguments through each tool's `args_model` before calling
    `execute`.
    """

    name: str
    """The tool's name, as exposed to the LLM."""

    description: str
    """A natural-language description of what the tool does, for the LLM."""

    args_model: type[ArgsT]
    """The Pydantic model describing the tool's arguments.  Two roles:

    - its JSON schema is sent to the LLM as the tool's input schema;
    - arguments the LLM returns are validated and coerced through it before
      reaching `execute`.
    """

    @abstractmethod
    async def execute(self, args: ArgsT) -> ResultT:
        """Run the tool with validated `args` and return its result."""

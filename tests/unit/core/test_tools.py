"""Tests for `avior.core.tools`."""

from pydantic import BaseModel, ConfigDict, Field

from avior.core.context import RunContext
from avior.core.tools import Tool


class _AddArgs(BaseModel):
    a: int
    b: int


class _Add(Tool[_AddArgs, int]):
    """A tool that adds two integers."""

    name = "add"
    description = "Add two integers."
    args_model = _AddArgs

    async def execute(self, ctx: RunContext[object], args: _AddArgs) -> int:
        return args.a + args.b


async def test_tool_coerces_args_via_args_model_then_executes() -> None:
    """`args_model` validates and coerces raw args before `execute` runs."""

    # GIVEN a tool whose args model has integer fields
    tool = _Add()

    # WHEN raw arguments (a string among them) are validated and the tool runs
    args = tool.args_model.model_validate({"a": "2", "b": 3})
    ctx = RunContext[object](deps=None, tool_name="add", tool_call_id="c1", run_step=1)
    result = await tool.execute(ctx, args)

    # THEN the string was coerced to an int and the tool returned their sum
    assert result == 5


class _DescribedArgs(BaseModel):
    """Args documenting one field by docstring, one by both ways at once."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    city: str
    """from docstring"""

    units: str = Field(default="celsius", description="from Field")
    """from docstring"""


class _Described(Tool[_DescribedArgs, str]):
    """A tool documenting its arguments through its model."""

    name = "described"
    description = "A tool whose arguments are documented on the model."
    args_model = _DescribedArgs

    async def execute(self, ctx: RunContext[object], args: _DescribedArgs) -> str:
        return args.city


# These two tests guard promises the `Tool.args_model` docstring makes about
# documenting fields.  Both behaviors are Pydantic's, not ours, so the tests are
# canaries: a failure means Pydantic changed, and the docstring claim must be
# revised to match.


def test_attribute_docstring_becomes_field_description() -> None:
    """A field's docstring becomes its schema description with the config on.

    With `use_attribute_docstrings=True`, the docstring under a field reaches
    the LLM as that field's description.
    """

    # GIVEN a tool whose field is documented only by a docstring under it
    tool = _Described()

    # WHEN the args schema that the provider sends to the LLM is generated
    schema = tool.args_model.model_json_schema()

    # THEN the docstring is the field's description in the schema
    assert schema["properties"]["city"]["description"] == "from docstring"


def test_field_description_wins_over_attribute_docstring() -> None:
    """`Field(description=...)` wins where a field also has a docstring."""

    # GIVEN a tool whose field has both a `Field` description and a docstring
    tool = _Described()

    # WHEN the args schema that the provider sends to the LLM is generated
    schema = tool.args_model.model_json_schema()

    # THEN the `Field` description is the one in the schema
    assert schema["properties"]["units"]["description"] == "from Field"

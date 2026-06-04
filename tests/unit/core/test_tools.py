"""Tests for `avior.core.tools`."""

from pydantic import BaseModel

from avior.core.tools import Tool


class _AddArgs(BaseModel):
    a: int
    b: int


class _Add(Tool[_AddArgs, int]):
    """A tool that adds two integers."""

    name = "add"
    description = "Add two integers."
    args_model = _AddArgs

    async def execute(self, args: _AddArgs) -> int:
        return args.a + args.b


async def test_tool_coerces_args_via_args_model_then_executes() -> None:
    """`args_model` validates and coerces raw args before `execute` runs."""

    # GIVEN a tool whose args model has integer fields
    tool = _Add()

    # WHEN raw arguments (a string among them) are validated and the tool runs
    args = tool.args_model.model_validate({"a": "2", "b": 3})
    result = await tool.execute(args)

    # THEN the string was coerced to an int and the tool returned their sum
    assert result == 5

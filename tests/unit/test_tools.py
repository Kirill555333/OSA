import pytest

from osa.tools import CalculatorTool, ToolInterface, ToolRegistry


def test_calculator_implements_tool_interface() -> None:
    tool = CalculatorTool()

    assert isinstance(tool, ToolInterface)
    assert tool.name == "calculator"


def test_calculator_evaluates_expression() -> None:
    tool = CalculatorTool()

    result = tool.execute(
        {
            "expression": "2 + 3 * 4",
        }
    )

    assert result.success is True
    assert result.output == "14"
    assert result.error is None


def test_calculator_supports_parentheses() -> None:
    tool = CalculatorTool()

    result = tool.execute(
        {
            "expression": "(10 + 5) * 2",
        }
    )

    assert result.success is True
    assert result.output == "30"


def test_calculator_rejects_unsupported_code() -> None:
    tool = CalculatorTool()

    result = tool.execute(
        {
            "expression": "__import__('os').getcwd()",
        }
    )

    assert result.success is False
    assert result.error is not None


def test_calculator_handles_division_by_zero() -> None:
    tool = CalculatorTool()

    result = tool.execute(
        {
            "expression": "10 / 0",
        }
    )

    assert result.success is False
    assert result.error is not None


def test_calculator_requires_expression_string() -> None:
    tool = CalculatorTool()

    result = tool.execute(
        {
            "expression": 123,
        }
    )

    assert result.success is False
    assert "must be a string" in result.error


def test_registry_registers_and_returns_tools() -> None:
    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    assert len(registry) == 1
    assert registry.names() == ("calculator",)
    assert registry.get("calculator") is calculator


def test_registry_rejects_duplicate_tools() -> None:
    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    with pytest.raises(ValueError):
        registry.register(calculator)


def test_registry_describes_tools() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    assert registry.describe() == (
        {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
        },
    )

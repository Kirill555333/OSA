"""Safe arithmetic calculator for OSA."""

from __future__ import annotations

import ast
import operator
from typing import Any, Mapping

from osa.tools.registry import ToolInterface, ToolResult


class CalculatorTool(ToolInterface):
    """Evaluate basic arithmetic expressions safely."""

    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "calculator"

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return "Evaluate a basic arithmetic expression."

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for calculator arguments."""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Evaluate the expression provided in the arguments."""
        expression = arguments.get("expression")

        if not isinstance(expression, str):
            return ToolResult(
                success=False,
                error="Argument 'expression' must be a string.",
            )

        expression = expression.strip()

        if not expression:
            return ToolResult(
                success=False,
                error="Argument 'expression' cannot be empty.",
            )

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate(tree.body)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(
                success=False,
                error=f"Invalid arithmetic expression: {exc}",
            )

        return ToolResult(
            success=True,
            output=str(result),
        )

    def _evaluate(self, node: ast.AST) -> int | float:
        """Recursively evaluate an allowed arithmetic AST."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value,
                (int, float),
            ):
                raise ValueError("Only integer and floating-point numbers are allowed.")

            return node.value

        if isinstance(node, ast.BinOp):
            operator_function = self._operators.get(type(node.op))

            if operator_function is None:
                raise ValueError(
                    f"Operator '{type(node.op).__name__}' is not allowed."
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            return operator_function(left, right)

        if isinstance(node, ast.UnaryOp):
            operator_function = self._operators.get(type(node.op))

            if operator_function is None:
                raise ValueError(
                    f"Operator '{type(node.op).__name__}' is not allowed."
                )

            operand = self._evaluate(node.operand)

            return operator_function(operand)

        raise ValueError(
            f"Expression element '{type(node).__name__}' is not allowed."
        )

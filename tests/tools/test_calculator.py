import pytest

from yapa.tools.core.calculator import calculator


class TestCalculator:
    async def test_add(self):
        result = await calculator.execute(expression="1 + 2")
        assert result == 3

    async def test_subtract(self):
        result = await calculator.execute(expression="10 - 4")
        assert result == 6

    async def test_multiply(self):
        result = await calculator.execute(expression="3 * 7")
        assert result == 21

    async def test_divide(self):
        result = await calculator.execute(expression="15 / 3")
        assert result == 5.0

    async def test_float_result(self):
        result = await calculator.execute(expression="7 / 2")
        assert result == 3.5

    async def test_invalid_expression_raises(self):
        with pytest.raises(Exception):
            await calculator.execute(expression="invalid +")

    async def test_name_and_metadata(self):
        assert calculator.name == "calculator"
        assert calculator.needs_approval is False
        assert "expression" in calculator.parameters["properties"]

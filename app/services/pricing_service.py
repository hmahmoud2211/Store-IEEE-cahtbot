"""Programmatic price arithmetic. The LLM never computes totals or rankings."""

from __future__ import annotations

from decimal import Decimal
from itertools import combinations

from app.ingestion.price_parser import PriceRange, format_amount
from app.rag.types import CalculationResult, ComparisonResult, ComponentFact, LineItem


class PricingService:
    def estimate(
        self,
        items: list[tuple[ComponentFact, Decimal, str, bool, list[str]]],
        unresolved: list[str] | None = None,
    ) -> CalculationResult:
        """items: (component, quantity, mention, assumed, alternatives)."""
        lines: list[LineItem] = []
        unpriced: list[str] = list(unresolved or [])
        for component, quantity, mention, assumed, alternatives in items:
            if component.price is None:
                unpriced.append(component.name)
                continue
            lines.append(
                LineItem(
                    component=component,
                    quantity=quantity,
                    line_total=component.price.multiply(quantity),
                    mention=mention,
                    assumed=assumed,
                    alternatives=alternatives,
                )
            )
        currencies = {line.line_total.currency for line in lines}
        total: PriceRange | None = None
        mixed = len(currencies) > 1
        if lines and not mixed:
            total = PriceRange(
                min=sum((line.line_total.min for line in lines), Decimal(0)),
                max=sum((line.line_total.max for line in lines), Decimal(0)),
                currency=currencies.pop(),
                unit="total",
            )
        return CalculationResult(items=lines, total=total, unresolved=unpriced, mixed_currency=mixed)

    def compare(self, components: list[ComponentFact]) -> ComparisonResult:
        statements: list[str] = []
        priced = [c for c in components if c.price is not None]
        for c in components:
            if c.price is None:
                statements.append(f"{c.name}: no price listed in the knowledge base.")

        for a, b in combinations(priced, 2):
            statements.append(self._pair_statement(a, b))

        if len(priced) > 2:
            cheapest = min(priced, key=lambda c: (c.price.min, c.price.max))
            statements.append(f"Lowest starting price overall: {cheapest.name} ({cheapest.price.format()}).")

        categories = {c.category for c in components}
        if len(categories) == 1 and len(components) > 1:
            statements.append(f"All compared components are in the same category: {categories.pop()}.")
        elif len(components) > 1:
            statements.append("Categories: " + "; ".join(f"{c.name} = {c.category}" for c in components) + ".")
        return ComparisonResult(components=components, statements=statements)

    @staticmethod
    def _pair_statement(a: ComponentFact, b: ComponentFact) -> str:
        pa, pb = a.price, b.price
        assert pa is not None and pb is not None
        if pa.currency != pb.currency:
            return f"{a.name} and {b.name} are priced in different currencies ({pa.currency} vs {pb.currency})."
        if pa.max < pb.min or pb.max < pa.min:
            cheap, dear = (a, b) if pa.max < pb.min else (b, a)
            pc, pd = cheap.price, dear.price
            assert pc is not None and pd is not None
            if not pc.is_range and not pd.is_range:
                diff = pd.min - pc.min
                return (
                    f"{cheap.name} ({pc.format()}) is cheaper than {dear.name} ({pd.format()}) "
                    f"by {format_amount(diff)} {pc.currency}."
                )
            low, high = pd.min - pc.max, pd.max - pc.min
            return (
                f"{cheap.name} ({pc.format()}) is cheaper than {dear.name} ({pd.format()}); "
                f"the difference is roughly {format_amount(low)}–{format_amount(high)} {pc.currency}."
            )
        if pa.min == pb.min and pa.max == pb.max:
            return f"{a.name} and {b.name} have the same approximate price ({pa.format()})."
        return (
            f"The price ranges of {a.name} ({pa.format()}) and {b.name} ({pb.format()}) overlap, "
            "so neither is definitely cheaper."
        )

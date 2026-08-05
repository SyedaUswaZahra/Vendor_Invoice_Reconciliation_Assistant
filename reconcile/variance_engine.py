from typing import List, Tuple, Dict

from schemas.line_item import LineItemSchema


class VarianceEngine:
    """Deterministic pure-Python comparator for qty, price, total variances (FR-5, FR-6)."""

    def __init__(self, tolerance: float = 0.05) -> None:
        """Store a configurable tolerance fraction (default 5%)."""
        self.tolerance = tolerance

    def compute(
        self, invoice_item: LineItemSchema, po_item: LineItemSchema
    ) -> Dict[str, float]:
        """Compute per-line variances dict with keys qty_variance, price_variance, total_variance."""
        qty_diff = abs(invoice_item.quantity - po_item.quantity)
        price_diff = abs(invoice_item.unit_price - po_item.unit_price)
        total_diff = abs(invoice_item.line_total - po_item.line_total)

        qty_variance = qty_diff / po_item.quantity if po_item.quantity else 0.0
        price_variance = (
            price_diff / po_item.unit_price if po_item.unit_price else 0.0
        )
        total_variance = (
            total_diff / po_item.line_total if po_item.line_total else 0.0
        )

        return {
            "qty_variance": qty_variance,
            "price_variance": price_variance,
            "total_variance": total_variance,
        }

    def is_flagged(self, variances: Dict[str, float]) -> bool:
        """Return True if any variance exceeds the configured tolerance."""
        return any(
            variance > self.tolerance for variance in variances.values()
        )

    def reconcile_batch(
        self, matches: List[Tuple[LineItemSchema, LineItemSchema, float]]
    ) -> List[Dict]:
        """Compute variances for all matches and flag mismatches."""
        results: List[Dict] = []
        for invoice_item, po_item, score in matches:
            variances = self.compute(invoice_item, po_item)
            flagged = self.is_flagged(variances)
            results.append(
                {
                    "invoice_item": invoice_item,
                    "po_item": po_item,
                    "similarity_score": score,
                    "variances": variances,
                    "flagged": flagged,
                }
            )
        return results

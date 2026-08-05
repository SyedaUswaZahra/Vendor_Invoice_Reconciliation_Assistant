from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from schemas.line_item import LineItemSchema


class DiscrepancyExplainerChain:
    """ChatOpenAI chain producing plain-English explanations for discrepancies (FR-7)."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model = ChatOpenAI(model=model_name, temperature=0)

        self.prompt = PromptTemplate(
            template=(
                "You are an expert accounts-payable analyst. Explain the discrepancy "
                "between a matched purchase-order (PO) line item and an invoice line "
                "item in concise, plain English.\n\n"
                "PO LINE ITEM:\n"
                "  quantity: {po_quantity}\n"
                "  unit_price: {po_unit_price}\n"
                "  tax_rate: {po_tax_rate}\n"
                "  line_total: {po_line_total}\n\n"
                "INVOICE LINE ITEM:\n"
                "  quantity: {invoice_quantity}\n"
                "  unit_price: {invoice_unit_price}\n"
                "  tax_rate: {invoice_tax_rate}\n"
                "  line_total: {invoice_line_total}\n\n"
                "VARIANCES:\n"
                "  qty_variance: {qty_variance:.4f}\n"
                "  price_variance: {price_variance:.4f}\n"
                "  total_variance: {total_variance:.4f}\n\n"
                "Describe what differs, by how much, and the likely cause. Keep the "
                "explanation to 2-4 sentences."
            ),
            input_variables=[
                "po_quantity",
                "po_unit_price",
                "po_tax_rate",
                "po_line_total",
                "invoice_quantity",
                "invoice_unit_price",
                "invoice_tax_rate",
                "invoice_line_total",
                "qty_variance",
                "price_variance",
                "total_variance",
            ],
        )

        self.chain = self.prompt | self.model

    def explain(
        self,
        invoice_item: LineItemSchema,
        po_item: LineItemSchema,
        variances: Dict[str, float],
    ) -> str:
        """Return a plain-English explanation of the discrepancy."""
        response = self.chain.invoke(
            {
                "po_quantity": po_item.quantity,
                "po_unit_price": po_item.unit_price,
                "po_tax_rate": po_item.tax_rate,
                "po_line_total": po_item.line_total,
                "invoice_quantity": invoice_item.quantity,
                "invoice_unit_price": invoice_item.unit_price,
                "invoice_tax_rate": invoice_item.tax_rate,
                "invoice_line_total": invoice_item.line_total,
                "qty_variance": variances.get("qty_variance", 0.0),
                "price_variance": variances.get("price_variance", 0.0),
                "total_variance": variances.get("total_variance", 0.0),
            }
        )
        return response.content

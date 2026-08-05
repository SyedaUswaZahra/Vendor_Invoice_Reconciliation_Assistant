from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.gmail import GmailSendMessage

from schemas.line_item import LineItemSchema


class DraftActionChain:
    """Drafts a follow-up email to the supplier for high-risk discrepancies (FR-10)."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        self.model = ChatOpenAI(model=model_name, temperature=0)

        self.prompt = PromptTemplate(
            template=(
                "You are an accounts-payable professional drafting a follow-up email "
                "to a supplier regarding discrepancies found during invoice "
                "reconciliation.\n\n"
                "SUPPLIER EMAIL:\n{supplier_email}\n\n"
                "DISCREPANCIES:\n{discrepancies}\n\n"
                "Write a professional, concise follow-up email that:\n"
                "1. Addresses the supplier politely.\n"
                "2. Summarizes the discrepancies clearly.\n"
                "3. Requests clarification or corrected documentation.\n"
                "4. Provides a reasonable deadline for response.\n\n"
                "Return only the email body text (no subject line, no signature block)."
            ),
            input_variables=["supplier_email", "discrepancies"],
        )

        self.chain = self.prompt | self.model

    def draft_email(self, supplier_email: str, discrepancies: List[Dict]) -> str:
        """Return a draft follow-up email body for the supplier."""
        discrepancies_text = self._format_discrepancies(discrepancies)
        response = self.chain.invoke(
            {
                "supplier_email": supplier_email,
                "discrepancies": discrepancies_text,
            }
        )
        return response.content

    def send(self, email_body: str, to: str) -> None:
        """Send via Gmail tool; must be human-reviewed before invocation."""
        # IMPORTANT: This method must only be invoked after a human has reviewed
        # the drafted email body. Automated sending without review is prohibited.
        tool = GmailSendMessage(
            to=to,
            subject="Follow-up on Invoice Discrepancies",
            message=email_body,
        )
        tool.run("")

    @staticmethod
    def _format_discrepancies(discrepancies: List[Dict]) -> str:
        """Serialize discrepancy dicts into readable text for the prompt."""
        if not discrepancies:
            return "No discrepancies provided."
        lines = []
        for i, disc in enumerate(discrepancies, start=1):
            lines.append(
                f"{i}. {disc.get('description', 'N/A')} — "
                f"qty variance: {disc.get('qty_variance', 'N/A')}, "
                f"price variance: {disc.get('price_variance', 'N/A')}, "
                f"total variance: {disc.get('total_variance', 'N/A')}"
            )
        return "\n".join(lines)

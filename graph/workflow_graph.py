from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

from loaders.invoice_loader import InvoiceLoader
from chains.extraction_chain import LineItemExtractionChain
from matcher.matcher_node import MatcherNode
from matcher.embedding_store import EmbeddingStore
from reconcile.variance_engine import VarianceEngine
from chains.explainer_chain import DiscrepancyExplainerChain
from report.assembler import ReportAssembler
from chains.draft_action_chain import DraftActionChain


class ReconciliationState(TypedDict):
    file_path: str
    po_items: List
    raw_text: str
    invoice_items: List
    matches: List
    variances: List[Dict]
    explanations: List[str]
    report: str
    high_risk: bool
    draft_email: str


class WorkflowGraph:
    high_risk_threshold: float = 500.0

    def __init__(self) -> None:
        self.graph = None
        self.app = None

    def build(self) -> None:
        """Wire all nodes and conditional edges into the LangGraph state machine."""
        builder = StateGraph(ReconciliationState)

        # Node functions
        def ingest(state: ReconciliationState) -> dict:
            loader = InvoiceLoader(state["file_path"])
            raw_text = loader.load()
            return {"raw_text": raw_text}

        def preprocess(state: ReconciliationState) -> dict:
            chain = LineItemExtractionChain()
            invoice_items = chain.extract(state["raw_text"])
            return {"invoice_items": invoice_items}

        def match(state: ReconciliationState) -> dict:
            po_store = EmbeddingStore()
            po_store.index(state["po_items"])
            matcher = MatcherNode(po_store)
            matches = matcher.match(state["invoice_items"])
            return {"matches": matches}

        def reconcile(state: ReconciliationState) -> dict:
            engine = VarianceEngine()
            variances = engine.reconcile_batch(state["matches"])
            return {"variances": variances}

        def explain(state: ReconciliationState) -> dict:
            chain = DiscrepancyExplainerChain()
            explanations = []
            for variance in state["variances"]:
                explanation = chain.explain(
                    variance["invoice_item"],
                    variance["po_item"],
                    variance["variances"],
                )
                explanations.append(explanation)
            return {"explanations": explanations}

        def assemble(state: ReconciliationState) -> dict:
            assembler = ReportAssembler()
            rows = []
            for variance in state["variances"]:
                inv = variance["invoice_item"]
                po = variance["po_item"]
                variances = variance["variances"]
                rows.append(
                    {
                        "description": getattr(inv, "description", None)
                        or getattr(po, "description", ""),
                        "expected_qty": po.quantity,
                        "actual_qty": inv.quantity,
                        "expected_price": po.unit_price,
                        "actual_price": inv.unit_price,
                        "variance": variances.get("total_variance", 0.0),
                        "explanation": state["explanations"][
                            state["variances"].index(variance)
                        ],
                    }
                )
            report = assembler.assemble(rows)
            return {"report": report}

        def draft_action(state: ReconciliationState) -> dict:
            chain = DraftActionChain()
            discrepancies = []
            for variance in state["variances"]:
                inv = variance["invoice_item"]
                var = variance["variances"]
                discrepancies.append(
                    {
                        "description": getattr(inv, "description", ""),
                        "qty_variance": var.get("qty_variance", 0.0),
                        "price_variance": var.get("price_variance", 0.0),
                        "total_variance": var.get("total_variance", 0.0),
                    }
                )
            email = chain.draft_email(
                supplier_email="supplier@example.com",
                discrepancies=discrepancies,
            )
            return {"draft_email": email}

        def risk_router(state: ReconciliationState) -> str:
            total_variance = 0.0
            for variance in state["variances"]:
                total_variance += variance["variances"].get("total_variance", 0.0)
                inv = variance["invoice_item"]
                po = variance["po_item"]
                if inv.tax_rate != po.tax_rate:
                    return "draft_action"
            if total_variance > self.high_risk_threshold:
                return "draft_action"
            return END

        # Add nodes
        builder.add_node("ingest", ingest)
        builder.add_node("preprocess", preprocess)
        builder.add_node("match", match)
        builder.add_node("reconcile", reconcile)
        builder.add_node("explain", explain)
        builder.add_node("assemble", assemble)
        builder.add_node("draft_action", draft_action)

        # Wire edges
        builder.set_entry_point("ingest")
        builder.add_edge("ingest", "preprocess")
        builder.add_edge("preprocess", "match")
        builder.add_edge("match", "reconcile")
        builder.add_edge("reconcile", "explain")
        builder.add_edge("explain", "assemble")
        builder.add_conditional_edges(
            "assemble",
            risk_router,
            {
                "draft_action": "draft_action",
                END: END,
            },
        )
        builder.add_edge("draft_action", END)

        self.graph = builder
        self.app = builder.compile()

    def run(self, file_path: str, po_items: List) -> ReconciliationState:
        """Execute the workflow and return the final state."""
        if self.app is None:
            self.build()
        initial_state: ReconciliationState = {
            "file_path": file_path,
            "po_items": po_items,
            "raw_text": "",
            "invoice_items": [],
            "matches": [],
            "variances": [],
            "explanations": [],
            "report": "",
            "high_risk": False,
            "draft_email": "",
        }
        result = self.app.invoke(initial_state)
        return result

"""
FastAPI Backend — Criminal Network Analysis System
Exposes the NLP + graph analytics pipeline as REST endpoints
for a frontend dashboard to consume.

Run with: uvicorn src.api.main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Allow imports from src/nlp and src/graph
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

from nlp.entity_extractor import process_fir_file
from graph.build_graph import (
    build_graph_from_fir_data,
    build_graph_from_cdr,
    build_graph_from_transactions,
    compute_key_influencers,
    detect_communities,
    detect_suspicious_transaction_pattern,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

app = FastAPI(
    title="AI-Powered Criminal Network Analysis System",
    description="NCRB Problem Statement 26189 — Entity extraction, network graph analytics, and anomaly detection API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_full_graph() -> nx.MultiDiGraph:
    """Rebuild the graph fresh from source data files (kept simple for prototype;
    a production version would cache this / update incrementally)."""
    G = nx.MultiDiGraph()
    build_graph_from_fir_data(G, DATA_DIR / "extracted_entities.json")
    build_graph_from_cdr(G, DATA_DIR / "sample_cdr.csv")
    build_graph_from_transactions(G, DATA_DIR / "sample_transactions.csv")
    return G


@app.get("/")
def root():
    return {
        "system": "AI-Powered Criminal Network Analysis System",
        "problem_statement_id": 26189,
        "endpoints": ["/entities", "/graph", "/influencers", "/communities", "/alerts"],
    }


@app.get("/entities")
def get_entities():
    """Return entities + relationships extracted from raw FIR text via NLP."""
    return process_fir_file(str(DATA_DIR / "sample_fir_reports.txt"))


@app.get("/graph")
def get_graph():
    """Return the full network graph as nodes + edges, for frontend visualization."""
    G = _build_full_graph()
    nodes = [{"id": n, "type": data.get("type", "Unknown")} for n, data in G.nodes(data=True)]
    edges = [
        {"source": u, "target": v, "relation": data.get("relation", ""), "source_type": data.get("source_type", "")}
        for u, v, data in G.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


@app.get("/influencers")
def get_influencers(top_n: int = 5):
    """Return the top individuals ranked by centrality — the 'key players' in the network."""
    G = _build_full_graph()
    return compute_key_influencers(G, top_n=top_n)


@app.get("/communities")
def get_communities():
    """Return detected sub-groups/cells within the network."""
    G = _build_full_graph()
    return detect_communities(G)


@app.get("/alerts")
def get_alerts():
    """Return suspicious activity flags (e.g. transaction structuring patterns)."""
    return detect_suspicious_transaction_pattern(DATA_DIR / "sample_transactions.csv")

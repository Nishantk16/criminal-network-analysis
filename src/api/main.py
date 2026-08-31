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


@app.get("/audit-chain")
def get_audit_chain():
    """Return the tamper-proof evidence audit chain for the dashboard to visualize."""
    import json
    chain_file = DATA_DIR / "audit_chain.json"
    if not chain_file.exists():
        return []
    with open(chain_file) as f:
        return json.load(f)


@app.get("/search")
def search_entities(q: str):
    """Search for entities (persons, locations, vehicles) by partial name match."""
    G = _build_full_graph()
    q_lower = q.lower().strip()
    if not q_lower:
        return []
    matches = [
        {"id": n, "type": data.get("type", "Unknown")}
        for n, data in G.nodes(data=True)
        if q_lower in n.lower()
    ]
    return matches[:10]


@app.get("/entity/{name}")
def get_entity_detail(name: str):
    """
    Return a full case-file view for a single entity: its type, direct
    connections, every FIR/CDR/transaction record it appears in, and its
    centrality ranking if it's a person. This powers the dashboard's
    click-to-investigate detail panel.
    """
    import csv

    G = _build_full_graph()
    if name not in G.nodes:
        return {"error": f"No entity named '{name}' found in the network."}

    entity_type = G.nodes[name].get("type", "Unknown")

    # Direct connections (both directions), deduplicated
    connections = []
    seen = set()
    for u, v, data in G.edges(data=True):
        if u == name and v not in seen:
            connections.append({"name": v, "relation": data.get("relation", ""), "direction": "outgoing"})
            seen.add(v)
        elif v == name and u not in seen:
            connections.append({"name": u, "relation": data.get("relation", ""), "direction": "incoming"})
            seen.add(u)

    # FIR mentions
    fir_mentions = []
    for report in process_fir_file(str(DATA_DIR / "sample_fir_reports.txt")):
        if name in report["entities"]["persons"] or name in report["entities"]["locations"]:
            fir_mentions.append(report["raw_text"])

    # CDR records involving this entity
    call_records = []
    with open(DATA_DIR / "sample_cdr.csv") as f:
        for row in csv.DictReader(f):
            if row["caller_name"] == name or row["receiver_name"] == name:
                call_records.append({
                    "with": row["receiver_name"] if row["caller_name"] == name else row["caller_name"],
                    "timestamp": row["timestamp"],
                    "duration_sec": row["duration_sec"],
                    "location": row["tower_location"],
                    "direction": "outgoing" if row["caller_name"] == name else "incoming",
                })

    # Financial transactions involving this entity
    transactions = []
    with open(DATA_DIR / "sample_transactions.csv") as f:
        for row in csv.DictReader(f):
            if row["sender_name"] == name or row["receiver_name"] == name:
                transactions.append({
                    "with": row["receiver_name"] if row["sender_name"] == name else row["sender_name"],
                    "amount": row["amount"],
                    "timestamp": row["timestamp"],
                    "mode": row["mode"],
                    "direction": "sent" if row["sender_name"] == name else "received",
                })

    # Centrality ranking, if this is a person
    rank_info = None
    if entity_type == "Person":
        influencers = compute_key_influencers(G, top_n=100)
        for i, person in enumerate(influencers, 1):
            if person["name"] == name:
                rank_info = {"rank": i, **{k: v for k, v in person.items() if k != "name"}}
                break

    return {
        "name": name,
        "type": entity_type,
        "rank": rank_info,
        "connections": connections,
        "fir_mentions": fir_mentions,
        "call_records": call_records,
        "transactions": transactions,
    }

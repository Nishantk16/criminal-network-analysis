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

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
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
from api.auth import verify_credentials, create_token, verify_token
from api.report import generate_case_report

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


class LoginRequest(BaseModel):
    username: str
    password: str


def require_auth(authorization: str = Header(None)) -> dict:
    """
    Dependency that protects every case-data endpoint. Requires a valid,
    unexpired bearer token issued by /login. Without this, anyone who can
    reach the API could pull sensitive case data with no accountability —
    a real investigations system must know WHO is accessing WHAT.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header. Log in via /login first.")
    token = authorization.removeprefix("Bearer ").strip()
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")
    return payload


@app.post("/login")
def login(body: LoginRequest):
    """Authenticate an investigator and issue a signed, time-limited access token."""
    user = verify_credentials(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_token(user["username"], user["role"])

    # Log every successful login to the tamper-proof audit chain — investigations
    # systems need an accountable record of who accessed the system and when.
    try:
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from blockchain.audit_log import add_evidence_block
        add_evidence_block("USER_LOGIN", {"username": user["username"], "role": user["role"]})
    except Exception:
        pass  # audit logging failure should never block a legitimate login

    return {"token": token, "username": user["username"], "role": user["role"], "full_name": user["full_name"]}


@app.get("/")
def root():
    return {
        "system": "AI-Powered Criminal Network Analysis System",
        "problem_statement_id": 26189,
        "endpoints": ["/entities", "/graph", "/influencers", "/communities", "/alerts"],
    }


@app.get("/entities")
def get_entities(user: dict = Depends(require_auth)):
    """Return entities + relationships extracted from raw FIR text via NLP."""
    return process_fir_file(str(DATA_DIR / "sample_fir_reports.txt"))


@app.get("/graph")
def get_graph(user: dict = Depends(require_auth)):
    """Return the full network graph as nodes + edges, for frontend visualization."""
    G = _build_full_graph()
    nodes = [{"id": n, "type": data.get("type", "Unknown")} for n, data in G.nodes(data=True)]
    edges = [
        {"source": u, "target": v, "relation": data.get("relation", ""), "source_type": data.get("source_type", "")}
        for u, v, data in G.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


@app.get("/influencers")
def get_influencers(top_n: int = 5, user: dict = Depends(require_auth)):
    """Return the top individuals ranked by centrality — the 'key players' in the network."""
    G = _build_full_graph()
    return compute_key_influencers(G, top_n=top_n)


@app.get("/communities")
def get_communities(user: dict = Depends(require_auth)):
    """Return detected sub-groups/cells within the network."""
    G = _build_full_graph()
    return detect_communities(G)


@app.get("/alerts")
def get_alerts(user: dict = Depends(require_auth)):
    """Return suspicious activity flags (e.g. transaction structuring patterns)."""
    return detect_suspicious_transaction_pattern(DATA_DIR / "sample_transactions.csv")


@app.get("/audit-chain")
def get_audit_chain(user: dict = Depends(require_auth)):
    """Return the tamper-proof evidence audit chain for the dashboard to visualize."""
    import json
    chain_file = DATA_DIR / "audit_chain.json"
    if not chain_file.exists():
        return []
    with open(chain_file) as f:
        return json.load(f)


@app.get("/search")
def search_entities(q: str, user: dict = Depends(require_auth)):
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
def get_entity_detail(name: str, user: dict = Depends(require_auth)):
    """
    Return a full case-file view for a single entity: its type, direct
    connections, every FIR/CDR/transaction record it appears in, and its
    centrality ranking if it's a person. This powers the dashboard's
    click-to-investigate detail panel.
    """
    return _build_entity_detail(name)


def _build_entity_detail(name: str) -> dict:
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


@app.get("/report/{name}")
def download_case_report(name: str, user: dict = Depends(require_auth)):
    """
    Generate and return a court-ready PDF case report for one entity,
    including a chain-of-custody verification block tied to the tamper-proof
    evidence audit chain. Also logs the report's generation to that same
    chain, so there is an accountable record of who exported what evidence
    and when.
    """
    import json

    detail = _build_entity_detail(name)
    if "error" in detail:
        raise HTTPException(status_code=404, detail=detail["error"])

    chain_file = DATA_DIR / "audit_chain.json"
    audit_chain = json.load(open(chain_file)) if chain_file.exists() else []

    pdf_bytes = generate_case_report(detail, generated_by=user["sub"], audit_chain=audit_chain)

    try:
        from blockchain.audit_log import add_evidence_block
        add_evidence_block("REPORT_GENERATED", {"entity": name, "generated_by": user["sub"]})
    except Exception:
        pass

    safe_filename = name.replace(" ", "_")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="case_report_{safe_filename}.pdf"'},
    )

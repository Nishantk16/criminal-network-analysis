"""
Graph Construction & Analytics Module
Builds a criminal network graph from:
  - NLP-extracted entities/relationships (FIR reports)
  - CDR (call records)
  - Financial transaction records
Then runs analytics: centrality (key influencers), community detection (sub-groups).
"""

import json
import csv
import networkx as nx
from pathlib import Path

try:
    import community as community_louvain  # python-louvain package
except ImportError:
    community_louvain = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def build_graph_from_fir_data(graph: nx.MultiDiGraph, fir_json_path: Path):
    """Add nodes/edges extracted from FIR text (NLP output)."""
    with open(fir_json_path) as f:
        fir_data = json.load(f)

    for report in fir_data:
        # Clean up known NER noise (apostrophe-s, stray non-person tokens)
        persons = [p.replace("'s", "").strip() for p in report["entities"]["persons"]]
        for p in persons:
            graph.add_node(p, type="Person")

        for loc in report["entities"]["locations"]:
            graph.add_node(loc, type="Location")

        for veh in report["entities"]["vehicle_numbers"]:
            graph.add_node(veh, type="Vehicle")

        for rel in report["relationships"]:
            src = rel["source"].replace("'s", "").strip()
            tgt = rel["target"].replace("'s", "").strip()
            graph.add_edge(src, tgt, relation=rel["type"], source_type="FIR")


def build_graph_from_cdr(graph: nx.MultiDiGraph, cdr_csv_path: Path):
    """Add CALLED edges from Call Detail Records."""
    with open(cdr_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            caller = row["caller_name"]
            receiver = row["receiver_name"]
            graph.add_node(caller, type="Person")
            graph.add_node(receiver, type="Person")
            graph.add_edge(caller, receiver, relation="CALLED",
                            source_type="CDR", duration=row["duration_sec"],
                            location=row["tower_location"])


def build_graph_from_transactions(graph: nx.MultiDiGraph, txn_csv_path: Path):
    """Add TRANSFERRED_MONEY edges from financial transaction records."""
    with open(txn_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sender = row["sender_name"]
            receiver = row["receiver_name"]
            graph.add_node(sender, type="Person")
            graph.add_node(receiver, type="Person")
            graph.add_edge(sender, receiver, relation="TRANSFERRED_MONEY",
                            source_type="Transaction", amount=row["amount"])


def compute_key_influencers(graph: nx.MultiDiGraph, top_n=5):
    """Rank individuals by centrality measures to find key influencers/kingpins."""
    simple_graph = nx.DiGraph(graph)  # collapse multi-edges for centrality calc

    degree_centrality = nx.degree_centrality(simple_graph)
    betweenness_centrality = nx.betweenness_centrality(simple_graph)
    try:
        pagerank = nx.pagerank(simple_graph)
    except Exception:
        pagerank = {n: 0 for n in simple_graph.nodes}

    scores = []
    for node in simple_graph.nodes:
        if simple_graph.nodes[node].get("type") != "Person":
            continue
        scores.append({
            "name": node,
            "degree_centrality": round(degree_centrality.get(node, 0), 3),
            "betweenness_centrality": round(betweenness_centrality.get(node, 0), 3),
            "pagerank": round(pagerank.get(node, 0), 3),
        })

    scores.sort(key=lambda x: x["pagerank"], reverse=True)
    return scores[:top_n]


def detect_communities(graph: nx.MultiDiGraph):
    """Detect sub-groups/cells within the network using Louvain community detection."""
    undirected = nx.Graph(graph)
    if community_louvain is None:
        return {}
    partition = community_louvain.best_partition(undirected)

    groups = {}
    for node, comm_id in partition.items():
        groups.setdefault(comm_id, []).append(node)
    return groups


def detect_suspicious_transaction_pattern(txn_csv_path: Path):
    """
    Simple rule-based structuring detection:
    Flags repeated same-amount transfers between the same pair within a short window
    (classic 'smurfing' pattern to dodge reporting thresholds).
    """
    from collections import defaultdict
    pairs = defaultdict(list)

    with open(txn_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["sender_name"], row["receiver_name"], row["amount"])
            pairs[key].append(row["timestamp"])

    flags = []
    for (sender, receiver, amount), timestamps in pairs.items():
        if len(timestamps) >= 3:
            flags.append({
                "sender": sender,
                "receiver": receiver,
                "amount": amount,
                "repeat_count": len(timestamps),
                "flag": "Possible structuring / smurfing pattern"
            })
    return flags


def export_for_visualization(graph: nx.MultiDiGraph, out_path: Path):
    """Export graph as JSON (nodes+edges) for frontend visualization (Cytoscape.js/D3)."""
    nodes = [{"id": n, "type": data.get("type", "Unknown")} for n, data in graph.nodes(data=True)]
    edges = [{"source": u, "target": v, "relation": data.get("relation", ""),
              "source_type": data.get("source_type", "")} for u, v, data in graph.edges(data=True)]
    with open(out_path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)


if __name__ == "__main__":
    G = nx.MultiDiGraph()

    build_graph_from_fir_data(G, DATA_DIR / "extracted_entities.json")
    build_graph_from_cdr(G, DATA_DIR / "sample_cdr.csv")
    build_graph_from_transactions(G, DATA_DIR / "sample_transactions.csv")

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print("=== TOP KEY INFLUENCERS (by PageRank) ===")
    influencers = compute_key_influencers(G)
    for i, person in enumerate(influencers, 1):
        print(f"{i}. {person['name']} — PageRank: {person['pagerank']}, "
              f"Betweenness: {person['betweenness_centrality']}, "
              f"Degree: {person['degree_centrality']}")

    print("\n=== DETECTED SUB-GROUPS (Community Detection) ===")
    communities = detect_communities(G)
    for comm_id, members in communities.items():
        print(f"Group {comm_id}: {members}")

    print("\n=== SUSPICIOUS TRANSACTION PATTERNS ===")
    flags = detect_suspicious_transaction_pattern(DATA_DIR / "sample_transactions.csv")
    for flag in flags:
        print(flag)

    export_for_visualization(G, DATA_DIR / "graph_for_viz.json")
    print(f"\n✅ Graph data exported to {DATA_DIR / 'graph_for_viz.json'}")

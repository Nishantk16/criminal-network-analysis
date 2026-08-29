# AI-Powered Criminal Network Analysis System

**Problem Statement ID:** 26189
**Organization:** Ministry of Home Affairs — National Crime Records Bureau (NCRB), Women Safety Division
**Theme:** Blockchain & Cybersecurity

## Problem

Investigators work with fragmented, unstructured data — FIRs, Call Detail Records (CDRs), financial
transactions, surveillance reports, social media intel — spread across multiple systems. Manually
connecting the dots between suspects, locations, phones, and money flows is slow and error-prone.

## Solution

An AI pipeline that automatically:
1. **Extracts entities** (people, phone numbers, vehicles, locations, amounts) from raw FIR text using NLP (spaCy NER + domain-specific pattern matching)
2. **Builds a relationship graph** connecting entities across FIRs, CDRs, and financial records
3. **Identifies key influencers** in the network using graph centrality (PageRank, betweenness, degree)
4. **Detects criminal sub-groups/cells** using community detection (Louvain algorithm)
5. **Flags suspicious patterns** such as transaction structuring ("smurfing")
6. **Logs every finding to a tamper-proof audit chain** — so the chain of evidence can't be silently altered, and any tampering is instantly detectable

## Architecture

```
Raw Data (FIR text, CDR, Transactions)
        │
        ▼
┌───────────────────┐
│   NLP Layer        │  spaCy NER + regex → entities & relationships
│ (src/nlp/)          │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  Graph Layer        │  NetworkX → key influencers, communities, anomalies
│ (src/graph/)         │
└─────────┬──────────┘
          ▼
┌───────────────────┐        ┌──────────────────────────────┐
│  API Layer           │◄────►│  Tamper-Proof Audit Log         │
│ (src/api/) FastAPI    │      │ (src/blockchain/) + Soroban       │
└─────────┬──────────┘        │  smart contract (contracts/)      │
          ▼                    └──────────────────────────────┘
   Investigator Dashboard
   (graph visualization, alerts)
```

## Repository Structure

```
data/                     Sample FIR text, CDR, transaction records (dummy data for demo)
src/nlp/                  Entity extraction (spaCy NER + regex patterns)
src/graph/                Graph construction + analytics (NetworkX)
src/api/                  FastAPI backend exposing the pipeline as REST endpoints
src/blockchain/           Local tamper-proof audit chain (hash-chain simulation)
contracts/audit_log/      Soroban smart contract — on-chain version of the audit log,
                           for deployment to Stellar testnet
```

## Running the Prototype

```bash
pip install -r requirements.txt   # spacy, networkx, matplotlib, pyvis, fastapi, uvicorn, python-louvain
python -m spacy download en_core_web_sm

# Run NLP extraction
python src/nlp/entity_extractor.py

# Build graph + run analytics
python src/graph/build_graph.py

# Generate visualizations
python src/graph/visualize.py

# Run the tamper-proof audit log demo
python src/blockchain/audit_log.py

# Start the API server
uvicorn src.api.main:app --reload --port 8000
```

### API Endpoints

| Endpoint | Description |
|---|---|
| `GET /entities` | Entities + relationships extracted from FIR text |
| `GET /graph` | Full network graph (nodes + edges) |
| `GET /influencers` | Top individuals ranked by centrality (key players) |
| `GET /communities` | Detected criminal sub-groups/cells |
| `GET /alerts` | Suspicious activity flags |

## Why Blockchain?

Court proceedings require a verifiable **chain of custody** for evidence. If an AI system silently
flags someone as a "key influencer" or an "alert," investigators and courts need proof that this
finding — and the data behind it — hasn't been altered after the fact. Logging every extraction,
inference, and alert to an append-only, cryptographically-linked chain (with the on-chain Soroban
version providing a public, decentralized guarantee) gives that proof.

## Tech Stack

- **NLP:** spaCy (NER), regex pattern matching
- **Graph Analytics:** NetworkX, python-louvain
- **Backend:** FastAPI
- **Visualization:** Matplotlib (static), Pyvis (interactive)
- **Blockchain:** Local hash-chain (prototype) + Soroban/Stellar smart contract (production path)

## Status

This is a Round 1 prototype demonstrating the core pipeline end-to-end on sample data. Not yet
implemented: multilingual FIR support (Hindi/regional languages), OCR for scanned reports,
production-grade relation extraction (currently rule-based), and live Soroban testnet deployment.

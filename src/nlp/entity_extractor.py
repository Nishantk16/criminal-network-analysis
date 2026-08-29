"""
Entity Extraction Module
Extracts persons, locations, organizations (via spaCy NER)
and phone numbers, vehicle numbers, amounts (via regex patterns)
from unstructured crime report text.
"""

import re
import json
import spacy

nlp = spacy.load("en_core_web_sm")

# --- Regex patterns for domain-specific entities spaCy won't catch out of the box ---
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
VEHICLE_PATTERN = re.compile(r"\b[A-Z]{2}-\d{2}-[A-Z]{2}-\d{4}\b")
AMOUNT_PATTERN = re.compile(r"Rs\.?\s?[\d,]+")
DATE_PATTERN = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")


def extract_entities(text: str) -> dict:
    """Run spaCy NER + regex extraction on a single block of text."""
    doc = nlp(text)

    entities = {
        "persons": sorted(set(ent.text for ent in doc.ents if ent.label_ == "PERSON")),
        "locations": sorted(set(ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC"))),
        "organizations": sorted(set(ent.text for ent in doc.ents if ent.label_ == "ORG")),
        "phone_numbers": sorted(set(PHONE_PATTERN.findall(text))),
        "vehicle_numbers": sorted(set(VEHICLE_PATTERN.findall(text))),
        "amounts": sorted(set(AMOUNT_PATTERN.findall(text))),
        "dates": sorted(set(DATE_PATTERN.findall(text))),
    }
    return entities


def extract_relationships(text: str, entities: dict) -> list:
    """
    Very lightweight rule-based relationship extraction.
    Looks for co-occurrence of two persons in the same sentence,
    and tags the relationship type using keyword cues.
    This is a starting point — round-1 prototype level, not production NLP.
    """
    relationships = []
    doc = nlp(text)

    keyword_map = {
        "call": "CALLED",
        "contact": "CONTACTED",
        "transfer": "TRANSFERRED_MONEY",
        "transaction": "TRANSFERRED_MONEY",
        "met": "MET_WITH",
        "meeting": "MET_WITH",
        "associate": "ASSOCIATED_WITH",
        "financier": "FINANCIAL_LINK",
        "coordinator": "COORDINATES_WITH",
    }

    for sent in doc.sents:
        sent_text = sent.text
        persons_in_sent = [p for p in entities["persons"] if p in sent_text]
        if len(persons_in_sent) >= 2:
            rel_type = "ASSOCIATED_WITH"  # default
            for kw, label in keyword_map.items():
                if kw in sent_text.lower():
                    rel_type = label
                    break
            for i in range(len(persons_in_sent)):
                for j in range(i + 1, len(persons_in_sent)):
                    relationships.append({
                        "source": persons_in_sent[i],
                        "target": persons_in_sent[j],
                        "type": rel_type,
                        "evidence": sent_text.strip()
                    })
    return relationships


def process_fir_file(filepath: str) -> list:
    """Process a text file containing multiple FIR reports separated by blank lines."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    reports = [r.strip() for r in content.split("\n\n") if r.strip()]
    results = []

    for report in reports:
        entities = extract_entities(report)
        relationships = extract_relationships(report, entities)
        results.append({
            "raw_text": report[:80] + "...",
            "entities": entities,
            "relationships": relationships
        })

    return results


if __name__ == "__main__":
    output = process_fir_file("/home/claude/criminal-network-analysis/data/sample_fir_reports.txt")

    print(json.dumps(output, indent=2))

    with open("/home/claude/criminal-network-analysis/data/extracted_entities.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Extraction complete. Saved to data/extracted_entities.json")

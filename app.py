# app.py
# Governance-first anonymization demo (Internal Use)

from __future__ import annotations
import json
import re
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
import regex as uregex
import streamlit as st


# ===============================
# 1. HIGH-CONFIDENCE PII RULES
# ===============================

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+30\s?)?(?:2\d{9}|69\d{8})(?!\d)")
AFM_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")
AMKA_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
ID_RE = re.compile(r"\b(?:ΑΔΤ|Α\.Δ\.Τ\.|ΔΑΤ|Αρ\.?\s*Ταυτ\.?)\s*[:\-]?\s*[A-ZΑ-Ω]{1,3}\s?\d{5,8}\b", re.IGNORECASE)

ADDRESS_RE = re.compile(
    r"\b(?:Οδός|Οδ\.|Λεωφόρος|Λεωφ\.|Λ\.|Πλατεία|Πλ\.)\s+[Α-Ωα-ωάέήίόύώϊϋΐΰ\.\- ]+\s+\d{1,4}\b"
)

DATE_RE = re.compile(
    r"\b(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})\b"
)

# Low-confidence name heuristic
GREEK_NAME_HEURISTIC = uregex.compile(
    r"\b[Α-ΩΪΫΆΈΉΊΌΎΏ][α-ωάέήίόύώϊϋΐΰ]{2,}\s+[Α-ΩΪΫΆΈΉΊΌΎΏ][α-ωάέήίόύώϊϋΐΰ]{2,}\b"
)

# Institution blacklist
INSTITUTION_BLACKLIST = [
    "ΠΡΩΤΟΔΙΚΕΙΟ", "ΔΙΚΑΣΤΗΡΙΟ", "ΕΦΕΤΕΙΟ", "ΑΡΕΙΟΣ", "ΠΑΓΟΣ",
    "ΝΟΣΟΚΟΜΕΙΟ", "ΓΕΝΙΚΟ", "ΥΠΟΥΡΓΕΙΟ",
    "Α.Ε.", "Ο.Ε.", "ΙΚΕ",
    "ΛΕΩΦΟΡΟΣ", "ΟΔΟΣ", "ΠΛΑΤΕΙΑ"
]


def is_institution_like(text: str) -> bool:
    return any(k in text.upper() for k in INSTITUTION_BLACKLIST)


# ===============================
# 2. ROLE ANCHORS
# ===============================

ROLE_PATTERNS = [
    ("ENAGON", re.compile(r"Ενάγων\s*:\s*([^\n,]+)", re.IGNORECASE)),
    ("ENAGOMENI", re.compile(r"Εναγομένη\s*:\s*([^\n,]+)", re.IGNORECASE)),
    ("DIKIGOROS", re.compile(r"Δικηγόρος[^:]*:\s*([^\n,]+)", re.IGNORECASE)),
    ("DIKASTIS", re.compile(r"Ο\s+ΔΙΚΑΣΤΗΣ\s*\n\s*([^\n]+)", re.IGNORECASE)),
    ("MARTYRAS", re.compile(r"μάρτυρας[^,\n]*\s+ο\s+([^\n,]+)", re.IGNORECASE)),
]


# ===============================
# 3. DATA STRUCTURE
# ===============================

@dataclass
class Finding:
    id: str
    start: int
    end: int
    text: str
    category: str
    confidence: float
    role: str
    proposed: str
    status: str = "ACCEPT"
    edited: str = ""


def make_id(val: str) -> str:
    return hashlib.sha256(val.encode()).hexdigest()[:12]


# ===============================
# 4. DETECTION ENGINE
# ===============================

def detect(text: str) -> List[Finding]:
    findings = []
    memory = {}
    counters = {}

    def assign_token(key, template):
        if key not in memory:
            counters[key] = counters.get(key, 0) + 1
            memory[key] = template.format(n=counters[key])
        return memory[key]

    # --- High confidence PII ---
    for m in EMAIL_RE.finditer(text):
        findings.append(Finding(make_id(m.group()), m.start(), m.end(), m.group(), "EMAIL", 0.98, "", assign_token(("EMAIL", m.group()), "[EMAIL_{n}]")))

    for m in PHONE_RE.finditer(text):
        findings.append(Finding(make_id(m.group()), m.start(), m.end(), m.group(), "PHONE", 0.95, "", assign_token(("PHONE", m.group()), "[ΤΗΛΕΦΩΝΟ_{n}]")))

    for m in AFM_RE.finditer(text):
        findings.append(Finding(make_id(m.group()), m.start(), m.end(), m.group(), "AFM", 0.90, "", "[ID_AFM]"))

    for m in AMKA_RE.finditer(text):
        findings.append(Finding(make_id(m.group()), m.start(), m.end(), m.group(), "AMKA", 0.92, "", "[ID_AMKA]"))

    for m in ID_RE.finditer(text):
        findings.append(Finding(make_id(m.group()), m.start(), m.end(), m.group(), "ID", 0.90, "", "[ID_ΤΑΥΤΟΤΗΤΑΣ]"))

    for m in ADDRESS_RE.finditer(text):
        findings.append(Finding(make_id(m.group()), m.start(), m.end(), m.group(), "ADDRESS", 0.85, "", assign_token(("ADDRESS", m.group()), "[ΔΙΕΥΘΥΝΣΗ_{n}]")))

    # --- Role-based detection ---
    for role, pattern in ROLE_PATTERNS:
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            if is_institution_like(name):
                continue
            start, end = m.start(1), m.end(1)
            token = assign_token((role, name), f"[{role}_{{n}}]")
            findings.append(Finding(make_id(name), start, end, name, "PERSON", 0.80, role, token))

    # --- Heuristic PERSON (body only) ---
    body_start = text.find("Διάδικοι")
    if body_start == -1:
        body_start = 0

    for m in GREEK_NAME_HEURISTIC.finditer(text):
        if m.start() < body_start:
            continue
        span = m.group()
        if is_institution_like(span):
            continue
        left = text[max(0, m.start()-10):m.start()]
        if "Στην" in left:
            continue
        token = assign_token(("PERSON", span), "[ΠΡΟΣΩΠΟ_{n}]")
        findings.append(Finding(make_id(span), m.start(), m.end(), span, "PERSON", 0.55, "", token))

    # Remove overlaps
    findings.sort(key=lambda x: (x.start, -x.confidence))
    final = []
    last_end = -1
    for f in findings:
        if f.start < last_end:
            continue
        final.append(f)
        last_end = f.end

    return final


# ===============================
# 5. APPLY REDACTIONS
# ===============================

def apply_redactions(text: str, findings: List[Finding]):
    applied = []
    audit = []

    for f in findings:
        if f.status == "REJECT":
            continue
        replacement = f.edited.strip() if f.status == "EDIT" else f.proposed
        applied.append((f.start, f.end, f.text, replacement))
        audit.append({"text": f.text, "replacement": replacement, "confidence": f.confidence})

    applied.sort(reverse=True)
    out = text
    for start, end, original, rep in applied:
        out = out[:start] + rep + out[end:]

    return out, audit


# ===============================
# 6. LEAKAGE CHECK
# ===============================

def leakage_check(text):
    issues = []
    if EMAIL_RE.search(text):
        issues.append("Email detected")
    if PHONE_RE.search(text):
        issues.append("Phone detected")
    if AFM_RE.search(text):
        issues.append("AFM detected")
    return issues


# ===============================
# 7. STREAMLIT UI
# ===============================

st.set_page_config(layout="wide")
st.title("Ευφυής Ανωνυμοποίηση – Demo (Εσωτερική Αξιοποίηση)")

uploaded = st.file_uploader("Upload .txt")

if "text" not in st.session_state:
    st.session_state.text = ""

if uploaded:
    st.session_state.text = uploaded.read().decode("utf-8")

text = st.text_area("Κείμενο", st.session_state.text, height=300)

if st.button("Run Detection"):
    st.session_state.findings = detect(text)

if "findings" in st.session_state:
    findings = st.session_state.findings

    total = len(findings)
    hard = sum(1 for f in findings if f.category in ["EMAIL", "PHONE", "AFM", "AMKA", "ID", "ADDRESS"])

    col1, col2 = st.columns(2)
    col1.metric("Detected items", total)
    col2.metric("Hard PII", hard)

    df = pd.DataFrame([{
        "text": f.text,
        "type": f.category,
        "confidence": f.confidence,
        "proposed": f.proposed,
        "status": f.status,
        "edited": f.edited
    } for f in findings])

    edited = st.data_editor(df, use_container_width=True)

    for i, row in edited.iterrows():
        findings[i].status = row["status"]
        findings[i].edited = row["edited"]

    anonymized, audit = apply_redactions(text, findings)

    st.subheader("Anonymized Output")
    st.text_area("", anonymized, height=300)

    issues = leakage_check(anonymized)
    if issues:
        st.error(f"Leakage issues detected: {issues}")
    else:
        st.success("No high-confidence leakage detected.")

    st.download_button("Download anonymized.txt", anonymized.encode(), "anonymized.txt")
    st.download_button("Download audit.json", json.dumps(audit, indent=2, ensure_ascii=False).encode(), "audit.json")

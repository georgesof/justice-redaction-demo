# app.py
# Streamlit UI demo for anonymization/pseudonymization with HITL review.

from __future__ import annotations
import json
import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import pandas as pd
import regex as uregex
import streamlit as st


# -------------------------
# Detection rules (POC)
# -------------------------
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+30\s?)?(?:2\d{9}|69\d{8})(?!\d)")
AFM_RE = re.compile(r"(?<!\d)\d{9}(?!\d)")
AMKA_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
ID_RE = re.compile(r"\b(?:ΑΔΤ|Α\.Δ\.Τ\.|ΔΑΤ|Αρ\.?\s*Ταυτ\.?)\s*[:\-]?\s*[A-ZΑ-Ω]{1,3}\s?\d{5,8}\b", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{1,2}\s+(?:Ιανουαρίου|Φεβρουαρίου|Μαρτίου|Απριλίου|Μαΐου|Ιουνίου|Ιουλίου|Αυγούστου|Σεπτεμβρίου|Οκτωβρίου|Νοεμβρίου|Δεκεμβρίου)\s+\d{4})\b",
    re.IGNORECASE
)
ADDRESS_RE = re.compile(
    r"\b(?:Οδός|Οδ\.|Λεωφόρος|Λεωφ\.|Λ\.|Αγίου|Αγ\.|Πλατεία|Πλ\.)\s+[Α-ΩΪΫΆΈΉΊΌΎΏ][Α-Ωα-ωάέήίόύώϊϋΐΰ\.\- ]{2,40}\s+\d{1,4}\b"
)

# Very rough Greek person name heuristic (low confidence)
GREEK_NAME_HEURISTIC = uregex.compile(
    r"(?<!\[)\b"
    r"(?:[Α-ΩΪΫΆΈΉΊΌΎΏ][α-ωάέήίόύώϊϋΐΰ]{2,})"
    r"\s+"
    r"(?:[Α-ΩΪΫΆΈΉΊΌΎΏ][α-ωάέήίόύώϊϋΐΰ]{2,})"
    r"\b(?!\])"
)

POLICIES = {
    "open_data": {
        "mask_dates": "year_only",   # year_only / full_mask / keep
        "mask_addresses": True,
        "mask_persons": True,
        "hitl_threshold": 0.80,
    },
    "internal": {
        "mask_dates": "keep",
        "mask_addresses": True,
        "mask_persons": True,
        "hitl_threshold": 0.70,
    },
    "controlled": {
        "mask_dates": "keep",
        "mask_addresses": True,
        "mask_persons": True,
        "hitl_threshold": 0.65,
    },
}

TOKEN_TEMPLATES = {
    "EMAIL": "[EMAIL_{n}]",
    "PHONE": "[ΤΗΛΕΦΩΝΟ_{n}]",
    "AFM": "[ID_AFM]",
    "AMKA": "[ID_AMKA]",
    "ID": "[ID_ΤΑΥΤΟΤΗΤΑΣ]",
    "DATE": "[ΗΜΕΡΟΜΗΝΙΑ]",
    "DATE_YEAR": "[ΕΤΟΣ_{year}]",
    "ADDRESS": "[ΔΙΕΥΘΥΝΣΗ_{n}]",
    "PERSON": "[ΠΡΟΣΩΠΟ_{n}]",
}

@dataclass
class Finding:
    id: str
    start: int
    end: int
    text: str
    category: str
    confidence: float
    source: str
    proposed: str
    status: str = "ACCEPT"  # ACCEPT / REJECT / EDIT
    edited: str = ""        # if EDIT, use this


def _hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def detect_findings(text: str, policy_name: str) -> List[Finding]:
    policy = POLICIES[policy_name]
    findings = []

    def add_matches(pattern, category, confidence, source="rule"):
        for m in pattern.finditer(text):
            findings.append((m.start(), m.end(), m.group(0), category, confidence, source))

    add_matches(EMAIL_RE, "EMAIL", 0.98)
    add_matches(PHONE_RE, "PHONE", 0.95)
    add_matches(ID_RE, "ID", 0.90)
    add_matches(AMKA_RE, "AMKA", 0.92)
    add_matches(AFM_RE, "AFM", 0.88)
    add_matches(ADDRESS_RE, "ADDRESS", 0.80)

    if policy["mask_dates"] != "keep":
        add_matches(DATE_RE, "DATE", 0.75)

    if policy["mask_persons"]:
        for m in GREEK_NAME_HEURISTIC.finditer(text):
            findings.append((m.start(), m.end(), m.group(0), "PERSON", 0.55, "heuristic"))

    # resolve overlaps: keep earliest, then longer, then higher conf
    findings.sort(key=lambda x: (x[0], -(x[1]-x[0]), -x[4]))
    resolved = []
    last_end = -1
    for f in findings:
        if f[0] < last_end:
            continue
        resolved.append(f)
        last_end = f[1]

    # assign consistent tokens
    counters = {"EMAIL": 0, "PHONE": 0, "ADDRESS": 0, "PERSON": 0}
    memory: Dict[Tuple[str, str], str] = {}

    out: List[Finding] = []
    for start, end, val, cat, conf, src in resolved:
        proposed = ""

        if cat == "AFM":
            proposed = TOKEN_TEMPLATES["AFM"]
        elif cat == "AMKA":
            proposed = TOKEN_TEMPLATES["AMKA"]
        elif cat == "ID":
            proposed = TOKEN_TEMPLATES["ID"]
        elif cat == "DATE":
            mode = policy["mask_dates"]
            if mode == "year_only":
                years = re.findall(r"(19\d{2}|20\d{2})", val)
                proposed = TOKEN_TEMPLATES["DATE_YEAR"].format(year=years[-1]) if years else TOKEN_TEMPLATES["DATE"]
            else:
                proposed = TOKEN_TEMPLATES["DATE"]
        elif cat in ("EMAIL", "PHONE", "ADDRESS", "PERSON"):
            key = (cat, val)
            if key not in memory:
                counters[cat] += 1
                proposed = TOKEN_TEMPLATES[cat].format(n=counters[cat])
                memory[key] = proposed
            proposed = memory[key]
        else:
            continue

        fid = _hash_id(f"{start}-{end}-{cat}-{val}")
        out.append(Finding(
            id=fid, start=start, end=end, text=val, category=cat,
            confidence=conf, source=src, proposed=proposed
        ))

    return out


def apply_findings(text: str, findings: List[Finding]) -> Tuple[str, List[dict], List[dict]]:
    """
    Returns: anonymized_text, redaction_map, audit_events
    Uses accepted/edited findings; rejects ignored.
    """
    selected = []
    audit = []
    for f in findings:
        if f.status == "REJECT":
            audit.append({"ts": time.time(), "id": f.id, "action": "REJECT", "category": f.category, "text": f.text})
            continue
        if f.status == "EDIT":
            rep = f.edited.strip() or f.proposed
            audit.append({"ts": time.time(), "id": f.id, "action": "EDIT", "category": f.category, "text": f.text, "replacement": rep})
            selected.append((f.start, f.end, f.text, f.category, f.confidence, f.source, rep))
        else:
            audit.append({"ts": time.time(), "id": f.id, "action": "ACCEPT", "category": f.category, "text": f.text, "replacement": f.proposed})
            selected.append((f.start, f.end, f.text, f.category, f.confidence, f.source, f.proposed))

    # apply from end to start
    selected.sort(key=lambda x: x[0], reverse=True)

    out = text
    redactions = []
    for start, end, original, cat, conf, src, rep in selected:
        out = out[:start] + rep + out[end:]
        redactions.append({
            "start": start, "end": end,
            "original": original,
            "category": cat,
            "confidence": conf,
            "source": src,
            "replacement": rep
        })
    redactions.reverse()
    return out, redactions, audit


def render_highlighted_html(text: str, findings: List[Finding]) -> str:
    """
    Highlight accepted/edited findings in the ORIGINAL text, for visual explanation.
    """
    spans = []
    for f in findings:
        if f.status == "REJECT":
            continue
        rep = f.edited.strip() if f.status == "EDIT" else f.proposed
        spans.append((f.start, f.end, f.text, f.category, rep))

    spans.sort(key=lambda x: x[0], reverse=True)
    out = text
    for start, end, original, cat, rep in spans:
        safe_original = (original
                         .replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
        badge = f"{cat} → {rep}"
        chunk = f"""<mark style="padding:2px 4px;border-radius:6px;">
        {safe_original}
        <span style="font-size:11px;opacity:.7;margin-left:6px;">{badge}</span>
        </mark>"""
        out = out[:start] + chunk + out[end:]
    out = out.replace("\n", "<br/>")
    return f"<div style='font-family: ui-sans-serif, system-ui; line-height:1.6;'>{out}</div>"


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Justice Redaction POC", layout="wide")

st.title("Demo: Ευφυής Αυτοματοποιημένη Ανωνυμοποίηση / Ψευδωνυμοποίηση (HITL)")

with st.sidebar:
    st.header("Ρυθμίσεις")
    policy = st.selectbox("Policy", ["open_data", "internal", "controlled"], index=0)
    st.caption("open_data = πιο αυστηρό (π.χ. ημερομηνίες → έτος)")
    show_only_low = st.checkbox("Εμφάνιση μόνο low-confidence", value=False)
    st.divider()
    st.subheader("Εισαγωγή κειμένου")
    uploaded = st.file_uploader("Upload .txt", type=["txt"])
    st.caption("Το demo δουλεύει πάνω στο OCR/HTR text (όχι πάνω σε PDF).")

default_text = """Στο Μονομελές Πρωτοδικείο Αθηνών, την 12/03/2024, ο Ιωάννης Παπαδόπουλος κατέθεσε ότι
ο αντίδικος Μαρία Κωνσταντίνου (ΑΦΜ 123456789) κατοικεί Οδός Πατησίων 120.
Email: test.user@example.com, Τηλέφωνο: +30 6971234567.
ΑΔΤ: ΑΒ 123456.
"""

if "text" not in st.session_state:
    st.session_state.text = default_text

if uploaded is not None:
    st.session_state.text = uploaded.read().decode("utf-8", errors="replace")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Original (input)")
    st.text_area("Κείμενο", value=st.session_state.text, height=260, key="input_area")

    run_btn = st.button("Run detection", type="primary", use_container_width=True)

with colB:
    st.subheader("Τι θα παραχθεί")
    st.markdown("""
- **Anonymized text** (αναγνώσιμο, με συνεπείς tokens)
- **Redaction map (JSON)** (τι άλλαξε/πού/με confidence)
- **Audit events** (ACCEPT/EDIT/REJECT)
""")

# If user edited the input area, sync it back
st.session_state.text = st.session_state.input_area

if run_btn or "findings" not in st.session_state:
    st.session_state.findings = detect_findings(st.session_state.text, policy)

findings: List[Finding] = st.session_state.findings
threshold = POLICIES[policy]["hitl_threshold"]

# Build DataFrame for table editing
rows = []
for f in findings:
    is_low = f.confidence < threshold
    if show_only_low and not is_low:
        continue
    rows.append({
        "id": f.id,
        "category": f.category,
        "text": f.text,
        "confidence": round(f.confidence, 2),
        "source": f.source,
        "proposed": f.proposed,
        "status": f.status,
        "edited": f.edited
    })

df = pd.DataFrame(rows)

st.subheader("Findings & HITL decisions")
st.caption(f"Low-confidence threshold για policy '{policy}': {threshold:.2f}. "
           f"Κάτω από αυτό, συνήθως θες ανθρώπινη επιβεβαίωση.")

if df.empty:
    st.info("Δεν βρέθηκαν findings με τα τρέχοντα rules.")
else:
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "status": st.column_config.SelectboxColumn(
                "status",
                options=["ACCEPT", "REJECT", "EDIT"],
                required=True
            ),
            "edited": st.column_config.TextColumn("edited", help="Χρησιμοποιείται μόνο όταν status=EDIT"),
        }
    )

    # Apply user edits back to session findings
    by_id = {f.id: f for f in findings}
    for _, r in edited_df.iterrows():
        f = by_id.get(r["id"])
        if not f:
            continue
        f.status = str(r["status"])
        f.edited = str(r["edited"]) if pd.notna(r["edited"]) else ""

# Generate outputs
anonymized_text, redaction_map, audit = apply_findings(st.session_state.text, findings)

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Preview (highlighted decisions)")
    html = render_highlighted_html(st.session_state.text, findings)
    st.components.v1.html(html, height=340, scrolling=True)

with c2:
    st.subheader("Anonymized output")
    st.text_area("anonymized", value=anonymized_text, height=340)

st.subheader("Exports")
meta = {
    "policy": policy,
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "findings_count": len(redaction_map),
}
bundle = {
    "meta": meta,
    "anonymized_text": anonymized_text,
    "redaction_map": redaction_map,
    "audit_events": audit,
}

colx, coly, colz = st.columns(3)
with colx:
    st.download_button(
        "Download anonymized.txt",
        data=anonymized_text.encode("utf-8"),
        file_name="anonymized.txt",
        mime="text/plain",
        use_container_width=True
    )
with coly:
    st.download_button(
        "Download redaction_map.json",
        data=json.dumps({"meta": meta, "redactions": redaction_map}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="redaction_map.json",
        mime="application/json",
        use_container_width=True
    )
with colz:
    st.download_button(
        "Download bundle.json",
        data=json.dumps(bundle, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="bundle.json",
        mime="application/json",
        use_container_width=True
    )

st.divider()
st.caption("Σημαντικό: Αυτό είναι demo για τη ροή HITL + auditability. Για production θες legal-grade NER training, OCR quality gates, leakage testing και role-based policies.")

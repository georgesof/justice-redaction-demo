# app.py
# Streamlit UI demo (INTERNAL use) for anonymization/pseudonymization with governance-first UX.

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


# =========================
# 1) RULES (high-confidence PII)
# =========================
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
    r"\b(?:Οδός|Οδ\.|Λεωφόρος|Λεωφ\.|Λ\.|Αγίου|Αγ\.|Πλατεία|Πλ\.)\s+[Α-ΩΪΫΆΈΉΊΌΎΏ][Α-Ωα-ωάέήίόύώϊϋΐΰ\.\- ]{2,50}\s+\d{1,4}\b"
)

# Low-confidence name heuristic: 2 Greek capitalized tokens (kept, but hidden by default)
GREEK_NAME_HEURISTIC = uregex.compile(
    r"(?<!\[)\b"
    r"(?:[Α-ΩΪΫΆΈΉΊΌΎΏ][α-ωάέήίόύώϊϋΐΰ]{2,})"
    r"\s+"
    r"(?:[Α-ΩΪΫΆΈΉΊΌΎΏ][α-ωάέήίόύώϊϋΐΰ]{2,})"
    r"\b(?!\])"
)

# Institution/company/location keywords that should NOT be treated as PERSON
INSTITUTION_BLACKLIST = [
    "ΔΙΚΑΣΤΗΡΙΟ", "ΠΡΩΤΟΔΙΚΕΙΟ", "ΕΦΕΤΕΙΟ", "ΑΡΕΙΟΣ", "ΠΑΓΟΣ",
    "ΕΙΣΑΓΓΕΛΙΑ", "ΤΜΗΜΑ", "ΥΠΟΥΡΓΕΙΟ",
    "ΝΟΣΟΚΟΜΕΙΟ", "ΓΕΝΙΚΟ", "ΚΕΝΤΡΟ", "ΚΛΙΝΙΚΗ",
    "Α.Ε.", "ΑΕ", "Ο.Ε.", "ΟΕ", "Ε.Π.Ε.", "ΕΠΕ", "Ι.Κ.Ε.", "ΙΚΕ",
    "ΛΕΩΦΟΡΟΣ", "ΛΕΩΦ", "ΟΔΟΣ", "ΟΔ.", "ΠΛΑΤΕΙΑ", "ΠΛ."
]

def is_institution_like(span_text: str) -> bool:
    up = span_text.upper()
    return any(k in up for k in INSTITUTION_BLACKLIST)

# =========================
# 2) POLICIES (internal-first)
# =========================
POLICIES = {
    "internal": {
        "mask_dates": "keep",     # keep / full_mask / year_only
        "mask_addresses": True,
        "mask_persons": True,
        "hitl_threshold": 0.70,   # below -> needs review
        "default_view": "PII_ONLY"
    },
    "controlled": {
        "mask_dates": "keep",
        "mask_addresses": True,
        "mask_persons": True,
        "hitl_threshold": 0.65,
        "default_view": "PII_ONLY"
    },
    "open_data": {
        "mask_dates": "year_only",
        "mask_addresses": True,
        "mask_persons": True,
        "hitl_threshold": 0.80,
        "default_view": "PII_ONLY"
    }
}

# Token templates (PII)
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
    # Role-aware tokens
    "ENAGON": "[ΕΝΑΓΩΝ_{n}]",
    "ENAGOMENI": "[ΕΝΑΓΟΜΕΝΗ_{n}]",
    "MARTYRAS": "[ΜΑΡΤΥΡΑΣ_{n}]",
    "DIKIGOROS": "[ΔΙΚΗΓΟΡΟΣ_{n}]",
    "DIKASTIS": "[ΔΙΚΑΣΤΗΣ_{n}]",
}

# Findings that are considered "hard PII" for leadership demo
HARD_PII_CATEGORIES = {"EMAIL", "PHONE", "AFM", "AMKA", "ID", "ADDRESS", "DATE"}


# =========================
# 3) DATA STRUCTURES
# =========================
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
    role: str = ""          # for display


def _hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# =========================
# 4) ROLE ANCHORS (simple, high-value)
# =========================
ROLE_ANCHORS = [
    # label, regex that captures a name immediately after anchor
    ("ENAGON", re.compile(r"(?:^|\n)\s*Ενάγων\s*:\s*([^\n,]+)", re.IGNORECASE)),
    ("ENAGOMENI", re.compile(r"(?:^|\n)\s*Εναγομένη\s*:\s*([^\n,]+)", re.IGNORECASE)),
    ("DIKIGOROS", re.compile(r"(?:^|\n)\s*Δικηγόρος[^:]*:\s*([^\n,]+)", re.IGNORECASE)),
    ("DIKASTIS", re.compile(r"(?:^|\n)\s*Ο\s+ΔΙΚΑΣΤΗΣ\s*\n\s*([^\n]+)", re.IGNORECASE)),
    ("MARTYRAS", re.compile(r"(?:^|\n).{0,80}μάρτυρας.{0,20}\s+ο\s+([^\n,]+)", re.IGNORECASE)),
]

def extract_role_spans(text: str) -> List[Tuple[int, int, str, str]]:
    """
    Returns list of (start, end, captured_name, role_label)
    """
    spans = []
    for role, rx in ROLE_ANCHORS:
        for m in rx.finditer(text):
            name = m.group(1).strip()
            if not name:
                continue
            start = m.start(1)
            end = m.end(1)
            spans.append((start, end, name, role))
    return spans


# =========================
# 5) DETECTION
# =========================
def detect_findings(text: str, policy_name: str) -> List[Finding]:
    policy = POLICIES[policy_name]
    raw = []

    def add_matches(pattern, category, confidence, source="rule"):
        for m in pattern.finditer(text):
            raw.append((m.start(), m.end(), m.group(0), category, confidence, source, ""))

    # Hard PII (high confidence)
    add_matches(EMAIL_RE, "EMAIL", 0.98)
    add_matches(PHONE_RE, "PHONE", 0.95)
    add_matches(ID_RE, "ID", 0.92)
    add_matches(AMKA_RE, "AMKA", 0.92)
    add_matches(AFM_RE, "AFM", 0.88)
    add_matches(ADDRESS_RE, "ADDRESS", 0.85)

    # Dates depends on policy
    if policy["mask_dates"] != "keep":
        add_matches(DATE_RE, "DATE", 0.75)

    # Role-based PERSON spans (medium confidence but highly valuable)
    if policy["mask_persons"]:
        for start, end, name, role in extract_role_spans(text):
            # Skip if looks institutional
            if is_institution_like(name):
                continue
            raw.append((start, end, name, "PERSON", 0.78, "role_anchor", role))

    # Generic PERSON heuristic (low confidence) – kept but not shown by default
    if policy["mask_persons"]:
        for m in GREEK_NAME_HEURISTIC.finditer(text):
            span = m.group(0)
            if is_institution_like(span):
                continue
            raw.append((m.start(), m.end(), span, "PERSON", 0.55, "heuristic", ""))

    # Resolve overlaps: earliest, then longer, then higher confidence
    raw.sort(key=lambda x: (x[0], -(x[1] - x[0]), -x[4]))
    resolved = []
    last_end = -1
    for r in raw:
        if r[0] < last_end:
            continue
        resolved.append(r)
        last_end = r[1]

    # Assign consistent tokens (string-based within document)
    counters = {
        "EMAIL": 0, "PHONE": 0, "ADDRESS": 0,
        "PERSON": 0, "ENAGON": 0, "ENAGOMENI": 0, "MARTYRAS": 0, "DIKIGOROS": 0, "DIKASTIS": 0
    }
    memory: Dict[Tuple[str, str], str] = {}

    out: List[Finding] = []
    for start, end, val, cat, conf, src, role in resolved:
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
        elif cat in ("EMAIL", "PHONE", "ADDRESS"):
            key = (cat, val)
            if key not in memory:
                counters[cat] += 1
                memory[key] = TOKEN_TEMPLATES[cat].format(n=counters[cat])
            proposed = memory[key]
        elif cat == "PERSON":
            # If detected via role anchor, use role-specific token
            if role in ("ENAGON", "ENAGOMENI", "MARTYRAS", "DIKIGOROS", "DIKASTIS"):
                key = (role, val)
                if key not in memory:
                    counters[role] += 1
                    memory[key] = TOKEN_TEMPLATES[role].format(n=counters[role])
                proposed = memory[key]
            else:
                # generic person
                key = ("PERSON", val)
                if key not in memory:
                    counters["PERSON"] += 1
                    memory[key] = TOKEN_TEMPLATES["PERSON"].format(n=counters["PERSON"])
                proposed = memory[key]
        else:
            continue

        fid = _hash_id(f"{start}-{end}-{cat}-{val}-{role}")
        out.append(Finding(
            id=fid, start=start, end=end, text=val,
            category=cat, confidence=conf, source=src,
            proposed=proposed, role=role
        ))

    return out


# =========================
# 6) APPLY + EXPORT
# =========================
def apply_findings(text: str, findings: List[Finding]) -> Tuple[str, List[dict], List[dict]]:
    selected = []
    audit = []

    for f in findings:
        if f.status == "REJECT":
            audit.append({"ts": time.time(), "id": f.id, "action": "REJECT",
                          "category": f.category, "role": f.role, "text": f.text})
            continue

        if f.status == "EDIT":
            rep = f.edited.strip() or f.proposed
            audit.append({"ts": time.time(), "id": f.id, "action": "EDIT",
                          "category": f.category, "role": f.role, "text": f.text, "replacement": rep})
            selected.append((f.start, f.end, f.text, f.category, f.confidence, f.source, f.role, rep))
        else:
            audit.append({"ts": time.time(), "id": f.id, "action": "ACCEPT",
                          "category": f.category, "role": f.role, "text": f.text, "replacement": f.proposed})
            selected.append((f.start, f.end, f.text, f.category, f.confidence, f.source, f.role, f.proposed))

    # apply from end
    selected.sort(key=lambda x: x[0], reverse=True)
    out = text
    redactions = []
    for start, end, original, cat, conf, src, role, rep in selected:
        out = out[:start] + rep + out[end:]
        redactions.append({
            "start": start, "end": end,
            "original": original,
            "category": cat,
            "role": role,
            "confidence": conf,
            "source": src,
            "replacement": rep
        })
    redactions.reverse()
    return out, redactions, audit


def render_highlighted_html(text: str, findings: List[Finding]) -> str:
    spans = []
    for f in findings:
        if f.status == "REJECT":
            continue
        rep = f.edited.strip() if f.status == "EDIT" else f.proposed
        spans.append((f.start, f.end, f.text, f.category, f.role, rep))

    spans.sort(key=lambda x: x[0], reverse=True)
    out = text
    for start, end, original, cat, role, rep in spans:
        safe = (original.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        tag = f"{role or cat} → {rep}"
        chunk = f"""<mark style="padding:2px 4px;border-radius:8px;">
        {safe}
        <span style="font-size:11px;opacity:.70;margin-left:8px;">{tag}</span>
        </mark>"""
        out = out[:start] + chunk + out[end:]
    out = out.replace("\n", "<br/>")
    return f"<div style='font-family: ui-sans-serif, system-ui; line-height:1.6;'>{out}</div>"


# =========================
# 7) STREAMLIT UI (Leadership-first)
# =========================
st.set_page_config(page_title="Justice Redaction (Internal Demo)", layout="wide")
st.title("Demo: Ευφυής Ανωνυμοποίηση / Ψευδωνυμοποίηση — Εσωτερική Αξιοποίηση")

with st.sidebar:
    st.header("Demo Controls")
    policy = st.selectbox("Policy", ["internal", "controlled", "open_data"], index=0)
    st.caption("Για Υπουργείο: default = internal (ασφαλής εσωτερική αξιοποίηση).")

    view_mode = st.selectbox(
        "View",
        ["PII_ONLY", "LOW_CONF_REVIEW", "ADVANCED_ALL"],
        index=0
    )
    st.caption("PII_ONLY = δείχνει μόνο σίγουρα PII (χωρίς να εκθέτει ML λάθη).")

    uploaded = st.file_uploader("Upload .txt (OCR/HTR output)", type=["txt"])
    st.divider()
    st.subheader("Messaging (για Διοίκηση)")
    st.write("• Πολιτικές ανά χρήση (internal / controlled / open_data)")
    st.write("• Human-in-the-loop μόνο όπου χρειάζεται")
    st.write("• Audit + exports για ελέγχους (GDPR-by-design)")


default_text = """ΕΛΛΗΝΙΚΗ ΔΗΜΟΚΡΑΤΙΑ
ΜΟΝΟΜΕΛΕΣ ΠΡΩΤΟΔΙΚΕΙΟ ΑΘΗΝΩΝ
Αριθμός Απόφασης: 1456/2024

Στην Αθήνα, σήμερα την 12/03/2024, το Μονομελές Πρωτοδικείο Αθηνών συνεδρίασε δημόσια στο ακροατήριό του.

Διάδικοι:

Ενάγων: Ιωάννης Παπαδόπουλος του Δημητρίου και της Μαρίας, γεννηθείς την 15/05/1985,
κάτοικος Αθηνών, οδός Πατησίων 120, Τ.Κ. 11256,
ΑΦΜ 123456789, ΑΜΚΑ 15058512345,
τηλ. 6971234567, email: ioannis.papadopoulos@gmail.com

Εναγομένη: Μαρία Κωνσταντίνου του Νικολάου,
γεννηθείσα την 22/11/1988,
κάτοικος Πειραιά, Οδός Αγίου Δημητρίου 45,
ΑΦΜ 987654321,
τηλ. 2101234567, email: maria.konstantinou@yahoo.gr

Δικηγόρος ενάγοντος: Ανδρέας Λεμονής, ΑΔΤ ΑΒ 345678
Δικηγόρος εναγομένης: Ελένη Μάρκου, ΑΔΤ ΧΖ 987654

Στο πλαίσιο της αποδεικτικής διαδικασίας εξετάστηκε ως μάρτυρας
ο Πέτρος Δημητρίου του Αλεξάνδρου, κάτοικος Γλυφάδας,
οδός Κύπρου 12, τηλ. 6987654321.

Ο ΔΙΚΑΣΤΗΣ
Αλέξανδρος Γεωργίου
"""

if "text" not in st.session_state:
    st.session_state.text = default_text

if uploaded is not None:
    st.session_state.text = uploaded.read().decode("utf-8", errors="replace")

left, right = st.columns([1, 1])

with left:
    st.subheader("Original (input)")
    st.session_state.text = st.text_area("Κείμενο", value=st.session_state.text, height=320)

    run_btn = st.button("Run", type="primary", use_container_width=True)

with right:
    st.subheader("Outputs (what the system produces)")
    st.markdown("""
- **Anonymized text** (αναγνώσιμο, με role-aware tokens)
- **Redaction map (JSON)**: τι άλλαξε/πού/με confidence
- **Audit trail**: accept/edit/reject events
""")


if run_btn or "findings" not in st.session_state:
    st.session_state.findings = detect_findings(st.session_state.text, policy)

findings: List[Finding] = st.session_state.findings
threshold = POLICIES[policy]["hitl_threshold"]

# Governance KPIs for leadership
total = len(findings)
low = sum(1 for f in findings if f.confidence < threshold)
hard_pii = sum(1 for f in findings if f.category in HARD_PII_CATEGORIES)
est_manual_sec_per_item = 12  # rough demo assumption
est_saved = max(0, hard_pii * est_manual_sec_per_item * 0.6)  # assume 60% time reduction

k1, k2, k3, k4 = st.columns(4)
k1.metric("Detected items", total)
k2.metric("Hard PII items", hard_pii)
k3.metric("Low-confidence for review", low)
k4.metric("Estimated time saved", f"{int(est_saved//60)} min")

st.caption(f"Low-confidence threshold για policy '{policy}': {threshold:.2f}.")


# Build table according to view mode
rows = []
for f in findings:
    if view_mode == "PII_ONLY" and f.category not in HARD_PII_CATEGORIES:
        continue
    if view_mode == "LOW_CONF_REVIEW" and not (f.confidence < threshold):
        continue

    role_label = f.role if f.role else ""
    rows.append({
        "id": f.id,
        "type": f.category,
        "role": role_label,
        "text": f.text,
        "confidence": round(f.confidence, 2),
        "source": f.source,
        "proposed": f.proposed,
        "status": f.status,
        "edited": f.edited
    })

df = pd.DataFrame(rows)

st.subheader("Review table (HITL)")
if df.empty:
    st.info("Δεν υπάρχουν εγγραφές για το τρέχον view. Δοκίμασε ADVANCED_ALL ή άλλαξε policy/view.")
else:
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "status": st.column_config.SelectboxColumn(
                "status", options=["ACCEPT", "REJECT", "EDIT"], required=True
            ),
            "edited": st.column_config.TextColumn(
                "edited",
                help="Χρησιμοποιείται μόνο όταν status=EDIT (π.χ. [ΕΝΑΓΩΝ_1])"
            ),
        }
    )

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
    st.subheader("Explainability preview (highlighted)")
    html = render_highlighted_html(st.session_state.text, findings)
    st.components.v1.html(html, height=360, scrolling=True)

with c2:
    st.subheader("Anonymized output")
    st.text_area("anonymized", value=anonymized_text, height=360)

st.subheader("Exports (for compliance & reuse)")
meta = {
    "policy": policy,
    "view_mode": view_mode,
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "detected_items": total,
    "hard_pii_items": hard_pii,
    "low_confidence_items": low
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
st.caption("Demo σημείωση: αυτό είναι governance-first UI. Για production απαιτούνται legal-trained models, OCR quality gates και leakage testing.")

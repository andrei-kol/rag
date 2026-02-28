"""
Generate realistic synthetic financial and legal documents.

This script creates all the sample documents used throughout the project:
  - Annual financial reports (with revenue tables, narrative, footnotes)
  - Service contracts and NDAs
  - Internal policy documents
  - KYC documents

Two-step process:
  1. GPT-4o-mini generates the text content (structured JSON with sections)
  2. reportlab formats it into a proper-looking PDF (header, footer, page numbers,
     tables styled as financial tables)

Why synthetic data?
  - We control the ground truth: we know exact revenue figures, key clauses, etc.
  - Lets us build a proper evaluation dataset (Article 5)
  - No confidentiality concerns
  - Reproducible

Run:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --skip-llm  # use cached content
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make sure `src` is importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import settings

# ---------------------------------------------------------------------------
# Document definitions — what we want to generate
# ---------------------------------------------------------------------------

CLIENTS = {
    "acme_corp": {
        "name": "Acme Corp",
        "industry": "B2B SaaS",
        "country": "United States",
        "founded": 2015,
    },
    "globex_inc": {
        "name": "Globex Inc",
        "industry": "Industrial Manufacturing",
        "country": "Germany",
        "founded": 1998,
    },
    "initech_llc": {
        "name": "Initech LLC",
        "industry": "Fintech Startup",
        "country": "United Kingdom",
        "founded": 2022,
    },
}

DOCUMENT_SPECS = [
    # --- Acme Corp ---
    {
        "client_id": "acme_corp",
        "doc_type": "financial_report",
        "year": 2023,
        "output_path": "data/raw/clients/acme_corp/financial_report_2023.pdf",
        "doc_metadata": {
            "client_id": "acme_corp",
            "doc_type": "financial_report",
            "year": 2023,
            "confidentiality": "confidential",
        },
    },
    {
        "client_id": "acme_corp",
        "doc_type": "financial_report",
        "year": 2022,
        "output_path": "data/raw/clients/acme_corp/financial_report_2022.pdf",
        "doc_metadata": {
            "client_id": "acme_corp",
            "doc_type": "financial_report",
            "year": 2022,
            "confidentiality": "confidential",
        },
    },
    {
        "client_id": "acme_corp",
        "doc_type": "contract",
        "year": 2024,
        "output_path": "data/raw/clients/acme_corp/service_contract_2024.pdf",
        "doc_metadata": {
            "client_id": "acme_corp",
            "doc_type": "contract",
            "year": 2024,
            "confidentiality": "confidential",
        },
    },
    # --- Globex Inc ---
    {
        "client_id": "globex_inc",
        "doc_type": "financial_report",
        "year": 2023,
        "output_path": "data/raw/clients/globex_inc/financial_report_2023.pdf",
        "doc_metadata": {
            "client_id": "globex_inc",
            "doc_type": "financial_report",
            "year": 2023,
            "confidentiality": "confidential",
        },
    },
    {
        "client_id": "globex_inc",
        "doc_type": "nda",
        "year": 2023,
        "output_path": "data/raw/clients/globex_inc/nda_2023.pdf",
        "doc_metadata": {
            "client_id": "globex_inc",
            "doc_type": "nda",
            "year": 2023,
            "confidentiality": "confidential",
        },
    },
    # --- Initech LLC ---
    {
        "client_id": "initech_llc",
        "doc_type": "financial_projections",
        "year": 2024,
        "output_path": "data/raw/clients/initech_llc/financial_projections_2024.pdf",
        "doc_metadata": {
            "client_id": "initech_llc",
            "doc_type": "financial_projections",
            "year": 2024,
            "confidentiality": "confidential",
        },
    },
    # --- Internal policies ---
    {
        "client_id": None,
        "doc_type": "risk_management_policy",
        "year": 2024,
        "output_path": "data/raw/policies/risk_management_policy.pdf",
        "doc_metadata": {
            "doc_type": "policy",
            "policy_name": "risk_management",
            "year": 2024,
            "confidentiality": "internal",
        },
    },
    {
        "client_id": None,
        "doc_type": "kyc_aml_procedures",
        "year": 2024,
        "output_path": "data/raw/policies/kyc_aml_procedures.pdf",
        "doc_metadata": {
            "doc_type": "policy",
            "policy_name": "kyc_aml",
            "year": 2024,
            "confidentiality": "internal",
        },
    },
]

# ---------------------------------------------------------------------------
# LLM content generation
# ---------------------------------------------------------------------------

PROMPTS: dict[str, str] = {
    "financial_report": """\
Generate a realistic annual financial report for {company_name} ({industry}, {country}).
Year: {year}

Return a JSON object with this exact structure:
{{
  "title": "Annual Financial Report {year}",
  "company": "{company_name}",
  "executive_summary": "2-3 paragraph narrative about the year",
  "revenue_table": [
    ["Metric", "Q1 {year}", "Q2 {year}", "Q3 {year}", "Q4 {year}", "Full Year {year}", "Full Year {prev_year}"],
    ["Revenue ($M)", "...", "...", "...", "...", "...", "..."],
    ["Gross Profit ($M)", "...", "...", "...", "...", "...", "..."],
    ["Operating Expenses ($M)", "...", "...", "...", "...", "...", "..."],
    ["Net Income ($M)", "...", "...", "...", "...", "...", "..."],
    ["Gross Margin (%)", "...", "...", "...", "...", "...", "..."]
  ],
  "business_highlights": ["3-5 bullet points as strings"],
  "risks_and_outlook": "1-2 paragraph narrative",
  "footnotes": "Brief legal disclaimer"
}}

Use realistic numbers consistent across quarters. Make the company sound credible.
Return only the JSON, no markdown fences.""",

    "contract": """\
Generate a realistic B2B service contract between FinPlatform Inc. (service provider)
and {company_name} (client). Year: {year}.

Return a JSON object:
{{
  "title": "Master Service Agreement",
  "parties": {{
    "provider": "FinPlatform Inc., registered in Delaware, USA",
    "client": "{company_name}, {country}"
  }},
  "effective_date": "{year}-01-15",
  "term": "24 months, auto-renewing",
  "sections": [
    {{"heading": "1. Services", "body": "...2-3 sentences..."}},
    {{"heading": "2. Payment Terms", "body": "...include specific amounts and schedule..."}},
    {{"heading": "3. Confidentiality", "body": "..."}},
    {{"heading": "4. Intellectual Property", "body": "..."}},
    {{"heading": "5. Limitation of Liability", "body": "..."}},
    {{"heading": "6. Termination", "body": "..."}},
    {{"heading": "7. Governing Law", "body": "..."}}
  ],
  "sla_table": [
    ["Service", "Uptime SLA", "Response Time", "Penalty"],
    ["API Platform", "99.9%", "< 200ms p95", "5% monthly credit"],
    ["Data Processing", "99.5%", "< 4 hours", "3% monthly credit"],
    ["Support", "Business hours", "< 4 hours", "N/A"]
  ],
  "signature_block": "Standard signature lines placeholder"
}}
Return only the JSON.""",

    "nda": """\
Generate a realistic Non-Disclosure Agreement between FinPlatform Inc. and {company_name}.
Year: {year}.

Return a JSON object:
{{
  "title": "Non-Disclosure Agreement",
  "parties": {{
    "disclosing": "FinPlatform Inc.",
    "receiving": "{company_name}"
  }},
  "effective_date": "{year}-03-01",
  "term": "3 years",
  "sections": [
    {{"heading": "1. Definition of Confidential Information", "body": "..."}},
    {{"heading": "2. Obligations of Receiving Party", "body": "..."}},
    {{"heading": "3. Exclusions", "body": "..."}},
    {{"heading": "4. Term and Termination", "body": "..."}},
    {{"heading": "5. Remedies", "body": "..."}},
    {{"heading": "6. Governing Law", "body": "..."}}
  ],
  "signature_block": "Standard signature lines"
}}
Return only the JSON.""",

    "financial_projections": """\
Generate realistic 3-year financial projections for {company_name} ({industry}, {country}),
a startup seeking Series A funding. Base year: {year}.

Return a JSON object:
{{
  "title": "Financial Projections {year}–{year_plus_2}",
  "company": "{company_name}",
  "executive_summary": "1-2 paragraphs about the growth story",
  "assumptions": ["4-6 key assumptions as strings"],
  "projections_table": [
    ["Metric", "{year}", "{year_plus_1}", "{year_plus_2}"],
    ["ARR ($M)", "...", "...", "..."],
    ["Revenue ($M)", "...", "...", "..."],
    ["Gross Margin (%)", "...", "...", "..."],
    ["Operating Expenses ($M)", "...", "...", "..."],
    ["EBITDA ($M)", "...", "...", "..."],
    ["Headcount", "...", "...", "..."]
  ],
  "use_of_funds": [
    ["Category", "Amount ($M)", "% of Raise"],
    ["Engineering", "...", "..."],
    ["Sales & Marketing", "...", "..."],
    ["Operations", "...", "..."],
    ["Working Capital", "...", "..."]
  ],
  "risks": ["3-4 key risks as strings"]
}}
Return only the JSON.""",

    "risk_management_policy": """\
Generate a realistic internal Risk Management Policy for a B2B fintech platform.
Year: {year}.

Return a JSON object:
{{
  "title": "Risk Management Policy",
  "version": "2.1",
  "effective_date": "{year}-01-01",
  "owner": "Chief Risk Officer",
  "sections": [
    {{"heading": "1. Purpose and Scope", "body": "...2-3 sentences..."}},
    {{"heading": "2. Risk Categories", "body": "...define market, credit, operational, regulatory risks..."}},
    {{"heading": "3. Risk Appetite", "body": "..."}},
    {{"heading": "4. Risk Assessment Process", "body": "..."}},
    {{"heading": "5. Risk Thresholds", "body": "..."}},
    {{"heading": "6. Reporting and Escalation", "body": "..."}},
    {{"heading": "7. Review and Update", "body": "..."}}
  ],
  "risk_matrix_table": [
    ["Risk Level", "Probability", "Impact", "Action Required"],
    ["Critical", "High", "High", "Immediate escalation to Board"],
    ["High", "Medium-High", "High", "CEO notification within 24h"],
    ["Medium", "Medium", "Medium", "Risk Committee review"],
    ["Low", "Low", "Low", "Document and monitor"]
  ],
  "footnote": "This policy is reviewed annually."
}}
Return only the JSON.""",

    "kyc_aml_procedures": """\
Generate a realistic KYC/AML Procedures document for a B2B fintech platform.
Year: {year}.

Return a JSON object:
{{
  "title": "KYC & AML Procedures Guide",
  "version": "3.0",
  "effective_date": "{year}-01-01",
  "regulatory_basis": ["FATF Recommendations", "EU AML Directive", "FinCEN requirements"],
  "sections": [
    {{"heading": "1. Customer Identification Program (CIP)", "body": "..."}},
    {{"heading": "2. Customer Due Diligence (CDD)", "body": "..."}},
    {{"heading": "3. Enhanced Due Diligence (EDD)", "body": "...for high-risk clients..."}},
    {{"heading": "4. Beneficial Ownership", "body": "..."}},
    {{"heading": "5. Transaction Monitoring", "body": "..."}},
    {{"heading": "6. Suspicious Activity Reporting (SAR)", "body": "..."}},
    {{"heading": "7. Record Keeping", "body": "..."}}
  ],
  "onboarding_checklist_table": [
    ["Document", "Required For", "Retention Period"],
    ["Certificate of Incorporation", "All clients", "7 years"],
    ["Director ID (passport/ID)", "All clients", "7 years"],
    ["Proof of address", "All clients", "7 years"],
    ["Beneficial ownership declaration", "Clients with >25% owners", "7 years"],
    ["Source of funds declaration", "High-risk clients", "7 years"],
    ["PEP screening result", "All clients", "7 years"]
  ],
  "footnote": "Non-compliance with this procedure is subject to disciplinary action."
}}
Return only the JSON.""",
}


# ---------------------------------------------------------------------------
# PDF generation with reportlab
# ---------------------------------------------------------------------------

STYLES = getSampleStyleSheet()

_TITLE_STYLE = ParagraphStyle(
    "DocTitle",
    parent=STYLES["Title"],
    fontSize=20,
    spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"),
)
_SUBTITLE_STYLE = ParagraphStyle(
    "DocSubtitle",
    parent=STYLES["Normal"],
    fontSize=11,
    textColor=colors.HexColor("#555555"),
    spaceAfter=20,
)
_HEADING_STYLE = ParagraphStyle(
    "SecHeading",
    parent=STYLES["Heading2"],
    fontSize=13,
    textColor=colors.HexColor("#1a1a2e"),
    spaceBefore=16,
    spaceAfter=6,
    borderPad=0,
)
_BODY_STYLE = ParagraphStyle(
    "Body",
    parent=STYLES["Normal"],
    fontSize=10,
    leading=15,
    spaceAfter=8,
)
_BULLET_STYLE = ParagraphStyle(
    "Bullet",
    parent=_BODY_STYLE,
    leftIndent=20,
    bulletIndent=10,
)

_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ("ALIGN", (0, 0), (0, -1), "LEFT"),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
])


def _add_page_number(canvas, doc):
    """Footer callback: adds page number and document title."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2 * cm, 1.2 * cm, "CONFIDENTIAL — FinPlatform Inc.")
    canvas.drawRightString(
        A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}"
    )
    canvas.restoreState()


def build_pdf_from_content(content: dict, output_path: Path) -> None:
    """
    Build a formatted PDF from structured content dict.
    Handles financial_report, contract, nda, financial_projections, policy formats.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
    )

    story = []

    # --- Title block ---
    story.append(Paragraph(content.get("title", "Document"), _TITLE_STYLE))
    if "company" in content:
        story.append(Paragraph(content["company"], _SUBTITLE_STYLE))
    if "parties" in content:
        p = content["parties"]
        parties_text = f"{p.get('provider', p.get('disclosing', ''))} ↔ {p.get('client', p.get('receiving', ''))}"
        story.append(Paragraph(parties_text, _SUBTITLE_STYLE))
    if "effective_date" in content:
        story.append(Paragraph(f"Effective: {content['effective_date']}", _SUBTITLE_STYLE))

    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.4 * cm))

    # --- Executive summary / narrative ---
    for key in ("executive_summary", "purpose"):
        if key in content:
            story.append(Paragraph("Executive Summary" if key == "executive_summary" else "Purpose", _HEADING_STYLE))
            story.append(Paragraph(content[key], _BODY_STYLE))

    # --- Main data table (revenue, projections, etc.) ---
    for key in ("revenue_table", "projections_table", "sla_table"):
        if key in content:
            label = key.replace("_", " ").title()
            story.append(Paragraph(label, _HEADING_STYLE))
            tbl = Table(content[key], hAlign="LEFT", repeatRows=1)
            tbl.setStyle(_TABLE_STYLE)
            story.append(tbl)
            story.append(Spacer(1, 0.3 * cm))

    # --- Sections (contracts, policies, NDAs) ---
    if "sections" in content:
        for section in content["sections"]:
            story.append(Paragraph(section["heading"], _HEADING_STYLE))
            story.append(Paragraph(section["body"], _BODY_STYLE))

    # --- Business highlights / assumptions / risks ---
    for key, label in [
        ("business_highlights", "Business Highlights"),
        ("assumptions", "Key Assumptions"),
        ("risks", "Key Risks"),
        ("regulatory_basis", "Regulatory Basis"),
    ]:
        if key in content:
            story.append(Paragraph(label, _HEADING_STYLE))
            for item in content[key]:
                story.append(Paragraph(f"• {item}", _BULLET_STYLE))
            story.append(Spacer(1, 0.2 * cm))

    # --- Additional tables ---
    for key, label in [
        ("use_of_funds", "Use of Funds"),
        ("risk_matrix_table", "Risk Matrix"),
        ("onboarding_checklist_table", "Onboarding Checklist"),
    ]:
        if key in content:
            story.append(Paragraph(label, _HEADING_STYLE))
            tbl = Table(content[key], hAlign="LEFT", repeatRows=1)
            tbl.setStyle(_TABLE_STYLE)
            story.append(tbl)
            story.append(Spacer(1, 0.3 * cm))

    # --- Risks and outlook ---
    if "risks_and_outlook" in content:
        story.append(Paragraph("Risks and Outlook", _HEADING_STYLE))
        story.append(Paragraph(content["risks_and_outlook"], _BODY_STYLE))

    # --- Footnote ---
    if "footnotes" in content or "footnote" in content:
        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        note = content.get("footnotes", content.get("footnote", ""))
        fn_style = ParagraphStyle("Footnote", parent=_BODY_STYLE, fontSize=8, textColor=colors.grey)
        story.append(Paragraph(note, fn_style))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    print(f"  ✓ {output_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_content_with_llm(spec: dict, client_info: dict | None) -> dict:
    """Call GPT to generate document content for a given spec."""
    doc_type = spec["doc_type"]
    prompt_template = PROMPTS.get(doc_type)
    if not prompt_template:
        raise ValueError(f"No prompt template for doc_type: {doc_type}")

    ci = client_info or {}
    prompt = prompt_template.format(
        company_name=ci.get("name", "N/A"),
        industry=ci.get("industry", "N/A"),
        country=ci.get("country", "N/A"),
        year=spec["year"],
        prev_year=spec["year"] - 1,
        year_plus_1=spec["year"] + 1,
        year_plus_2=spec["year"] + 2,
    )

    oai = OpenAI(api_key=settings.openai_api_key)
    response = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def load_cached_content(cache_path: Path) -> dict | None:
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


def save_cached_content(content: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(content, f, indent=2)


def main(skip_llm: bool = False) -> None:
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "data" / "processed" / "_llm_cache"

    print(f"\n{'='*60}")
    print("  RAG Financial Docs — Synthetic Data Generator")
    print(f"{'='*60}\n")

    for spec in DOCUMENT_SPECS:
        output_path = project_root / spec["output_path"]
        cache_path = cache_dir / f"{output_path.stem}.json"
        client_info = CLIENTS.get(spec["client_id"]) if spec["client_id"] else None

        label = f"{spec['doc_type']} / {spec.get('client_id', 'internal')} / {spec['year']}"
        print(f"→ Generating: {label}")

        # Load from cache or generate with LLM
        content = load_cached_content(cache_path)
        if content is None or not skip_llm:
            content = generate_content_with_llm(spec, client_info)
            save_cached_content(content, cache_path)

        # Build PDF
        build_pdf_from_content(content, output_path)

    print(f"\n✅ Done. Documents saved to data/raw/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic documents")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Re-use cached LLM responses (faster, no API cost)",
    )
    args = parser.parse_args()
    main(skip_llm=args.skip_llm)

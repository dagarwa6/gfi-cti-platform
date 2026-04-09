# GFI Cyber Threat Intelligence Platform — Milestone 2

**CIS 8684 · Cyber Threat Intelligence · Section 003 · Spring 2026**

**Group:** Devansh Agarwal (Coordinator), Anica, Noreen, Guled, Ville
**Institution:** Georgia State University — J. Mack Robinson College of Business

---

## Project Overview

This Streamlit application is **Milestone 2** of a four-milestone project to build a
**Cyber Threat Intelligence (CTI) platform for Global Financial Institutions (GFI)**.
The platform targets major financial institutions operating investment banking,
capital markets, and retail banking — facing threat actors from ransomware groups
to nation-state APTs.

The platform's core innovation is a **dual-level ELO scoring engine** (introduced in
Milestones 3–4) that dynamically prioritizes CVEs and threat actors based on
organizational context.

---

## Live Demo

**https://gfi-cti-m2.streamlit.app**

---

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run milestone2_app.py
```

The app will open in your browser at `http://localhost:8501`.

All data is fetched live — internet access required. API calls are cached 30–120 minutes per source.

---

## Requirements

```
streamlit>=1.32.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.20.0
requests>=2.31.0
python-dateutil>=2.9.0
```

A `requirements.txt` file is included in this directory.

---

## App Sections (Sidebar Navigation)

| Section | Description |
|---|---|
| ✅ What's New | Milestone 1 + 2 change checklists |
| 🏦 Industry Background | GFI overview: services, size, players, IT criticality |
| 👥 Stakeholders & Use Case | 3 personas, 6 user stories, CTI use case design |
| 📈 Threat Trends & Assets | Interactive threat trends, 8 critical assets, exposure matrix |
| 💎 Diamond Models | 2 complete Diamond Models with Plotly visualization + JSON export |
| 📊 Live Dashboard | CISA KEV + EPSS live data with interactive filters and KPIs |
| 💼 Intelligence Buy-In | Breach cost trends, ROI calculator, business case |
| 📡 Data Sources | 7 live sources with justification, collection strategy, metadata |
| 🔍 Data Explorer | 4-tab interactive explorer with cross-source correlations |
| ⚖️ Ethics & Security | Ethics, privacy, security practices, reproducibility |
| 👨‍💼 Team | Roles, contributions, and electronic signatures |

---

## Data Sources (Milestone 2)

All data sources are **free, publicly accessible OSINT** (except VirusTotal which requires a free API key):

| Source | URL | Type | Auth |
|---|---|---|---|
| URLhaus | urlhaus.abuse.ch | CSV (GET) | None |
| MalwareBazaar | bazaar.abuse.ch | JSON (POST) | None |
| Ransomware.live | api.ransomware.live | JSON (GET) | None |
| ThreatFox | threatfox.abuse.ch | CSV (GET) | None |
| SEC EDGAR 8-K | efts.sec.gov | JSON (GET) | None (User-Agent required) |
| VirusTotal | virustotal.com/api/v3 | JSON (GET) | Free API key |
| CISA KEV | cisa.gov | JSON (GET) | None |
| EPSS | api.first.org | JSON (GET) | None |

Hardcoded fallback data is included for all sources to ensure live demo reliability.

---

## Data Folder Structure

```
GFI-CTI-Platform/
├── milestone2_app.py          # Main Streamlit app (Milestones 1 + 2)
├── milestone1_app.py          # Milestone 1 reference version
├── requirements.txt           # Python dependencies (pinned versions)
├── README.md                  # This file
├── .streamlit/
│   └── config.toml            # Dark theme configuration
├── data/                      # Optional: cached/offline snapshots
│   ├── urlhaus_snapshot.json
│   ├── ransomware_snapshot.json
│   └── sec_edgar_snapshot.json
└── docs/
    └── GFI_CTI_Proposal.pptx
```

No external data files are required. All data is fetched live from APIs at runtime
and cached via `@st.cache_data(ttl=N)`. Fallback data is hardcoded for reliability.

---

## How to Reproduce Analysis

1. Python >= 3.10 and `pip` required
2. Clone the repo or extract the ZIP
3. `pip install -r requirements.txt`
4. `streamlit run milestone2_app.py`
5. Open `http://localhost:8501` — all data fetched live (cached 30–120 min)
6. VirusTotal: if you have an API key, place it in `.streamlit/secrets.toml` as `VT_API_KEY = "your_key"`. Without it, demo fallback data loads automatically.

---

## Milestone Roadmap

| Milestone | Due | Focus |
|---|---|---|
| M1 | March 26, 2026 | Industry background, threat trends, diamond models, dashboard starter |
| **M2 (current)** | **April 9, 2026** | **7 data sources, collection strategy, data explorer, ethics, security** |
| M3 | April 23, 2026 | ELO scoring engine (CVE-level + Threat Actor-level), analytics |
| M4 | April 30, 2026 | Final polish, triage dashboard, role-based views, STIX export |

---

## Team

| Name | Role |
|---|---|
| **Devansh Agarwal** ⭐ | Project Coordinator, Streamlit App Lead |
| Anica | Data Collection & Pipeline Engineer |
| Noreen | Threat Modeling & Intelligence Research |
| Guled | Analytics & Validation Lead |
| Ville | Visualizations & Dashboard Design |

⭐ = Team Coordinator (point of contact for instructor)

---

*Submitted for CIS 8684 — Cyber Threat Intelligence · Spring 2026 · Georgia State University*

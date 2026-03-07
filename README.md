# GFI Cyber Threat Intelligence Platform — Milestone 1

**CIS 8684 · Cyber Threat Intelligence · Section 003 · Spring 2026**

**Group:** Devansh Agarwal (Coordinator), Anica, Noreen, Guled, Ville
**Institution:** Georgia State University — J. Mack Robinson College of Business

---

## Project Overview

This Streamlit application is Milestone 1 of a four-milestone project to build a
**Cyber Threat Intelligence (CTI) platform for Global Financial Institutions (GFI)**.
The platform targets firms like JPMorgan Chase, Goldman Sachs, Citigroup, and HSBC —
institutions that simultaneously operate investment banking, capital markets, and retail
banking, and face a unique combination of threat actors ranging from ransomware groups
to nation-state APTs.

The platform's core innovation is a **dual-level ELO scoring engine** (introduced in
Milestones 3–4) that dynamically prioritizes CVEs and threat actors based on
organizational context — replacing generic CVSS/EPSS rankings with organization-specific
threat prioritization.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run milestone1_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Requirements

```
streamlit>=1.33
pandas>=2.0
numpy>=1.24
plotly>=5.18
requests>=2.31
scikit-learn>=1.3
networkx>=3.1
```

A `requirements.txt` file is included in this directory.

---

## App Sections (Sidebar Navigation)

| Section | Description |
|---|---|
| ✅ What's New | Milestone 1 change checklist (required) |
| 🏦 Industry Background | GFI overview: services, size, players, IT criticality |
| 👥 Stakeholders & Use Case | 3 personas, 6 user stories, CTI use case design |
| 📈 Threat Trends & Assets | Interactive threat trends, 8 critical assets, exposure matrix |
| 💎 Diamond Models | 2 complete Diamond Models with Plotly visualization + JSON export |
| 📊 Live Dashboard | CISA KEV + EPSS live data with interactive filters and KPIs |
| 💼 Intelligence Buy-In | Breach cost trends, ROI calculator, business case |
| 👨‍💼 Team | Roles, contributions, and electronic signatures |

---

## Data Sources

All data sources used in Milestone 1 are **free and publicly accessible**:

| Source | URL | Type |
|---|---|---|
| CISA KEV | https://www.cisa.gov/known-exploited-vulnerabilities-catalog | Live JSON API |
| EPSS | https://api.first.org/data/v1/epss | Live JSON API |
| Synthetic Trends | Seeded with realistic financial sector values | Local (deterministic) |

Additional sources (Feodo Tracker, PhishTank, Ransomware.live, SEC EDGAR) will be
integrated in Milestone 2.

---

## Data Folder Structure

```
Cyber Threat Intelligence/
├── milestone1_app.py        ← Main Streamlit application (this milestone)
├── requirements.txt         ← Python package dependencies
└── README.md                ← This file
```

No external data files are required for Milestone 1. All live data is fetched from
APIs at runtime and cached for 1 hour. The app includes graceful fallback handling
if APIs are unavailable.

---

## How to Reproduce Analysis

1. Clone or download this directory
2. Install dependencies: `pip install -r requirements.txt`
3. Launch: `streamlit run milestone1_app.py`
4. Use the **sidebar** to navigate between sections
5. Use the **Global Filters** in the sidebar to adjust sub-sector, threat categories, and year range
6. The **Live Dashboard** section fetches fresh CISA KEV and EPSS data on each app launch (cached 1 hour)

---

## Milestone Roadmap

| Milestone | Due | Focus |
|---|---|---|
| **M1 (this file)** | March 26, 2026 | Industry background, threat trends, diamond models, dashboard starter |
| M2 | April 9, 2026 | Data source integration: Feodo Tracker, PhishTank, Ransomware.live, SEC EDGAR |
| M3 | April 23, 2026 | ELO scoring engine (CVE-level + Threat Actor-level), analytics, visualizations |
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

# GFI Cyber Threat Intelligence Platform — Milestone 3

**CIS 8684 · Cyber Threat Intelligence · Section 003 · Spring 2026**

**Group:** Devansh Agarwal (Coordinator), Anica, Noreen, Guled, Ville
**Institution:** Georgia State University — J. Mack Robinson College of Business

---

## Project Overview

This Streamlit application is **Milestone 3** of a four-milestone project to build a
**Cyber Threat Intelligence (CTI) platform for Global Financial Institutions (GFI)**.
The platform targets major financial institutions operating investment banking,
capital markets, and retail banking — facing threat actors from ransomware groups
to nation-state APTs.

**Milestone 3** introduces three analytical approaches aligned with Module 3 course content:

1. **CVE Ransomware-Risk Classification** — Random Forest binary classifier predicting
   which CISA KEV CVEs are likely to be weaponised in ransomware campaigns.
2. **Temporal Anomaly Detection** — Rolling z-score anomaly detection on KEV monthly
   addition rates and Ransomware.live daily victim counts.
3. **Cross-Source IOC Correlation** — Compound Confidence Scoring (CCS) via set
   intersection across URLhaus, ThreatFox, and MalwareBazaar malware families.

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
streamlit run milestone3_app.py
```

The app will open in your browser at `http://localhost:8501`.

All data is fetched live — internet access required. API calls are cached 30-120 minutes per source. Hardcoded fallback data loads automatically if any API is unavailable.

---

## Requirements

```
streamlit>=1.32.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.20.0
requests>=2.31.0
python-dateutil>=2.9.0
scikit-learn>=1.4.0
```

A `requirements.txt` file is included. Python >= 3.10 required.

---

## What's New in Milestone 3

- **CVE Ransomware-Risk Classifier (Random Forest)** — 13 features, batch EPSS for all KEV CVEs, ROC-AUC ~0.76
- **Temporal Anomaly Detection** — Rolling z-score on KEV monthly additions + ransomware victim counts
- **Cross-Source IOC Correlation** — CCS scoring across 3 abuse.ch feeds
- **Interactive Analytics Panel** — Selectbox + parameter sliders with dynamic chart updates
- **Operational Metrics** — MTTD reduction chart + live precision/recall threshold sweep
- **Validation & Error Analysis** — 70/30 holdout, confusion matrix, assumptions documented
- **12 Analytics Visualizations** — All with captions and source attribution
- **5 Key Insights** — Severity-rated findings with actionable recommendations

---

## App Sections (Sidebar Navigation)

| Section | Description |
|---|---|
| ✅ What's New | Milestone 1 + 2 + 3 change checklists |
| 🏦 Industry Background | GFI overview: services, size, players, IT criticality |
| 👥 Stakeholders & Use Case | 3 personas, 6 user stories, CTI use case design |
| 📈 Threat Trends & Assets | Interactive threat trends, 8 critical assets, exposure matrix |
| 💎 Diamond Models | 2 complete Diamond Models with Plotly visualization + JSON export |
| 📊 Live Dashboard | CISA KEV + EPSS live data with interactive filters and KPIs |
| 💼 Intelligence Buy-In | Breach cost trends, ROI calculator, business case |
| 📡 Data Sources | 7 live sources with justification, collection strategy, metadata |
| 🔍 Data Explorer | 4-tab interactive explorer with cross-source correlations |
| ⚖️ Ethics & Security | Ethics, privacy, security practices, reproducibility |
| 📊 Analytics (M3) | **NEW** — 3 analytical approaches, interactive panel, operational metrics, validation, insights |
| 👨‍💼 Team | Roles, contributions, and electronic signatures |

---

## Data Sources

All data sources are **free, publicly accessible OSINT** (except VirusTotal which requires a free API key):

| Source | URL | Type | Auth | Used In |
|---|---|---|---|---|
| CISA KEV | cisa.gov | JSON (GET) | None | Classifier, Dashboard |
| EPSS | api.first.org | JSON (GET) | None | Classifier (batch query) |
| URLhaus | urlhaus.abuse.ch | CSV (GET) | None | Cross-Source Correlation |
| MalwareBazaar | bazaar.abuse.ch | JSON (POST) | None | Cross-Source Correlation |
| ThreatFox | threatfox.abuse.ch | CSV (GET) | None | Cross-Source Correlation |
| Ransomware.live | api.ransomware.live | JSON (GET) | None | Anomaly Detection |
| SEC EDGAR 8-K | efts.sec.gov | JSON (GET) | None (User-Agent required) | Data Sources |
| VirusTotal | virustotal.com/api/v3 | JSON (GET) | Free API key | Data Sources |

No external data files are required. All data is fetched live from APIs at runtime and cached via `@st.cache_data(ttl=N)`. Fallback data is hardcoded for all sources to ensure live demo reliability.

---

## Analytics Methodology (Milestone 3)

### Approach 1: CVE Ransomware-Risk Classification

- **Algorithm:** Random Forest (scikit-learn) with `class_weight="balanced"`
- **Features (13):** EPSS score, EPSS percentile, days in KEV, remediation window, vendor criticality, product frequency, and 7 vulnerability-type keyword flags (RCE, privilege escalation, buffer overflow, auth bypass, deserialization, path traversal, SQL injection)
- **Target:** `knownRansomwareCampaignUse` from CISA KEV (binary: Known vs Unknown)
- **EPSS coverage:** Batch-queries EPSS API for all ~1,579 KEV CVE IDs (100% coverage)
- **Evaluation:** Confusion matrix, precision, recall, F1, ROC-AUC (~0.76), feature importance
- **User controls:** Test size, tree count, max depth, classification threshold

### Approach 2: Temporal Anomaly Detection

- **Method:** Rolling z-score (semi-supervised point anomaly detection)
- **Data:** KEV `dateAdded` aggregated monthly; Ransomware.live daily victim counts
- **User controls:** Z-score threshold, rolling window size

### Approach 3: Cross-Source IOC Correlation

- **Method:** Set intersection of malware family names across 3 abuse.ch feeds
- **Scoring:** Compound Confidence Score (CCS) — 1 (single-source) to 3 (tri-source)
- **Data:** URLhaus tags, ThreatFox families, MalwareBazaar signatures

---

## Folder Structure

```
GFI-CTI-Platform/
├── milestone3_app.py          # Main Streamlit app (Milestones 1 + 2 + 3)
├── milestone2_app.py          # Milestone 2 reference version
├── milestone1_app.py          # Milestone 1 reference version
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .streamlit/
    ├── config.toml            # Dark theme configuration
    └── secrets.toml           # VirusTotal API key (gitignored)
```

---

## How to Reproduce Analysis

1. Python >= 3.10 and `pip` required
2. Clone the repo or extract the ZIP
3. `pip install -r requirements.txt`
4. `streamlit run milestone3_app.py`
5. Open `http://localhost:8501` — all data fetched live (cached 30-120 min)
6. Navigate to **Analytics (M3)** page to see all three analytical approaches
7. VirusTotal (optional): place API key in `.streamlit/secrets.toml` as `VT_API_KEY = "your_key"`

---

## Milestone Roadmap

| Milestone | Due | Focus |
|---|---|---|
| M1 | March 26, 2026 | Industry background, threat trends, diamond models, dashboard starter |
| M2 | April 9, 2026 | 7 data sources, collection strategy, data explorer, ethics, security |
| **M3 (current)** | **April 23, 2026** | **3 analytical approaches, interactive panel, visualizations, insights** |
| M4 | April 30, 2026 | Final polish, triage dashboard, role-based views, STIX export |

---

## Team

| Name | Role |
|---|---|
| **Devansh Agarwal** | Project Coordinator, Streamlit App Lead |
| Anica | Data Collection & Pipeline Engineer |
| Noreen | Threat Modeling & Intelligence Research |
| Guled | Analytics & Validation Lead |
| Ville | Visualizations & Dashboard Design |

---

*Submitted for CIS 8684 — Cyber Threat Intelligence · Spring 2026 · Georgia State University*

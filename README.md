# ⚽ Machine Learning Scouting Framework for Professional Football
### Identifying Undervalued U23 Talent Across Secondary European Leagues via Gaussian CDF & Domain-Adapted ML

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Tableau](https://img.shields.io/badge/Tableau-Interactive%20Dashboards-E97627.svg)](https://public.tableau.com/)

---

## 📌 Executive Summary
Traditional player recruitment often suffers from cognitive heuristics, eye-test bias, and market price inflation tied to league reputation rather than underlying athletic output.

This project delivers an end-to-end, data-driven talent identification pipeline evaluating **U23 prospects across 22 secondary European leagues** using Wyscout event metrics. Rather than predicting volatile market transfer values, the model isolates **latent on-pitch efficiency** and projects it into a target league benchmark.

### 🏆 Out-of-Time Backtesting & Business KPIs (Summer 2025 – April 2026)
* **21.2% Transfer Hit-Rate:** 14 out of 66 shortlisted prospects achieved immediate career-tier moves to higher divisions.
* **+297.2% Portfolio Value Appreciation:** Aggregate market valuation of the identified talent pool grew from **€29.55M** to **€117.37M**.
* **92.5% Capital Preservation:** Successfully filtered out overhyped, high-cost academy profiles that subsequently declined in value.
* **5-Month Market Lead:** The framework identified breakout athletic markers on average 5 months prior to public transfer market revaluations.

---

## ⚙️ Methodology & Mathematical Architecture

### 1. Positional Feature Weighting via Decision Ensembles
To avoid arbitrary manual weighting, positional metric importance is determined algorithmically using a **Random Forest Classifier** trained on target league positional elite benchmarks.
* Metrics are segmented into functional positional buckets (*Threat, Playmaking, Carrying, Defense, Build-up*).
* Decouples raw statistical volume from tactical fit, capturing systemic requirements while smoothing out single-player statistical noise (Synthetic Elite DNA).

### 2. Cross-League Translation Factors
Raw statistical output across distinct competitions cannot be evaluated uniformly. Difficulty translation incorporates both athletic ranking and financial depth:
$$W_{\text{league}} = \sqrt{0.75 \times \text{NormRank}_{\text{Opta}} + 0.25 \times \text{NormMV}_{\text{MarketValue}}}$$
$$\text{Final Weight} = 0.50 + \left(0.50 \times W_{\text{league}}\right)$$
* **0.50 Baseline:** Preserves intrinsic individual attributes (isolated 1v1 dribbling, raw physical dueling, baseline acceleration).
* **0.50 Scaled Component:** Adjusts collective, opponent-dependent tactical production.

### 3. The Prospect Rating (Gaussian CDF Formulation)
Comparing candidates against aggregate statistical maximums creates the *Frankenstein Paradox* (an unachievable composite player). To reflect genuine population distribution, standardized positional performance ($Z$-scores) scaled by league difficulty is mapped via the standard normal Cumulative Distribution Function (CDF):
$$\text{Prospect Rating} = \Phi\left(GEM_{\text{raw}} \times \text{Coefficient}_{\text{League}}\right) \times 100$$
Prospects exceeding the **90th percentile threshold** ($\ge$ top decile of the target tier) are routed to the final recruitment shortlist.

---

## 📊 Shortlist Showcase & Visual Intelligence
Candidates are visualized through dual-layered radar architectures (**Raw Domestic Dominance** vs. **Adjusted Target-League Equivalency**):

* **Tadeáš Vachoušek** (Zbrojovka Brno | Attacking Midfielder | Prospect Rating: **86.365**) – Elite goal conversion (0.70 goals/90 on 0.33 xG) & key passing volume.
* **Álvaro Cortés** (Barcelona Atlètic | Central Defender | Prospect Rating: **82.741**) – Ball-playing center-back (92.5% pass accuracy, 10 progressive passes/90).
* **Pedro Brazão** (Bodrumspor | Attacking Midfielder | Prospect Rating: **84.448**) – Dynamic carrier and transition threat (3.22 progressive runs/90).
* **Niv Yehoshua** (Maccabi Petah Tikva | Central Midfielder | Prospect Rating: **82.205**) – Deep-lying playmaker with elite line-breaking vision.

> 🔗 **Interactive Tableau Dashboards:**  https://public.tableau.com/app/profile/luk.p.bi./vizzes

---

## 🛠️ Pipeline Architecture & Scripts
* `src/data_processing.py`: Data ingestion, schema standardization, positional filtering (U23, min. 900 minutes).
* `src/correlation_matrix.py`: Multicollinearity screening ($r > 0.75$) to eliminate metric redundancy.
* `src/model.py`: Random Forest feature importance extraction, league normalization, and Prospect Rating CDF computation.
* `src/visualization.py`: Data transformation into long-format schema for dynamic radar dashboards.
* `src/business_validation.py`: Business KPI tracking (Arbitrage Index, Capital Protection, Portfolio ROI).
* `src/validation_spain.py`: External model calibration against tier-one La Liga data.

---

## 📬 Contact & Author
**Bc. Lukáš Pôbiš**  
*Master's Degree Candidate in Applied Data Analytics and AI – Prague University of Economics and Business (VŠE)*  
* **LinkedIn:** [linkedin.com/in/lukaspobis](https://www.linkedin.com/in/lukaspobis)  
* **Email:** pobisluk@gmail.com
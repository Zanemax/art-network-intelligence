# Art Network Intelligence

Art Network Intelligence is an MVP for exploring whether graph-derived art world
signals can help predict future artist success. It combines artist, gallery,
museum, collector, curator, auction, fair, and press data into a temporal taste
graph, extracts point-in-time artist features, trains baseline and machine
learning models, and presents the results in a Streamlit product demo.

This repository is for research, product prototyping, and technical validation.
It is not investment advice. All predictions are experimental and should not be
used for real investment decisions.

## Problem

Emerging and mid-career artist evaluation often depends on scattered qualitative
signals: gallery representation, institutional exposure, collector networks,
curator attention, auction momentum, press visibility, and proximity to major
art-world institutions.

The core question for this MVP is:

Can those signals be structured as a temporal graph and tested historically
against future artist outcomes?

The project is designed to make that question testable, auditable, and easy to
demo before investing in a larger data operation.

## MVP Scope

Current capabilities:

- Synthetic art market dataset for local development.
- Production-ready raw CSV schemas for real data intake.
- Manual research template for entering the first 50 real artists.
- NetworkX temporal graph with typed nodes and dated relationship edges.
- Artist-level graph feature extraction as of a prediction date.
- Target labels for 3-year institutional, market, and gallery success.
- Baseline rankers and scikit-learn classifiers.
- Walk-forward historical backtesting.
- Similar artist search based on graph-derived career trajectories.
- Plain-English prediction explanations and data-quality warnings.
- Streamlit product demo with artist profiles, graph exploration, model results,
  similar artists, and data quality views.

Out of scope for this MVP:

- Production-grade data pipelines.
- Paid data-provider integrations.
- Human-reviewed investment recommendations.
- Portfolio construction or financial optimization.
- Claims of predictive validity beyond the included toy/synthetic data.

## Data Model

The graph is built from normalized CSV tables:

- `artists`
- `galleries`
- `museums`
- `institutions`
- `collectors`
- `curators`
- `events`
- `relationships`
- `auction_results`
- `press_mentions`

Production templates live in:

```text
data/raw/templates/
```

Each entity has a stable ID. Each relationship is timestamped and includes
source metadata:

- `source_url`
- `confidence_score`
- `notes`

The data dictionary is in:

```text
docs/data_dictionary.md
```

## Taste Graph

The taste graph is a typed temporal graph built with NetworkX.

Node examples:

- artist
- gallery
- museum
- collector
- curator
- exhibition
- acquisition
- auction result
- press mention

Relationship examples:

- gallery represents artist
- artist included in exhibition
- museum acquired artist
- collector collects artist
- curator curated artist
- artist has auction result
- artist mentioned in press

Every graph edge supports temporal reasoning with:

- `source_id`
- `target_id`
- `relationship_type`
- `start_date`
- `end_date`
- `source_url`
- `confidence_score`
- `notes`

This allows the system to build graph snapshots as of historical prediction
dates and avoid using future information in features.

## Temporal Prediction

Temporal prediction is structured around artist-date rows:

1. Choose a prediction date.
2. Build a graph snapshot as of that date.
3. Extract artist features using only information available up to that date.
4. Generate 3-year success labels from the future window after that date.
5. Train on earlier prediction dates.
6. Test on later prediction dates.

Supported target labels:

- `institutional_success_3y`: major museum acquisition, major solo show, or
  major biennial inclusion within 3 years.
- `market_success_3y`: auction median price increases by at least 2x within
  3 years.
- `gallery_success_3y`: artist moves to a higher-tier gallery within 3 years.

Supported model families:

- Logistic Regression
- Random Forest
- Gradient Boosting

Baseline comparisons:

- random baseline
- museum count baseline
- gallery prestige baseline
- simple weighted score baseline

## Dashboard

Run the Streamlit product demo:

```bash
streamlit run src/app/streamlit_app.py
```

Dashboard pages:

- Artist Search
- Artist Profile
- Taste Graph Explorer
- Similar Artists
- Model Evaluation
- Data Quality

The Artist Profile page includes:

- breakout probability
- institutional success probability
- market success probability
- main drivers
- timeline of career events
- current gallery
- museum exposure
- collector and curator network signals
- similar artists
- data-quality warnings

## Add Real Artist Data

Use the manual research workflow for the first 50 real artists.

Start with:

```text
data/raw/templates/artist_research_template.csv
```

Research guidance:

```text
docs/manual_research_guide.md
docs/source_priority.md
```

The guide explains how to collect:

- artist bio
- gallery representation
- museum exhibitions
- museum acquisitions
- art fair appearances
- auction history
- press mentions
- collector and curator signals
- source URLs
- confidence scores

Convert filled research rows into normalized CSVs:

```bash
python -m src.data.import_research_template data/raw/templates/artist_research_template.csv
```

Output is written to:

```text
data/raw/imported/
```

## Backtesting

Run annual walk-forward backtests:

```bash
python -m src.models.backtest --target institutional_success_3y --start-year 2017 --end-year 2023
```

Outputs:

```text
reports/backtest_results.csv
reports/backtest_summary.md
```

The backtest trains only on prediction dates before the tested date and evaluates
future outcomes using the target's 3-year label window.

## Training

Train temporal models with an explicit train/test date split:

```bash
python -m src.models.train --target institutional_success_3y --train-end-date 2021-12-31 --test-start-date 2022-01-01
```

Outputs:

```text
reports/metrics.json
reports/predictions.csv
reports/data_quality.csv
artifacts/model.joblib
```

## Similar Artists

Find similar artists based on normalized graph-derived features:

```bash
python -m src.models.similarity --artist-id artist_ada_rios --as-of-date 2021-12-31 --top-n 5
```

The result includes similarity score, later outcomes, and shared signals.

## Data Quality

The data quality layer calculates:

- missing required fields
- stale data flags
- low confidence flags
- conflicting identity flags
- insufficient history flags
- source count
- average confidence score

If data quality is poor, explanations warn:

> Prediction confidence is low because this artist has sparse or low-confidence source data.

## Project Structure

```text
data/
  raw/
    templates/       Raw schema and manual research templates.
    imported/        Normalized CSVs generated from manual research rows.
  processed/         Future location for cleaned production datasets.
  synthetic/         Synthetic data used for MVP development and tests.

docs/
  data_dictionary.md
  manual_research_guide.md
  source_priority.md

src/
  app/               Streamlit product demo.
  data/              Synthetic data, validation, import, and quality checks.
  graph/             Temporal graph construction and graph feature extraction.
  models/            Targets, baselines, training, backtesting, explanations.

tests/               Automated tests.
reports/             Generated metrics, predictions, quality, and backtests.
artifacts/           Trained model artifacts.
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

If using macOS system Python in a restricted environment:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache-art-network pytest
```

## Limitations

- The included dataset is synthetic and intentionally small.
- Current model results are not evidence of real-world predictive performance.
- Real data coverage, identity resolution, and source verification are not yet
  production-grade.
- Confidence scores are manually assigned or synthetic placeholders.
- Auction history can be sparse, biased, delayed, or unavailable.
- Institutional and market success labels are simplified approximations.
- The system does not account for macro conditions, genre cycles, geography,
  primary-market pricing, private sales, or survivorship bias.
- Similarity and explanation outputs are designed for interpretability, not
  causal proof.

## Roadmap

Near-term:

- Enter and validate the first 50 real artists.
- Improve identity resolution for artists, galleries, museums, collectors, and
  curators.
- Add richer source provenance and reviewer status.
- Add model cards for each target and baseline.
- Improve Streamlit visual polish and navigation.

Modeling:

- Expand historical backtesting with real data.
- Add calibration curves and threshold analysis.
- Add fold-level feature importance reports.
- Compare target definitions across multiple windows.
- Add confidence intervals or uncertainty bands.

Data:

- Add ingestion from curated public datasets.
- Add deduplication workflows.
- Add source freshness monitoring.
- Track researcher, review date, and evidence quality.

Product:

- Add saved artist watchlists.
- Add exportable artist memos.
- Add side-by-side artist comparison.
- Add graph path explanations from artist to institution, gallery, collector, or
  curator.

## Disclaimer

This project is an experimental MVP. It is not investment advice, financial
advice, appraisal advice, or a recommendation to buy or sell artwork. Predictions
are experimental outputs from incomplete data and should be reviewed critically
by qualified domain experts before any real-world use.

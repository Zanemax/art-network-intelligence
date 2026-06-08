# Milestone: Replace Synthetic Data With 50 Real Artists

## Goal

Replace the synthetic MVP dataset with a manually researched dataset of 50 real
artists in one clearly defined niche, then validate whether the graph, feature,
baseline, ML, similarity, explanation, and backtesting workflows still run
end-to-end.

This milestone is for data and product validation only. It is not investment
advice, and model outputs remain experimental.

## Checklist

- [ ] Choose niche
  - Define the market segment, geography, medium, career stage, and time period.
  - Example: emerging painters represented by mid-tier European galleries.

- [ ] Define artist selection criteria
  - Decide inclusion and exclusion rules before collecting data.
  - Record why each artist belongs in the cohort.
  - Avoid cherry-picking only known winners.

- [ ] Research 50 artists
  - Use `docs/manual_research_guide.md`.
  - Use `docs/source_priority.md`.
  - Track source URLs and confidence scores for every populated section.

- [ ] Fill templates
  - Enter rows in `data/raw/templates/artist_research_template.csv`.
  - Repeat `artist_id` across rows when an artist has multiple events, auction
    results, press mentions, collectors, or curators.

- [ ] Validate data
  - Import manual research rows:

    ```bash
    python -m src.data.import_research_template data/raw/templates/artist_research_template.csv
    ```

  - Confirm normalized CSVs are written to `data/raw/imported/`.
  - Resolve validation errors before modeling.

- [ ] Build graph
  - Confirm normalized entities and relationships can be loaded into the graph.
  - Check stable IDs, relationship dates, source URLs, and confidence scores.

- [ ] Generate features
  - Build artist-level features as of selected prediction dates.
  - Confirm features only use data available before each prediction date.

- [ ] Run baselines
  - Compare simple rankers before running ML:

    ```bash
    python -m src.models.evaluate_baselines --target institutional_success_3y
    ```

- [ ] Run ML model
  - Train temporal models with a clear train/test split:

    ```bash
    python -m src.models.train --target institutional_success_3y --train-end-date 2021-12-31 --test-start-date 2022-01-01
    ```

- [ ] Inspect similar artists
  - Spot-check similarity output for at least 10 artists:

    ```bash
    python -m src.models.similarity --artist-id ARTIST_ID --as-of-date 2021-12-31 --top-n 5
    ```

- [ ] Produce first backtest
  - Run annual walk-forward backtesting:

    ```bash
    python -m src.models.backtest --target institutional_success_3y --start-year 2017 --end-year 2023
    ```

  - Review `reports/backtest_results.csv`.
  - Review `reports/backtest_summary.md`.

- [ ] Document findings
  - Summarize data coverage.
  - Summarize validation issues.
  - Compare graph model results against baselines.
  - Identify weak features and missing sources.
  - Record whether the MVP is ready for the next 100-artist expansion.

## Acceptance Criteria

The milestone is complete when:

- [ ] A niche and artist selection rubric are documented.
- [ ] 50 real artists have stable `artist_id` values.
- [ ] Manual research rows exist for all 50 artists.
- [ ] Every populated row includes relevant `source_url` and `confidence_score`
      fields.
- [ ] Manual research rows successfully import into normalized CSVs in
      `data/raw/imported/`.
- [ ] Normalized CSVs pass schema validation.
- [ ] The graph can be built from the imported real dataset.
- [ ] Artist features can be generated for all 50 artists.
- [ ] Baseline evaluation runs without errors.
- [ ] Temporal ML training runs without errors.
- [ ] Similar artist search returns plausible comparable artists for sampled
      artists.
- [ ] A first historical backtest report is generated.
- [ ] `reports/data_quality.csv` is generated and reviewed.
- [ ] Known data gaps, source limitations, and model limitations are documented.
- [ ] README or project docs clearly state that outputs are experimental and not
      investment advice.

## Deliverables

- `data/raw/templates/artist_research_template.csv` filled with 50 real artists.
- Normalized CSVs in `data/raw/imported/`.
- `reports/data_quality.csv`.
- `reports/metrics.json`.
- `reports/predictions.csv`.
- `reports/backtest_results.csv`.
- `reports/backtest_summary.md`.
- Written findings document or issue comment summarizing what was learned.

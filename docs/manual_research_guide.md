# Manual Research Guide

Use `data/raw/templates/artist_research_template.csv` to enter the first 50 real
artists. Each row can represent one sourced observation for an artist. Repeating
the same `artist_id` across rows is expected when an artist has multiple shows,
auction results, press mentions, collectors, or curators.

## Workflow

1. Assign a stable `artist_id` before research begins. Use a readable ID such as
   `artist_jane_doe`.
2. Fill the artist bio fields from the strongest available source.
3. Add one observation per row. If a row only covers a museum show, leave auction
   and press fields blank.
4. Always include source URLs for populated sections.
5. Use ISO dates: `YYYY-MM-DD`.
6. Use confidence scores from `0.0` to `1.0`.
7. Run the importer:

```bash
python -m src.data.import_research_template data/raw/templates/artist_research_template.csv
```

Normalized CSVs will be written to `data/raw/imported/`.

## Semi-Automated Candidate Workflow

Use this workflow when you want help finding possible observations, but still
want a human researcher to decide what enters the graph.

1. Add artists to `data/raw/manual/artist_seed_list.csv`.
2. Generate candidate observations:

```bash
python -m src.data.research.generate_candidates
```

3. Review `data/raw/candidates/candidate_observations.csv`.
4. Set `accepted=yes` only for sourced observations you trust.
5. Promote accepted rows into manual research format:

```bash
python -m src.data.research.promote_accepted_candidates
```

6. Import the promoted manual CSV:

```bash
python -m src.data.import_research_template data/raw/manual/accepted_artist_research_template.csv
```

Candidate generation is intentionally conservative. It preserves existing
review decisions and does not push data into the graph. Promotion is append-only
by `candidate_id`, so rerunning it will not duplicate already promoted rows.

Candidate rows include two review fields:

- `needs_human_review_reason`: why the row is not safe to ingest automatically.
- `why_matched`: why the source was matched to the seed artist.

Supported candidate observation types include `artist_profile`,
`gallery_representation`, `museum_exhibition`, `museum_acquisition`,
`auction_result`, and `press_mention`. Unclear or likely false Wikidata matches
are kept as low-confidence `rejected_match` rows for transparency, not promoted
unless a researcher explicitly changes them and accepts them.

## Artist Bio

Collect `artist_name`, `birth_year`, `death_year`, `nationality`, `gender`,
`primary_medium`, `artist_website_url`, `bio_source_url`, and
`bio_confidence_score`.

Prefer the artist website, gallery profile, museum biography, or a recognized
artist database. Keep `artist_id` stable even if display names change.

## Gallery Representation

Collect gallery name, city, country, tier, prestige score, representation start
date, source URL, and confidence score.

Use `gallery_tier` values such as `emerging`, `mid`, `major`, `top`, or `mega`.
Set `gallery_prestige_score` from `0.0` to `1.0`.

## Museum Exhibitions

For museum shows, fill `museum_name`, `museum_city`, `museum_country`,
`museum_tier`, `museum_event_type`, `event_name`, event dates, source URL, and
confidence score.

Use event types such as `solo_show`, `group_show`, `major_solo_show`,
`biennial_inclusion`, or `museum_exhibition`.

## Museum Acquisitions

For acquisitions, fill `museum_name`, `acquisition_date`,
`acquisition_value_usd` when known, source URL, and confidence score.

Prefer accession pages, museum collection pages, annual reports, or official
press releases.

## Art Fair Appearances

Fill `art_fair_name`, `art_fair_date`, city, country, source URL, and confidence
score. Use official fair exhibitor pages or gallery booth announcements.

## Auction History

Fill auction house, sale name, lot number, sale date, work title, medium,
creation year, estimates, price, currency, source URL, and confidence score.

Use realized price where available. If only estimates are available, leave
`price_usd` blank.

## Press Mentions

Fill outlet, article title, author, publication date, article URL, mention
count, sentiment score, and confidence score.

Use `mention_count` for the number of meaningful mentions in the article. Use
`sentiment_score` from `-1.0` to `1.0`, where `0` is neutral.

## Collector And Curator Signals

Fill collector and curator names only when there is a source confirming the
relationship. Examples include collection pages, exhibition pages, fair texts,
catalogues, interviews, or press releases.

Do not infer private ownership unless the source says so.

## Source URLs

Every populated section should include the strongest source URL available. If a
row has both an auction result and a press mention, include both section-specific
URLs.

## Confidence Scores

- `1.0`: official primary source, accession page, gallery page, auction house
  sale result, museum exhibition page.
- `0.8`: reputable secondary source with clear evidence.
- `0.6`: credible press or database entry without primary confirmation.
- `0.4`: weak source, incomplete details, or ambiguous identity.
- Below `0.4`: avoid using unless clearly marked in notes.

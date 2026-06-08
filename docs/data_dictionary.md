# Data Dictionary

Raw production data should be loaded into files matching the templates in
`data/raw/templates/`. Use snake_case column names, stable IDs, dated
relationships, source URLs, and confidence scores.

## Shared Conventions

- IDs must be stable, unique within their file, and never derived from row order.
- Relationship and event records must include dates using ISO format:
  `YYYY-MM-DD`.
- `source_url` should point to the evidence source for the row.
- `confidence_score` is a float from `0.0` to `1.0` describing source confidence.
- Empty optional values are allowed, but required ID, date, source, and confidence
  fields should be populated for production ingestion.

## artists.csv

Artist master records.

Required columns: `artist_id`, `canonical_name`, `birth_year`, `death_year`,
`nationality`, `gender`, `primary_medium`, `artist_website_url`, `source_url`,
`confidence_score`, `notes`.

## galleries.csv

Gallery master records, linked to the general institution table when possible.

Required columns: `gallery_id`, `institution_id`, `canonical_name`, `city`,
`country`, `founded_year`, `prestige_score`, `source_url`, `confidence_score`,
`notes`.

## museums.csv

Museum master records, linked to the general institution table when possible.

Required columns: `museum_id`, `institution_id`, `canonical_name`, `city`,
`country`, `founded_year`, `tier`, `prestige_score`, `source_url`,
`confidence_score`, `notes`.

## collectors.csv

Collector records for people, families, foundations, or corporate collections.

Required columns: `collector_id`, `display_name`, `collector_type`, `city`,
`country`, `visibility_level`, `source_url`, `confidence_score`, `notes`.

## curators.csv

Curator records with optional institutional affiliation.

Required columns: `curator_id`, `display_name`, `affiliated_institution_id`,
`city`, `country`, `source_url`, `confidence_score`, `notes`.

## events.csv

Exhibitions, fairs, biennials, talks, residencies, and other dated art-world
events.

Required columns: `event_id`, `event_type`, `event_name`, `institution_id`,
`artist_id`, `curator_id`, `start_date`, `end_date`, `event_date`, `city`,
`country`, `source_url`, `confidence_score`, `notes`.

## relationships.csv

Graph edge records between any two known nodes.

Required columns: `relationship_id`, `source_node_id`, `source_node_type`,
`target_node_id`, `target_node_type`, `relationship_type`, `relationship_date`,
`start_date`, `end_date`, `source_url`, `confidence_score`, `notes`.

## auction_results.csv

Dated auction sale results and estimates.

Required columns: `auction_result_id`, `artist_id`, `auction_house`,
`sale_name`, `lot_number`, `sale_date`, `work_title`, `medium`,
`creation_year`, `estimate_low_usd`, `estimate_high_usd`, `price_usd`,
`currency`, `source_url`, `confidence_score`, `notes`.

## press_mentions.csv

Article-level press records and mention counts.

Required columns: `press_mention_id`, `artist_id`, `outlet_name`,
`article_title`, `author`, `publication_date`, `url`, `mention_count`,
`sentiment_score`, `source_url`, `confidence_score`, `notes`.

## institutions.csv

Shared institution registry for museums, galleries, auction houses, fairs,
foundations, universities, publications, and other organizations.

Required columns: `institution_id`, `institution_type`, `canonical_name`,
`city`, `country`, `founded_year`, `tier`, `prestige_score`, `source_url`,
`confidence_score`, `notes`.

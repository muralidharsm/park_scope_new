# Where to put the datasets

PARK-SCOPE reads the two CSVs provided by HackerEarth. Drop them into this
`data/` folder exactly as you downloaded them — the loader auto-detects them by
pattern (`*police_violation*.csv` and `*Astram*event*.csv`), so the hash
suffixes in the filenames are fine. If you rename them, update the names in
`src/config.py`.

Expected (either form works):

    data/jan_to_may_police_violation_anonymized791b166.csv
    data/Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv

These files are NOT committed to the repo (see `.gitignore`) because they are
large and not ours to redistribute. The pipeline turns them into the small
artifacts in `artifacts/`, which is what the dashboard actually reads.

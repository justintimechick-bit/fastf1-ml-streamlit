Real source captures acquired on 8 November 2026, before the Brazil race.

- `schedule.json`: FastF1's 2026 event schedule.
- `results.parquet` and `results-metadata.json`: Brazil qualifying via FastF1/Jolpica.
- `index.html` and `event.html`: the FIA season index and Brazil event document page.
- `roster.txt` and `qualifying.txt`: text extracted by the existing FIA parser from
  the official entry-list and final-qualifying PDFs. These are source observations,
  including the Rafael Camara substitution, rather than generated test records.
- `sources.json`: source URLs, retrieval times and original response SHA-256 hashes.
  PDF hashes identify the original PDF bytes, not the extracted text files.

Tests replace transport and PDF extraction with these captures, retaining the real
calendar, document-index, entrant, identity and classification parsing. The clock
is fixed before race start so the tests remain reproducible after the weekend.

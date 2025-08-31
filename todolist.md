# Tool Accuracy Improvement Checklist

Legend: [ ] todo, [~] investigate/spike, [x] done

## Priority Tools
- [ ] Rank order (most impact):
  1) ingest_csv
  2) detect_recurring
  3) export_clean_monthly_recurring
  4) analyze_subscription_value
  5) predict_next_recurring
  6) ingest_excel
  7) suggest_budgets / analyze_spending_patterns
  8) generate_tax_report
  9) export_transactions
  10) ingest_pdfs

## ingest_csv (ING Dutch CSVs)
- [ ] Add duplicate-ingestion guard (hash on date+amount+description+source_file) to prevent double counts
- [ ] Preserve and parse IBAN from Mededelingen into a normalized token for downstream grouping (consider new column or side table)
- [ ] Normalize category from `Mutatiesoort` plus heuristic merchant-based categories (e.g., Albert Heijn→groceries, Vodafone/Odido→telecom, Essent→utilities, VGZ→insurance, Klarna→loans)
- [ ] Unit tests against `documents/NL22INGB0669807419_01-01-2025_23-08-2025.csv` verifying row count, signs (Af/Bij), date parsing (%Y%m%d), and key merchant categorizations

## detect_recurring
- [ ] Fix frequency misclassification (monthly patterns detected as daily)
  - [ ] Prefer month-based detection first; de-duplicate multiple occurrences within the same month before daily/weekly checks
  - [ ] Compute intervals on unique months; only fall back to day-interval heuristics if month-based detection fails
- [ ] Tighten normalization for Dutch bank descriptors (retain merchant, drop labels like “Naam:”, “Omschrijving:”, “Kenmerk:”, keep first meaningful 3–4 words)
- [ ] Include amount clustering with adaptive tolerance per merchant type (utilities slightly higher, card fees lower)
- [ ] Calibrate confidence scoring; reward consistency of month presence and payment day; penalize irregular intra-month clusters
- [ ] Regression test against sample docs (expect “ING creditcard”, VGZ, Essent, Vodafone/Odido, NN Schadeverzekering, Stichting BrabantWonen) to classify monthly (not daily)

## export_clean_monthly_recurring
- [ ] Ensure only truly monthly items are included (align with fixed detection)
- [ ] Improve merchant naming in `payment_name` using refined normalization (e.g., “Essent Retail Energie”, “VGZ Zorgverzekeraar”)
- [ ] Add cross-check: months_active ≥ N and consistent avg day-of-month window
- [ ] Snapshot tests comparing output with current CSVs in `documents/recurring_payments_*.csv`

## analyze_subscription_value
- [ ] Align selection threshold with new confidence scale (e.g., default ≥ 0.7)
- [ ] Incorporate recent usage by matching IBAN/normalized merchant where available instead of loose substring
- [ ] Add “potential savings” estimation by toggling low-value subscriptions
- [ ] Tests on dataset to ensure high-cost entertainment flagged; essentials (utilities/insurance) marked keep

## predict_next_recurring
- [ ] Use avg day-of-month (reuse `_calculate_avg_day`) for monthly predictions instead of fixed 30 days
- [ ] Advance by calendar months (with month-end handling), not fixed day deltas
- [ ] Confidence-aware horizon (only predict if confidence ≥ threshold)
- [ ] Verify predictions align with sample recurring patterns

## ingest_excel
- [ ] Support ING-like sign column (“Af Bij”) and separated debit/credit; confirm on `documents/XLS250827222000.xls`
- [ ] Add duplicate-ingestion guard same as CSV
- [ ] Expand auto-mapping for Dutch headers (e.g., Mutatiesoort, Naam/Omschrijving)
- [ ] Tests on provided XLS to validate row count, signs, and parsing

## suggest_budgets / analyze_spending_patterns
- [ ] Use refined categories (from merchant heuristics) to improve insight quality vs coarse Mutatiesoort values
- [ ] Guard against outliers (single large purchases) skewing suggestions
- [ ] Add “protected categories” default list tuned to dataset (e.g., rent/utilities/insurance)
- [ ] Snapshot tests on dataset: top categories, monthly trends, daily-of-week breakdowns

## generate_tax_report
- [ ] Enrich tax category mapping using description-based heuristics (e.g., insurance, charity, medical keywords in Dutch; map to tax buckets)
- [ ] Allow user overrides via a simple rules config (merchant→tax_category)
- [ ] Tests: transactions from CSV mapped away from generic `other_deductions` when clear keywords exist

## export_transactions
- [ ] Add option to de-duplicate or tag duplicates (if ingestion run multiple times)
- [ ] Verify exported sums match DB queries on dataset

## ingest_pdfs (AFSCHRIFT*.pdf, mutov*.pdf)
- [ ] Use `pdfplumber` fallback automatically; add optional OCR (tesseract) path for scanned statements
- [ ] Provide “analyze-only vs import-transactions” modes (pipe to `document_agents.PDFAnalyzer` and insert parsed rows)
- [ ] Tests: extract metadata and at least stable snippet counts for sample PDFs

## Cross‑cutting
- [ ] Add unique index or constraint strategy to avoid duplicates (schema or logic-level)
- [ ] Performance: ensure indices used by new queries (recurring, budgets) remain efficient
- [ ] Document merchant/category heuristics and provide an override file (YAML/JSON) in `documents/` for user tuning
- [ ] Add CLI command to run validation suite on `documents/` (smoke checks + counts)


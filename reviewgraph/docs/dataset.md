# Dataset Documentation & Data Dictionary

## 1. Primary Dataset

**Name:** fake-reviews-data  
**Source:** https://github.com/bretthollenbeck/fake-reviews-data  
**Format:** CSV  
**License:** See repository for license terms. Verify before use.

### Inspection Instructions

Before running any pipeline scripts, inspect the actual dataset:

```bash
cd reviewgraph
python scripts/inspect_dataset.py --dataset-path data/raw/fake_reviews.csv
```

This will print:
- Column names and sample values
- Detected label column and its unique values
- Missing value counts
- Statistical summary

**Always verify the output of `inspect_dataset.py` before making assumptions about column names or label meanings.**

---

## 2. Expected Column Mapping

The dataset loader auto-detects columns using the following alias table. Verify after running `inspect_dataset.py`.

| Raw Column (possible names) | Normalized Name | Type | Description | Notes |
|---|---|---|---|---|
| `id`, `review_id`, `reviewID` | `review_id` | string | Unique review identifier | Generated hash if missing |
| `reviewerID`, `reviewer_id`, `user_id` | `reviewer_id` | string | Reviewer identifier | Generated hash if missing |
| `asin`, `product_id`, `productID`, `item_id` | `product_id` | string | Product identifier | Generated hash if missing |
| `reviewText`, `review_text`, `reviewContent`, `text`, `comment` | `review_text` | string | Raw review text | Rows with NaN kept but flagged |
| `overall`, `rating`, `stars`, `score` | `rating` | float | Star rating (1–5) | NaN rows kept |
| `unixReviewTime`, `date`, `timestamp`, `reviewTime` | `timestamp` | string/int | Review date | Parsed where possible |
| `label`, `fake`, `class`, `category`, `flagged` | `label` | int (0/1) | Suspiciousness label | See Section 3 |

---

## 3. Label Definition

### bretthollenbeck/fake-reviews-data

Expected label values and their meaning:

| Raw Value | Normalized | Meaning |
|---|---|---|
| `CG` | `1` | **Computer Generated** — identified as algorithmically generated; treated as **suspicious** |
| `OR` | `0` | **Original** — human-written review; treated as **genuine** |
| `1`, `fake`, `True` | `1` | Suspicious |
| `0`, `genuine`, `False` | `0` | Genuine |

### ⚠️ Critical Limitation

> The label `CG` indicates **computer-generated text**, not **verified fraud**. A computer-generated review is not automatically fraudulent, and a human-written review may still be manipulative. The system treats `CG` as a **suspiciousness signal**, not as verified ground truth.

The system explicitly labels classifier outputs as `model_score`, never as `fraud_probability`.

---

## 4. Preprocessing Steps

1. Load CSV with automatic column detection
2. Remove fully duplicate rows (exact match across all columns)
3. Generate stable hash-based IDs for records missing `review_id`, `reviewer_id`, or `product_id`
4. Normalize label column using the mapping above (NaN labels remain NaN — never auto-filled)
5. Compute text features: `review_length`, `word_count`, `char_count`, `avg_word_length`
6. Compute reviewer statistics: `reviewer_review_count`, `reviewer_distinct_products`
7. Compute product statistics: `product_avg_rating`, `product_review_count`
8. Flag rows with missing `review_text`
9. Save processed output to `data/processed/reviews_processed.csv`

---

## 5. Missing Value Handling Policy

| Column | Policy |
|---|---|
| `review_text` | Flag as `text_missing=True`; keep row; exclude from text features |
| `rating` | Keep NaN; exclude from rating features |
| `timestamp` | Keep NaN; exclude from temporal features |
| `label` | **Never fill.** Keep NaN; report count of unlabeled rows |
| `reviewer_id` | Generate hash from available fields |
| `product_id` | Generate hash from available fields |

---

## 6. Sample Dataset

**Location:** `data/sample/sample_reviews.csv`  
**Purpose:** Local development and testing when real dataset is not available  
**Status:** ⚠️ **SYNTHETIC DATA — NOT FROM ANY REAL DATASET**

All rows have `is_synthetic=True`. The sample must never be mixed with real dataset evaluation.

### Sample Structure

- 100 rows: 70 genuine (label=0), 30 suspicious (label=1)
- 10 products: `P001`–`P010`
- 30 reviewers: `REV001`–`REV030`
- Suspicious reviewers (`REV026`–`REV030`): multiple reviews on same day, repetitive text, extreme ratings

### Why Synthetic Patterns Exist

The suspicious reviews in the sample have exaggerated patterns (ALL CAPS, repetitive phrases, extreme ratings) to ensure the ML classifier has clearly distinguishable training signal on the small sample. Real suspicious reviews are far more subtle.

---

## 7. Optional: Amazon Reviews Dataset

**Source:** https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html  
**Purpose:** Supplementary product metadata and unlabeled reviews for semantic retrieval exploration

### ⚠️ Restrictions

- Unlabeled Amazon reviews must **not** be used as ground truth for fake-review classification
- Use only for expanding the knowledge graph with product metadata and reviewer connectivity
- Check dataset-specific license before use

---

## 8. Processed Dataset Data Dictionary

After running `preprocess_data.py`, the output CSV has these columns:

| Column | Type | Source | Description |
|---|---|---|---|
| `review_id` | str | raw/generated | Stable unique identifier |
| `reviewer_id` | str | raw/generated | Stable reviewer identifier |
| `product_id` | str | raw/generated | Stable product identifier |
| `review_text` | str/NaN | raw | Original review text |
| `rating` | float/NaN | raw | Star rating 1–5 |
| `timestamp` | str/NaN | raw | Review date |
| `label` | float/NaN | raw+normalized | 0=genuine, 1=suspicious, NaN=unlabeled |
| `is_synthetic` | bool | preprocessing | True only for synthetic sample rows |
| `text_missing` | bool | preprocessing | True when review_text is NaN |
| `word_count` | int | preprocessing | Number of words in review |
| `char_count` | int | preprocessing | Number of characters |
| `avg_word_length` | float | preprocessing | Average word length |
| `reviewer_review_count` | int | preprocessing | Total reviews by this reviewer |
| `reviewer_distinct_products` | int | preprocessing | Distinct products reviewed |
| `product_avg_rating` | float | preprocessing | Mean rating across product reviews |
| `product_review_count` | int | preprocessing | Total reviews for this product |
| `predicted_class` | str/NaN | model inference | Set after model prediction |
| `model_score` | float/NaN | model inference | Model output score (NOT fraud probability) |

---

## 9. Provenance and License

| Dataset | Source | License | Notes |
|---|---|---|---|
| fake-reviews-data | https://github.com/bretthollenbeck/fake-reviews-data | See repo | Verify before use |
| Sample CSV | Synthetically generated | MIT (this project) | Clearly marked synthetic |
| Amazon Reviews | McAuley Lab, UCSD | Varies by subset | Check per-dataset license |

**All dataset processing preserves original identifiers for traceability.** Generated identifiers are hashes of available fields and are deterministic.

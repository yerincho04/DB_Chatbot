# Brand Resolver Spec (V1)

## Goal
Remove hardcoded alias mapping and resolve user brand mentions to canonical `brand_id`/`brand_name` from data.

## Scope
- Input: free-text brand mention from user query (Korean/English/mixed/typo)
- Output: deterministic resolver result used by tools before data lookup
- Data source: `brand_master` (normalized JSON from build pipeline)

## Resolver I/O Contract

### Input
- `query_text: str`
- `top_k: int = 3`

### Output
- `status: "resolved" | "ambiguous" | "not_found"`
- `query_text: str`
- `match:`
  - `brand_id: int`
  - `brand_name: str`
  - `confidence: float (0.0 ~ 1.0)`
- `candidates: list`
  - item:
    - `brand_id: int`
    - `brand_name: str`
    - `confidence: float`
    - `stage: str`
- `reason: str`

## Matching Pipeline (No Hardcoded Alias Dictionary)
1. `exact`:
- Case-insensitive exact match on `brand_name`.
- If single hit, return `resolved` with confidence `1.00`.

2. `normalized_exact`:
- Normalize both query and candidate by:
  - lowercase
  - remove spaces
  - keep alnum + Hangul only
- If single hit, return `resolved` with confidence `0.98`.

3. `fuzzy_rank`:
- Compute string similarity score against all candidate names using:
  - `difflib.SequenceMatcher` ratio (stdlib)
- Rank by score desc.
- Convert score to confidence using direct ratio.

4. `decision_gate`:
- If top1 confidence >= `0.90` and top1 - top2 >= `0.08`: `resolved`.
- If top1 confidence >= `0.75`: `ambiguous` with top `top_k` candidates.
- Else: `not_found`.

## Clarification Policy
- `ambiguous`:
  - Ask user to choose one of top candidates in Korean.
  - Example: "브랜드명이 모호합니다. 다음 중 어떤 브랜드인가요? 1) BBQ 2) BHC"
- `not_found`:
  - Say no match found and ask for rephrase.

## Integration Plan
1. Add resolver module in `data_access.py` (or `resolver.py`).
2. Replace `resolve_brand` hardcoded alias map path with resolver pipeline above.
3. Keep existing tool interfaces unchanged; tools call resolver internally.
4. If resolver status is `ambiguous/not_found`, raise localized `ValueError` with candidate hints.

## Acceptance Criteria
1. Korean phonetic mentions resolve without alias map where similarity is high.
2. New brand rows added to `brand_master` are automatically considered without code change.
3. No hardcoded brand alias dictionary remains in runtime resolution path.
4. Ambiguous mentions do not auto-guess; they return clarification.

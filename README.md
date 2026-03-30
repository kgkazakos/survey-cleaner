# survey-cleaner

Remove speeders, straight-liners, and gibberish open-ends from raw survey exports — before they corrupt your analysis.

```
$ survey-cleaner clean responses.csv --output cleaned.csv --report report.json

  Total: 312 responses · Flagged: 41 (13.1%) · Clean: 271
  Speeders: 18  Straight-liners: 9  Pattern-liners: 4  Gibberish/short: 12
```

## The problem

Raw survey exports are dirty. Qualtrics, SurveyMonkey, and Typeform all deliver the same thing: a mix of genuine responses and noise from respondents who clicked through in 18 seconds, rated every question 3, or typed "asdfghjkl" into the open-end field. Most teams catch some of this manually. Most don't catch all of it. `survey-cleaner` automates the triage.

## What it catches

**Speeders** — respondents who completed the survey faster than a configurable minimum time (default: 120 seconds). The severity score scales inversely with how far below the threshold they fell.

**Straight-liners** — respondents who gave the same rating across all Likert-scale items (zero variance). Likert columns are auto-detected from the data, or you can specify them explicitly.

**Pattern-liners** — respondents who followed a simple repeating sequence (e.g. 1,2,3,1,2,3 or 1,2,1,2) across rating items. Detected via period analysis up to configurable maximum period.

**Gibberish / too-short open-ends** — open-text responses scored on a composite of:
- Character-level Shannon entropy (low = repetitive)
- Consecutive-repeat character ratio
- Non-alphabetic character ratio
- Keyboard mash pattern detection (qwerty rows, common runs)
- Repeated single-word detection ("no no no no")
- Minimum character threshold

## Install

```bash
pip install survey-cleaner
```

With optional LLM enrichment for open-ends:

```bash
pip install "survey-cleaner[openai]"     # OpenAI backend
pip install "survey-cleaner[anthropic]"  # Anthropic backend
pip install "survey-cleaner[gemini]"     # Gemini backend
pip install "survey-cleaner[all]"        # all backends
```

## Usage

### CLI

```bash
# Clean a CSV and write results
survey-cleaner clean responses.csv --output cleaned.csv --report report.json

# Adjust thresholds
survey-cleaner clean responses.csv \
  --min-time 90 \
  --variance-threshold 0.1 \
  --gibberish-threshold 0.4

# Specify columns explicitly
survey-cleaner clean responses.csv \
  --open-end-cols "q5_text,q12_text,nps_reason" \
  --likert-cols "q1,q2,q3,q4,q5,q6,q7" \
  --id-col ResponseId

# Report only — no files written
survey-cleaner report responses.csv

# Dry run — shows flags without writing output
survey-cleaner clean responses.csv --dry-run

# Disable specific detectors
survey-cleaner clean responses.csv --no-speeders --no-straight-liners
```

### Python API

```python
import pandas as pd
from survey_cleaner import SurveyCleaner

df = pd.read_csv("responses.csv")

cleaner = SurveyCleaner(
    id_col="ResponseId",
    min_seconds=120,
    likert_cols=["q1", "q2", "q3", "q4", "q5"],
    open_end_cols=["open_feedback", "nps_reason"],
)

cleaned_df, result = cleaner.clean_df(df)

print(f"Flagged {result.flagged_count} / {result.total_responses} responses")
print(f"  Speeders: {result.speeder_count}")
print(f"  Straight-liners: {result.straight_liner_count}")
print(f"  Gibberish: {result.gibberish_count}")

# Access individual flags
for flag in result.flagged:
    print(flag.row_index, flag.respondent_id, flag.codes)

# Serialise the full report
import json
print(json.dumps(result.to_dict(), indent=2))
```

## Auto-detection

When column names are not specified, `survey-cleaner` auto-detects:

- **Timing column** — by matching common platform names: `duration_seconds`, `time_taken`, `Duration (in seconds)`, `elapsed_time`, etc.
- **Likert columns** — numeric columns with integer values that fit a known Likert range (1–5, 1–7, 0–4, etc.), with ≤ 7 unique values
- **Open-end columns** — string columns whose name matches common patterns (`comment`, `open`, `feedback`, `text`, `reason`, etc.), or high-uniqueness string columns with mean length > 15 characters

## Output

### JSON report structure

```json
{
  "total_responses": 312,
  "flagged_count": 41,
  "clean_count": 271,
  "flag_rate": 0.1314,
  "speeder_count": 18,
  "straight_liner_count": 9,
  "pattern_liner_count": 4,
  "gibberish_count": 12,
  "config": { ... },
  "flagged": [
    {
      "row_index": 7,
      "respondent_id": "R008",
      "reasons": [
        {
          "code": "speeder",
          "detail": "Completed in 34s (minimum: 120s)",
          "severity": 0.717
        }
      ],
      "max_severity": 0.717
    }
  ]
}
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Flag rate ≤ 20% |
| `1` | Flag rate > 20% — useful as a CI gate on data quality |

## CI / pipeline integration

```yaml
- name: Check survey quality
  run: survey-cleaner clean data/responses.csv --quiet
  # Exits 1 if >20% of responses are flagged
```

## Part of the CPR Ecosystem

`survey-cleaner` is Tool 4 in the [CPR Orbital](https://github.com/kgkazakos) series — 40 open-source tools for AI-native product research infrastructure, shipping every Friday from March to December 2026.

**Related tools in this series:**

| Tool | What it does |
|------|-------------|
| [`transcript-cleaner`](https://github.com/kgkazakos/transcript-cleaner) | Fix diarization errors and speaker gaps in raw transcripts |
| [`uxr-prompt-pack`](https://github.com/kgkazakos/uxr-prompt-pack) | Versioned LLM prompt library for UX research tasks |
| [`qual-sniffer`](https://github.com/kgkazakos/qual-sniffer) | Detect methodological weaknesses in discussion guides |
| `insight-differ` *(coming Apr 17)* | Compare two research reports and surface contradictions |

Part of the broader **Computational Product Research (CPR)** discipline. Core projects — CausalTrack, Synthetic User Council, Co-Reasoning Canvas — are developed separately and target CHI 2027/2028.

## License

MIT

"""Basic usage example for survey-cleaner."""

import pandas as pd
from survey_cleaner import SurveyCleaner

# Simulate a raw survey export
df = pd.DataFrame(
    {
        "ResponseId": ["R001", "R002", "R003", "R004", "R005", "R006"],
        "duration_seconds": [340, 25, 280, 310, 90, 420],
        "q1": [4, 3, 4, 5, 2, 3],
        "q2": [3, 3, 2, 5, 4, 2],
        "q3": [5, 3, 3, 5, 1, 4],
        "q4": [2, 3, 5, 5, 3, 1],
        "q5": [4, 3, 1, 5, 5, 5],
        "open_feedback": [
            "The onboarding flow is intuitive and the dashboard is well laid out.",
            "asdfghjkl qwerty lorem", 
            "Great product once you get past the setup.",
            "Really impressed with the team features.",
            "ok",
            "The Slack integration alone is worth the price.",
        ],
    }
)

print(f"Raw survey: {len(df)} responses\n")

cleaner = SurveyCleaner(
    id_col="ResponseId",
    min_seconds=120,
    likert_cols=["q1", "q2", "q3", "q4", "q5"],
    open_end_cols=["open_feedback"],
)

cleaned_df, result = cleaner.clean_df(df)

print(f"Total: {result.total_responses}")
print(f"Flagged: {result.flagged_count} ({result.flag_rate*100:.1f}%)")
print(f"  Speeders: {result.speeder_count}")
print(f"  Straight-liners: {result.straight_liner_count}")
print(f"  Gibberish/short open-ends: {result.gibberish_count}")
print(f"Clean: {result.clean_count}\n")

for fr in result.flagged:
    codes = [r.code.value for r in fr.reasons]
    print(f"  Row {fr.row_index} ({fr.respondent_id}): {', '.join(codes)}")
    for r in fr.reasons:
        print(f"    → {r.detail}")

print(f"\nCleaned DataFrame shape: {cleaned_df.shape}")

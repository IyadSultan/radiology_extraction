import pandas as pd
from src.extractor import process_batch

# Process multiple reports
reports_df = pd.read_csv("radiology_reports.csv")
results = await process_batch(reports_df, deps)

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
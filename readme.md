# Radiology Report Extraction Tool

A Python-based tool for extracting structured information from radiology reports, with a focus on oncology follow-up imaging. Features include RECIST criteria evaluation, findings classification, and standardized data extraction.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd radiology-extraction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start Example

```python
import asyncio
from radiology_extractor import process_radiology_report, ExtractionDependencies
from preprocessing import preprocess_report
from response_assessment import assess_response, RECISTEvaluator

# Example report
report_text = """
CT CHEST WITH CONTRAST

CLINICAL HISTORY: Lung cancer follow-up

COMPARISON: CT chest from 3 months ago

FINDINGS:
Right upper lobe mass measures 3.2 x 2.8 cm (previously 4.1 x 3.5 cm),
representing partial response to therapy.
"""

async def extract_single_report():
    # Initialize dependencies
    deps = ExtractionDependencies(
        modality_mapper=ModalityMapper(pd.read_csv('resources/modalities.csv')),
        location_mapper=LocationMapper(pd.read_csv('resources/ICDO3Topography.csv'))
    )
    
    # Process report
    result = await process_radiology_report(report_text, deps)
    print(result.dict())

# Run the example
asyncio.run(extract_single_report())
```

## Processing Multiple Files

```python
import pandas as pd
from pathlib import Path

async def process_directory():
    # Load reports from CSV
    reports_df = pd.read_csv('radiology_reports.csv')
    
    # Initialize dependencies
    deps = ExtractionDependencies(
        modality_mapper=ModalityMapper(pd.read_csv('resources/modalities.csv')),
        location_mapper=LocationMapper(pd.read_csv('resources/ICDO3Topography.csv'))
    )
    
    # Process in batches
    results = await process_batch(reports_df, deps, batch_size=200)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('extraction_results.csv', index=False)

# Run batch processing
if __name__ == "__main__":
    asyncio.run(process_directory())
```

## Using Response Assessment

```python
from response_assessment import RECISTEvaluator, TargetLesion, ResponseCategory

# Initialize evaluator
evaluator = RECISTEvaluator()

# Create target lesions
target_lesions = [
    TargetLesion(
        location="Right upper lobe",
        current_size=32.0,  # in mm
        prior_size=41.0     # in mm
    ),
    TargetLesion(
        location="Liver segment 7",
        current_size=15.0,
        prior_size=20.0
    )
]

# Evaluate response
response = evaluator.evaluate_target_response(target_lesions)
print(f"Response Category: {response.value}")  # e.g., "PR" for Partial Response
```

## Preprocessing Reports

```python
from preprocessing import preprocess_report

# Preprocess a report
preprocessed_data = preprocess_report(report_text)

# Access structured data
sections = preprocessed_data['sections']
measurements = preprocessed_data['measurements']
recist_data = preprocessed_data['recist_measurements']

# Print findings section
print(sections.get('findings', 'No findings section found'))
```

## Working with JSON Fields in Results

After extracting data to CSV, you can parse the JSON fields:

```python
import pandas as pd
import json

# Read results
results_df = pd.read_csv('extraction_results.csv')

# Parse JSON fields for analysis
def parse_measurements(row):
    if pd.notna(row['metastatic_lesions']):
        lesions = json.loads(row['metastatic_lesions'])
        return len(lesions)
    return 0

# Count metastatic lesions per report
results_df['lesion_count'] = results_df.apply(parse_measurements, axis=1)
```

## Resource Files

Place these files in the `resources` directory:

1. `modalities.csv`: Imaging modality reference data
```
Modality,Category,Subcategory,Description
CT,Cross-sectional,Standard,Computed Tomography
MRI,Cross-sectional,Standard,Magnetic Resonance Imaging
...
```

2. `ICDO3Topography.csv`: Anatomical site codes
```
ICDO3,term
C34.9,Lung
C71.9,Brain
...
```

## Batch Processing Options

The tool supports various batch processing options:

1. Process by date range:
```python
# Filter reports by date
date_mask = (reports_df['Study_Date'] >= start_date) & 
            (reports_df['Study_Date'] <= end_date)
filtered_df = reports_df[date_mask]
results = await process_batch(filtered_df, deps)
```

2. Process by modality:
```python
# Process only CT scans
ct_reports = reports_df[reports_df['Modality'].str.contains('CT', na=False)]
results = await process_batch(ct_reports, deps)
```

3. Resume interrupted processing:
```python
# Start from a specific index
results = await process_batch(reports_df, deps, start_idx=last_processed_idx)
```

## Error Handling

The tool includes comprehensive error handling:

```python
try:
    results = await process_batch(reports_df, deps)
except Exception as e:
    print(f"Error processing batch: {str(e)}")
    # Log error details
    with open('error_log.txt', 'a') as f:
        f.write(f"{datetime.now()}: {str(e)}\n")
```

## Output Format

The tool produces a CSV file with these columns:

- Basic fields (strings, numbers)
- JSON-encoded complex fields (lists, dictionaries)
- Error information if applicable

Example output row:
```python
{
    'MRN': '12345',
    'Document_Number': 'RAD001',
    'Study_Date': '2024-01-15',
    'modality': 'CT',
    'metastatic_lesions': '[{"location": "liver", "size": "15mm"}]',
    'classifications': '["Malignancy", "Metastasis"]'
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details
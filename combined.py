# Combined Python and HTML files
# Generated from directory: C:\Users\isultan\Documents\radiology_extraction
# Total files found: 15



# Contents from: .\templates\evaluate.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radiology Report Evaluation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .lesions-data {
            font-size: 0.9em;
            margin-top: 0.5rem;
        }
        .lesion-item {
            border-left: 2px solid #dee2e6;
            padding-left: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .lesion-item:last-child {
            margin-bottom: 0;
        }
        .report-text {
            white-space: pre-line;
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-size: 1em;
            line-height: 1.3;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }
        .evaluation-form {
            margin-top: 0.5rem;
        }
        .field-group {
            margin-bottom: 0.25rem;
            padding: 0.35rem;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            background-color: white;
            font-size: 0.9em;
        }
        .field-group:hover {
            background-color: #f8f9fa;
        }
        .field-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0;
        }
        .meta-info {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 0.5rem;
        }
        .keyboard-shortcut {
            font-size: 0.8em;
            color: #666;
            margin-left: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }}">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <div class="evaluator-info d-flex justify-content-between align-items-center bg-light p-2 mb-3">
            <div>
                <strong>Evaluator:</strong> {{ evaluator_name }}
            </div>
            <div>
                Report {{ current_index }} of {{ total_reports }} ({{ percentage }}% complete)
            </div>
        </div>

        <div class="row">
            <!-- Report Section -->
            <div class="col-6">
                <div class="card h-100">
                    <div class="card-header py-2">
                        <strong>Radiology Report</strong>
                        <div class="meta-info">
                            <small>MRN: {{ mrn }} | Exam Date: {{ exam_date }}</small>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="report-text" id="report-content">{{ report }}</div>
                    </div>
                </div>
            </div>

            <!-- Evaluation Form Section -->
            <div class="col-6">
                <form action="{{ url_for('submit_evaluation') }}" method="post" class="h-100">
                    <input type="hidden" name="mrn" value="{{ mrn }}">
                    <input type="hidden" name="exam_date" value="{{ exam_date }}">
                    <input type="hidden" name="current_index" value="{{ current_index }}">
                    
                    <div class="card h-100">
                        <div class="card-header py-2">
                            <strong>Extracted Fields - Please Verify</strong>
                        </div>
                        <div class="card-body">
                            {% for field_name, field_value in fields.items() %}
                                <div class="field-group">
                                    <div class="d-flex align-items-center">
                                        <input class="form-check-input me-2" type="checkbox" 
                                               name="field_{{ field_name }}" 
                                               id="field_{{ field_name }}"
                                               {% if field_name in field_results %}
                                                   {% if field_results[field_name] %}checked{% endif %}
                                               {% else %}
                                                   checked
                                               {% endif %}
                                               data-shortcut="{{ loop.index }}">
                                        <label class="form-check-label field-label mb-0" for="field_{{ field_name }}">
                                            <strong>{{ field_name|replace('_', ' ')|title }}</strong>
                                            <span class="text-muted">[{{ loop.index }}]</span>
                                        </label>
                                        <input type="hidden" name="value_{{ field_name }}" value="{{ field_value }}">
                                    </div>
                                    <div class="ms-4 mt-1">
                                        {% if field_name in ['target_lesions', 'non_target_lesions', 'new_lesions'] and field_value %}
                                            <div class="lesions-data">
                                                {% set lesions = field_value|parse_json %}
                                                {% if lesions %}
                                                    {% for lesion in lesions %}
                                                        <div class="lesion-item">
                                                            {% if lesion.location is defined %}
                                                                <strong>Location:</strong> {{ lesion.location }}<br>
                                                            {% endif %}
                                                            {% if lesion.current_value is defined %}
                                                                <strong>Size:</strong> {{ lesion.current_value }} {{ lesion.current_unit }}<br>
                                                            {% endif %}
                                                            {% if lesion.response_category is defined %}
                                                                <strong>Response:</strong> {{ lesion.response_category }}
                                                            {% endif %}
                                                        </div>
                                                    {% endfor %}
                                                {% else %}
                                                    <em>No lesions found</em>
                                                {% endif %}
                                            </div>
                                        {% else %}
                                            {{ field_value if field_value else '<em>No data</em>'|safe }}
                                        {% endif %}
                                    </div>
                                </div>
                            {% endfor %}

                            <div class="mt-3">
                                <label for="evaluator_comments" class="form-label">Comments:</label>
                                <textarea class="form-control form-control-sm" id="evaluator_comments" 
                                          name="evaluator_comments" rows="2"></textarea>
                            </div>
                            
                            <div class="mt-3 d-flex justify-content-between align-items-center">
                                <a href="{{ url_for('evaluate', index=prev_index) }}" 
                                   class="btn btn-outline-secondary btn-sm {% if not has_previous %}disabled{% endif %}">
                                    ← Previous
                                </a>
                                <div>
                                    <button type="submit" class="btn btn-primary btn-sm">
                                        Submit [Enter]
                                    </button>
                                    <a href="{{ url_for('skip_report') }}" class="btn btn-secondary btn-sm ms-2">
                                        Skip [Esc]
                                    </a>
                                </div>
                                <a href="{{ url_for('evaluate', index=next_index) }}" 
                                   class="btn btn-outline-secondary btn-sm {% if not has_next %}disabled{% endif %}">
                                    Next →
                                </a>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key >= '1' && e.key <= '9') {
                const checkbox = document.querySelector(`[data-shortcut="${e.key}"]`);
                if (checkbox) {
                    checkbox.checked = !checkbox.checked;
                }
            } else if (e.key === 'Enter' && !e.ctrlKey && !e.altKey) {
                document.querySelector('form').submit();
            } else if (e.key === 'Escape') {
                window.location.href = '{{ url_for("skip_report") }}';
            }
        });

        // Clean up the report text on load
        window.onload = function() {
            const reportElement = document.getElementById('report-content');
            let text = reportElement.innerText;
            
            // Remove multiple empty lines and clean up spacing
            text = text
                .replace(/\n\s*\n/g, '\n')  // Replace multiple empty lines with single line
                .replace(/^\s+|\s+$/g, '')   // Remove leading/trailing spaces
                .replace(/[\t ]+/g, ' ')     // Replace multiple spaces with single space
                .replace(/\n[\t ]+/g, '\n')  // Remove spaces at start of lines
                .replace(/[\t ]+\n/g, '\n'); // Remove spaces at end of lines
            
            reportElement.innerText = text;
        };
    </script>
</body>
</html>

# Contents from: .\templates\index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radiology Report Evaluation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .lesions-data {
            font-size: 0.9em;
            margin-top: 0.5rem;
        }
        .lesion-item {
            border-left: 2px solid #dee2e6;
            padding-left: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .lesion-item:last-child {
            margin-bottom: 0;
        }
        .report-text {
            white-space: pre-line;
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-size: 1em;
            line-height: 1.3;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }
        .evaluation-form {
            margin-top: 0.5rem;
        }
        .field-group {
            margin-bottom: 0.25rem;
            padding: 0.35rem;
            border: 1px solid #dee2e6;
            border-radius: 0.25rem;
            background-color: white;
            font-size: 0.9em;
        }
        .field-group:hover {
            background-color: #f8f9fa;
        }
        .field-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0;
        }
        .meta-info {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 0.5rem;
        }
        .keyboard-shortcut {
            font-size: 0.8em;
            color: #666;
            margin-left: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="alert alert-{{ category }}">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <div class="evaluator-info d-flex justify-content-between align-items-center bg-light p-2 mb-3">
            <div>
                <strong>Evaluator:</strong> {{ evaluator_name }}
            </div>
            <div>
                Report {{ current_index }} of {{ total_reports }} ({{ percentage }}% complete)
            </div>
        </div>

        <div class="row">
            <!-- Report Section -->
            <div class="col-6">
                <div class="card h-100">
                    <div class="card-header py-2">
                        <strong>Radiology Report</strong>
                        <div class="meta-info">
                            <small>MRN: {{ mrn }} | Exam Date: {{ exam_date }}</small>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="report-text" id="report-content">{{ report }}</div>
                    </div>
                </div>
            </div>

            <!-- Evaluation Form Section -->
            <div class="col-6">
                <form action="{{ url_for('submit_evaluation') }}" method="post" class="h-100">
                    <input type="hidden" name="mrn" value="{{ mrn }}">
                    <input type="hidden" name="exam_date" value="{{ exam_date }}">
                    <input type="hidden" name="current_index" value="{{ current_index }}">
                    
                    <div class="card h-100">
                        <div class="card-header py-2">
                            <strong>Extracted Fields - Please Verify</strong>
                        </div>
                        <div class="card-body">
                            {% for field_name, field_value in fields.items() %}
                                <div class="field-group">
                                    <div class="d-flex align-items-center">
                                        <input class="form-check-input me-2" type="checkbox" 
                                               name="field_{{ field_name }}" 
                                               id="field_{{ field_name }}"
                                               {% if field_name in field_results %}
                                                   {% if field_results[field_name] %}checked{% endif %}
                                               {% else %}
                                                   checked
                                               {% endif %}
                                               data-shortcut="{{ loop.index }}">
                                        <label class="form-check-label field-label mb-0" for="field_{{ field_name }}">
                                            <strong>{{ field_name|replace('_', ' ')|title }}</strong>
                                            <span class="text-muted">[{{ loop.index }}]</span>
                                        </label>
                                        <input type="hidden" name="value_{{ field_name }}" value="{{ field_value }}">
                                    </div>
                                    <div class="ms-4 mt-1">
                                        {% if field_name in ['target_lesions', 'non_target_lesions', 'new_lesions'] and field_value %}
                                            <div class="lesions-data">
                                                {% set lesions = field_value|parse_json %}
                                                {% for lesion in lesions %}
                                                    <div class="lesion-item">
                                                        <strong>Location:</strong> {{ lesion.location }}<br>
                                                        {% if lesion.current_value is defined %}
                                                            <strong>Size:</strong> {{ lesion.current_value }} {{ lesion.current_unit }}<br>
                                                        {% endif %}
                                                        {% if lesion.response_category is defined %}
                                                            <strong>Response:</strong> {{ lesion.response_category }}
                                                        {% endif %}
                                                    </div>
                                                {% endfor %}
                                            </div>
                                        {% else %}
                                            {{ field_value }}
                                        {% endif %}
                                    </div>
                                </div>
                            {% endfor %}

                            <div class="mt-3">
                                <label for="evaluator_comments" class="form-label">Comments:</label>
                                <textarea class="form-control form-control-sm" id="evaluator_comments" 
                                          name="evaluator_comments" rows="2"></textarea>
                            </div>
                            
                            <div class="mt-3 d-flex justify-content-between align-items-center">
                                <a href="{{ url_for('evaluate', index=prev_index) }}" 
                                   class="btn btn-outline-secondary btn-sm {% if not has_previous %}disabled{% endif %}">
                                    ← Previous
                                </a>
                                <div>
                                    <button type="submit" class="btn btn-primary btn-sm">
                                        Submit [Enter]
                                    </button>
                                    <a href="{{ url_for('skip_report') }}" class="btn btn-secondary btn-sm ms-2">
                                        Skip [Esc]
                                    </a>
                                </div>
                                <a href="{{ url_for('evaluate', index=next_index) }}" 
                                   class="btn btn-outline-secondary btn-sm {% if not has_next %}disabled{% endif %}">
                                    Next →
                                </a>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key >= '1' && e.key <= '9') {
                const checkbox = document.querySelector(`[data-shortcut="${e.key}"]`);
                if (checkbox) {
                    checkbox.checked = !checkbox.checked;
                }
            } else if (e.key === 'Enter' && !e.ctrlKey && !e.altKey) {
                document.querySelector('form').submit();
            } else if (e.key === 'Escape') {
                window.location.href = '{{ url_for("skip_report") }}';
            }
        });

        // Clean up the report text on load
        window.onload = function() {
            const reportElement = document.getElementById('report-content');
            let text = reportElement.innerText;
            
            // Remove multiple empty lines and clean up spacing
            text = text
                .replace(/\n\s*\n/g, '\n')  // Replace multiple empty lines with single line
                .replace(/^\s+|\s+$/g, '')   // Remove leading/trailing spaces
                .replace(/[\t ]+/g, ' ')     // Replace multiple spaces with single space
                .replace(/\n[\t ]+/g, '\n')  // Remove spaces at start of lines
                .replace(/[\t ]+\n/g, '\n'); // Remove spaces at end of lines
            
            reportElement.innerText = text;
        };
    </script>
</body>
</html>

# Contents from: .\templates\no_report.html


# Contents from: .\templates\no_reports.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>No Reports Left</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="card">
            <div class="card-body text-center">
                <h2 class="mb-4">No Reports Left to Evaluate</h2>
                <p>All available reports have been evaluated or no unevaluated reports were found.</p>
                <p>Check the following:</p>
                <ul class="list-unstyled">
                    <li>Verify that path_extraction_results_200.csv exists and contains data</li>
                    <li>Confirm that pathology_1000_reports.csv contains the corresponding Notes</li>
                    <li>Check human_evaluation_results.csv to see completed evaluations</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>

# Contents from: .\templates\set_evaluator.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Set Evaluator Name</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3 class="mb-0">Enter Your Name</h3>
                    </div>
                    <div class="card-body">
                        <form method="POST">
                            <div class="mb-3">
                                <label for="evaluator_name" class="form-label">Evaluator Name:</label>
                                <input type="text" class="form-control" id="evaluator_name" 
                                       name="evaluator_name" required autofocus>
                            </div>
                            <button type="submit" class="btn btn-primary">Start Evaluation</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>

# Contents from: .\combine.py
import os

def get_files_recursively(directory, extensions):
    """
    Recursively get all files with specified extensions from directory and subdirectories
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        # Skip venv directory
        if 'venv' in dirs:
            dirs.remove('venv')
        for file in files:
            # Skip combine.py and combined.py
            if file in ['combined.py']:
                continue
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))
    return file_list

def combine_files(output_file, file_list):
    """
    Combine contents of all files in file_list into output_file
    """
    with open(output_file, 'a', encoding='utf-8') as outfile:
        for fname in file_list:
            # Add a header comment to show which file's contents follow
            outfile.write(f"\n\n# Contents from: {fname}\n")
            try:
                with open(fname, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
            except Exception as e:
                outfile.write(f"# Error reading file {fname}: {str(e)}\n")

def main():
    # Define the base directory (current directory in this case)
    base_directory = "."
    output_file = 'combined.py'
    extensions = ('.py', '.html')

    # Remove output file if it exists
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except Exception as e:
            print(f"Error removing existing {output_file}: {str(e)}")
            return

    # Get all files recursively
    all_files = get_files_recursively(base_directory, extensions)
    
    # Sort files by extension and then by name
    all_files.sort(key=lambda x: (os.path.splitext(x)[1], x))

    # Add a header to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("# Combined Python and HTML files\n")
        outfile.write(f"# Generated from directory: {os.path.abspath(base_directory)}\n")
        outfile.write(f"# Total files found: {len(all_files)}\n\n")

    # Combine all files
    combine_files(output_file, all_files)
    
    print(f"Successfully combined {len(all_files)} files into {output_file}")
    print("Files processed:")
    for file in all_files:
        print(f"  - {file}")

if __name__ == "__main__":
    main()

# Contents from: .\examples\__init__.py


# Contents from: .\examples\batch_processing.py
import pandas as pd
from src.extractor import process_batch

# Process multiple reports
reports_df = pd.read_csv("radiology_reports.csv")
results = await process_batch(reports_df, deps)

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

# Contents from: .\examples\example.py
# examples/example.py

import asyncio
import pandas as pd
import sys
import os
from pathlib import Path
import json
import yaml
import logging
from typing import Optional

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from extractor import RadiologyExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example radiology report
example_report = """
CT CHEST WITH CONTRAST

CLINICAL HISTORY: Lung cancer follow-up.
Study date: 2024-01-15

COMPARISON: CT chest from 2023-10-15

FINDINGS:
Right upper lobe mass measures 3.2 x 2.8 cm (previously 4.1 x 3.5 cm),
representing partial response to therapy.

Multiple bilateral pulmonary nodules, largest measuring 8mm in right
lower lobe, decreased in size from prior study (was 12mm).

No new pulmonary nodules.

Mediastinal and hilar lymph nodes are stable.

Small right pleural effusion, unchanged.

IMPRESSION:
1. Decrease in size of right upper lobe mass and pulmonary nodules,
consistent with partial response to therapy.
2. Stable small right pleural effusion.
"""

def print_separator(text: str = "", char: str = "-", length: int = 50):
    """Print a separator line with optional text"""
    if text:
        print(f"\n{text}")
    print(char * length)

def print_measurement(location: str, current_mm: float, prior_mm: Optional[float] = None, 
                     percent_change: Optional[float] = None, response: Optional[str] = None):
    """Print measurement details in a consistent format"""
    print(f"\nLocation: {location}")
    print(f"Current: {current_mm:.1f}mm")
    if prior_mm is not None:
        print(f"Prior: {prior_mm:.1f}mm")
    if percent_change is not None:
        print(f"Change: {percent_change:.1f}%")
    if response:
        print(f"Response: {response}")

async def run_example():
    """Run the example extraction"""
    logger.info("Starting radiology report extraction example")
    
    # Initialize reference data
    resources_path = Path(__file__).parent.parent / 'resources'
    try:
        # Load reference data with proper typing
        modalities_df = pd.DataFrame({
            'Modality': ['CT', 'MRI', 'PET/CT', 'X-ray', 'Ultrasound'],
            'Category': ['Cross-sectional', 'Cross-sectional', 'Nuclear', 'Radiograph', 'Ultrasound']
        })
        
        topography_df = pd.DataFrame({
            'ICDO3': ['C34.1', 'C34.2', 'C34.3', 'C34.9'],
            'term': ['Upper lobe of lung', 'Middle lobe of lung', 'Lower lobe of lung', 'Lung']
        })
        
        # Initialize extractor
        extractor = RadiologyExtractor(modalities_df, topography_df)
        
        # Process report
        logger.info("Processing example report")
        result = await extractor.process_report(example_report)
        
        # Print results
        print_separator("Extracted Information", "=")
        
        # Basic Information
        print_separator("Study Information")
        print(f"Modality: {result.modality}")
        print(f"Body Region: {result.body_region}")
        if result.ICDO3_site:
            print(f"ICD-O Site: {result.ICDO3_site} ({result.ICDO3_site_term})")
        print(f"Study Date: {result.study_date}")
        print(f"Comparison Date: {result.comparison_date}")
        
        # Clinical Information
        if result.clinical_history:
            print_separator("Clinical Information")
            print(f"History: {result.clinical_history}")
        
        # Target Lesions
        if result.target_lesions:
            print_separator("Target Lesions")
            for lesion in result.target_lesions:
                print_measurement(
                    location=lesion.location,
                    current_mm=lesion.standardized_value_mm,
                    prior_mm=lesion.prior_value * 10 if lesion.prior_value and lesion.prior_unit == 'cm' 
                            else lesion.prior_value,
                    percent_change=lesion.percent_change,
                    response=lesion.response_category
                )
        
        # Non-target Lesions
        if result.non_target_lesions:
            print_separator("Non-target Lesions")
            for lesion in result.non_target_lesions:
                print_measurement(
                    location=lesion.location,
                    current_mm=lesion.standardized_value_mm
                )
        
        # Response Assessment
        if result.tumor_response or result.overall_assessment:
            print_separator("Response Assessment")
            if result.tumor_response:
                print(f"Tumor Response: {result.tumor_response}")
            if result.overall_assessment:
                print(f"Overall Assessment: {result.overall_assessment}")
        
        # Save complete output
        output_file = Path('example_output.json')
        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
        
    except FileNotFoundError as e:
        logger.error(f"Required resource files not found: {e}")
    except Exception as e:
        logger.error(f"Error in extraction process: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_example())

# Contents from: .\examples\example_reports.py
# example_reports.py

MAMMOGRAPHY_EXAMPLE = """
BILATERAL DIAGNOSTIC MAMMOGRAM

CLINICAL HISTORY: 55-year-old female with right breast mass on self-exam.

TECHNIQUE: Standard CC and MLO views of both breasts with spot compression views of the right breast mass.

BREAST COMPOSITION: Heterogeneously dense breast tissue (category C).

FINDINGS:
Right Breast:
- 2.3 cm spiculated mass at 10 o'clock, 8 cm from the nipple
- Associated pleomorphic microcalcifications
- No skin thickening or nipple retraction

Left Breast:
- No suspicious masses, calcifications, or architectural distortion
- Stable scattered fibroglandular densities

IMPRESSION:
Right breast mass with suspicious morphology and associated calcifications.
BIRADS Category: 4C - High suspicion for malignancy
"""

CHEST_CT_FUNGAL_EXAMPLE = """
CT CHEST WITH CONTRAST

CLINICAL HISTORY: Neutropenic fever in patient with AML, concern for fungal infection.

COMPARISON: CT chest from 2 weeks ago.

FINDINGS:
1. Multiple bilateral pulmonary nodules with surrounding ground-glass halos:
   - Right upper lobe: 2.1 cm nodule (previously 1.5 cm)
   - Left lower lobe: 1.8 cm nodule (previously 1.2 cm)
   Both demonstrating characteristic "halo sign"

2. New cavitary lesion in right lower lobe measuring 2.5 cm with air-crescent sign.

3. Scattered ground-glass opacities throughout both lung fields.

4. No pleural effusion.

IMPRESSION:
1. Progressive pulmonary findings highly suspicious for invasive fungal infection,
   demonstrating characteristic halo signs and air-crescent sign.
2. Recommend correlation with galactomannan and beta-D-glucan testing.
"""

BRAIN_TUMOR_EXAMPLE = """
BRAIN MRI WITH AND WITHOUT CONTRAST

CLINICAL HISTORY: Follow-up of known left temporal glioblastoma.

COMPARISON: MRI from 6 weeks ago.

TECHNIQUE: Multiplanar multisequence MRI with and without gadolinium.

FINDINGS:
Left temporal lobe mass:
- Measures 4.2 x 3.8 x 3.5 cm (previously 3.8 x 3.2 x 3.0 cm)
- Heterogeneous enhancement
- Increased surrounding FLAIR signal consistent with vasogenic edema
- Mass effect with 6mm rightward midline shift
- New areas of restricted diffusion along medial margin

Adjacent structures:
- Partial effacement of left temporal horn
- Uncal herniation measuring 3mm
- No hydrocephalus

IMPRESSION:
1. Progressive left temporal glioblastoma with:
   - Interval size increase
   - Increased mass effect and midline shift
   - New areas of restricted diffusion suggesting hypercellular tumor
2. Increased vasogenic edema with early uncal herniation
"""

# Contents from: .\flask_human_evaluation.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from datetime import datetime
import json
import re
import ast
from pathlib import Path

app = Flask(__name__)
app.secret_key = '1234567890'

def safe_json_loads(x):
    if not x or not isinstance(x, str):
        return {}
    
    try:
        # Handle the specific case of ModalityType enum
        if 'ModalityType' in x:
            # Extract just the string value
            match = re.search(r"'(.*?)'", x)
            if match:
                return {'modality_type': match.group(1)}
            return {'modality_type': 'Unknown'}
            
        # Try standard JSON parsing
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            # Try evaluating as Python literal
            return ast.literal_eval(x)
    except (json.JSONDecodeError, TypeError, SyntaxError, ValueError) as e:
        print(f"Warning: Could not parse value: {x}. Error: {str(e)}")
        return {}

# Update the template filter
@app.template_filter('parse_json')
def parse_json_filter(value):
    if not value:
        return []
    if isinstance(value, (list, dict)):
        return value
    return safe_json_loads(value)

def load_data(current_index=0):
    try:
        # Check if file exists
        if not Path('radiology_results_10.csv').exists():
            print("Error: radiology_results_10.csv not found")
            return pd.DataFrame(), {}, {}

        # Load radiology results
        print("Loading radiology_results_10.csv...")
        results_df = pd.read_csv('radiology_results_10.csv')
        print(f"Loaded {len(results_df)} rows from CSV")
        
        # Drop rows where Report is missing or just contains placeholder text
        results_df = results_df[
            results_df['REPORT'].notna() & 
            ~results_df['REPORT'].str.contains('Report will be available upon request', case=False, na=False)
        ]
        print(f"After filtering: {len(results_df)} rows")
        
        # Convert JSON string fields to objects
        json_fields = ['target_lesions', 'non_target_lesions', 'new_lesions', 
                      'classifications', 'other_findings']
        for field in json_fields:
            if field in results_df.columns:
                print(f"Processing {field} field...")
                results_df[field] = results_df[field].apply(lambda x: 
                    safe_json_loads(x) if isinstance(x, str) else [])
        
        # Handle modality_specific separately
        if 'modality_specific' in results_df.columns:
            print("Processing modality_specific field...")
            results_df['modality_specific'] = results_df['modality_specific'].apply(safe_json_loads)
        
        # Load previous evaluations if they exist
        submitted_status = {}
        previous_evaluations = {}
        if Path('human_evaluation_results.csv').exists():
            print("Loading previous evaluations...")
            evaluated_df = pd.read_csv('human_evaluation_results.csv')
            for _, row in evaluated_df.iterrows():
                mrn = str(row['MRN'])
                submitted_status[mrn] = True
                previous_evaluations[mrn] = {
                    col: row[col] 
                    for col in row.index 
                    if col not in ['MRN', 'EXAM_DATE', 'evaluator_name', 'evaluation_timestamp']
                }
            print(f"Loaded {len(previous_evaluations)} previous evaluations")
        
        if results_df.empty:
            print("Warning: No reports found after processing")
            return pd.DataFrame(), submitted_status, previous_evaluations
            
        print(f"Successfully processed all data. Returning {len(results_df)} reports")
        return results_df, submitted_status, previous_evaluations
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(), {}, {}

def save_evaluation(evaluation_data):
    try:
        # Convert evaluation data to DataFrame
        eval_df = pd.DataFrame([evaluation_data])
        
        # Add timestamp
        eval_df['evaluation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle JSON fields
        json_fields = ['target_lesions', 'non_target_lesions', 'new_lesions', 
                      'classifications', 'other_findings', 'modality_specific']
        for field in json_fields:
            if field in eval_df.columns:
                eval_df[field] = eval_df[field].apply(lambda x: 
                    json.dumps(x) if isinstance(x, (dict, list)) else x)
        
        # Append to existing file or create new one
        if Path('human_evaluation_results.csv').exists():
            existing_df = pd.read_csv('human_evaluation_results.csv')
            # Update or append based on MRN
            mrn = evaluation_data['MRN']
            if mrn in existing_df['MRN'].values:
                existing_df.loc[existing_df['MRN'] == mrn] = eval_df.iloc[0]
                existing_df.to_csv('human_evaluation_results.csv', index=False)
            else:
                eval_df.to_csv('human_evaluation_results.csv', mode='a', header=False, index=False)
        else:
            eval_df.to_csv('human_evaluation_results.csv', index=False)
            
        return True
    except Exception as e:
        print(f"Error saving evaluation: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('evaluate', index=0))

@app.route('/evaluate/<int:index>', methods=['GET'])
def evaluate(index):
    if 'evaluator_name' not in session:
        return redirect(url_for('set_evaluator'))
    
    # Load the data
    df, submitted_status, previous_evaluations = load_data()
    
    if df.empty:
        return render_template('no_reports.html')
    
    total_reports = len(df)
    index = max(0, min(index, total_reports - 1))
    
    current_report = df.iloc[index]
    mrn = str(current_report['MRN'])
    
    # Extract fields to evaluate - now including all fields from the JSON
    fields_to_evaluate = {
        'procedure': current_report.get('PROCEDURE', ''),
        'modality': current_report.get('modality', ''),
        'primary_location': current_report.get('primary_location', ''),
        'clinical_history': current_report.get('clinical_history', ''),
        'body_region': current_report.get('body_region', ''),
        'comparison_date': current_report.get('comparison_date', ''),
        'target_lesions': current_report.get('target_lesions', ''),
        'non_target_lesions': current_report.get('non_target_lesions', ''),
        'new_lesions': current_report.get('new_lesions', ''),
        'reported_response': current_report.get('reported_response', ''),
        'recist_calculated_response': current_report.get('recist_calculated_response', ''),
        'classifications': current_report.get('classifications', ''),
        'other_findings': current_report.get('other_findings', ''),
        'overall_assessment': current_report.get('overall_assessment', ''),
        'ICDO3_site': current_report.get('ICDO3_site', ''),
        'ICDO3_site_term': current_report.get('ICDO3_site_term', ''),
        'ICDO3_site_similarity': current_report.get('ICDO3_site_similarity', ''),
        'tumor_response': current_report.get('tumor_response', ''),
        'modality_specific': current_report.get('modality_specific', ''),
        'error': current_report.get('error', '')
    }
    
    # Get previous evaluation results if they exist
    field_results = {}
    if mrn in previous_evaluations:
        prev_eval = previous_evaluations[mrn]
        field_results = {field: prev_eval.get(field, True) for field in fields_to_evaluate.keys()}
    
    exam_date = str(current_report['EXAM_DATE']) if 'EXAM_DATE' in current_report else 'N/A'
    report = str(current_report['REPORT']) if 'REPORT' in current_report else 'Report not found'
    
    return render_template('evaluate.html',
                          mrn=mrn,
                          exam_date=exam_date,
                          report=report,
                          fields=fields_to_evaluate,
                          field_results=field_results,
                          current_index=index + 1,
                          total_reports=total_reports,
                          percentage=round((index + 1) / total_reports * 100),
                          evaluator_name=session['evaluator_name'],
                          is_submitted=mrn in submitted_status,
                          has_previous=index > 0,
                          has_next=index < total_reports - 1,
                          prev_index=index - 1,
                          next_index=index + 1)
    


@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    try:
        form_data = request.form
        
        # Create evaluation record with metadata
        evaluation = {
            'MRN': form_data['mrn'],
            'EXAM_DATE': form_data['exam_date'],
            'evaluator_name': session['evaluator_name']
        }
        
        # Get all field names from hidden values and add True/False results
        field_names = [key.replace('value_', '') for key in form_data.keys() 
                      if key.startswith('value_')]
        
        for field_name in field_names:
            checkbox_name = f'field_{field_name}'
            evaluation[field_name] = checkbox_name in form_data
        
        if save_evaluation(evaluation):
            flash('Evaluation saved successfully!', 'success')
        else:
            flash('Error saving evaluation!', 'error')
        
        # Move to next report after submission
        current_index = int(request.form.get('current_index', 0))
        return redirect(url_for('evaluate', index=current_index))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/set_evaluator', methods=['GET', 'POST'])
def set_evaluator():
    if request.method == 'POST':
        evaluator_name = request.form.get('evaluator_name')
        if evaluator_name:
            session['evaluator_name'] = evaluator_name
            return redirect(url_for('index'))
    return render_template('set_evaluator.html')

@app.route('/skip_report', methods=['GET'])
def skip_report():
    flash('Report skipped!', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# Contents from: .\process_reports.py
# process_reports.py

import asyncio
import pandas as pd
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm

# Add the src directory to Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from extractor import RadiologyExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_reports(input_file: str, output_file: str, num_reports: int = 10):
    """
    Process reports from input CSV file and save results
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save results
        num_reports: Number of reports to process
    """
    try:
        # Read input CSV
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Select first n reports
        df = df.head(num_reports)
        
        # Load reference data from resources
        resources_path = Path(__file__).parent / 'resources'
        try:
            logger.info("Loading reference data...")
            modalities_df = pd.read_csv(resources_path / 'modalities.csv')
            topography_df = pd.read_csv(resources_path / 'ICDO3Topography.csv')
        except FileNotFoundError as e:
            logger.error(f"Required resource files not found in {resources_path}")
            logger.error("Please ensure modalities.csv and ICDO3Topography.csv exist in the resources directory")
            raise
        
        # Initialize extractor
        extractor = RadiologyExtractor(modalities_df, topography_df)
        
        # Process each report
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reports"):
            try:
                report_info = {
                    'MRN': row['MRN'],
                    'EXAM_DATE': row['EXAM DATE/TIME'],
                    'PROCEDURE': row['PROCEDURE'],
                    'REPORT': row['REPORT']
                }
                
                # Process report text
                result = await extractor.process_report(row['REPORT'])
                
                # Combine base info with extraction results
                combined_result = {
                    **report_info,
                    **result.dict(exclude_none=True)  # Exclude None values for cleaner output
                }
                
                results.append(combined_result)
                
            except Exception as e:
                logger.error(f"Error processing report {idx + 1}: {str(e)}")
                error_result = {
                    **report_info,
                    'error': str(e)
                }
                results.append(error_result)
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        results_df = pd.DataFrame(results)
        
        # Reorder columns to put primary fields first
        columns = ['MRN', 'EXAM_DATE', 'PROCEDURE', 'REPORT']
        other_columns = [col for col in results_df.columns if col not in columns]
        results_df = results_df[columns + other_columns]
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        logger.info(f"Successfully processed {len(results_df)} reports")
        if 'error' in results_df.columns:
            error_count = results_df['error'].notna().sum()
            logger.info(f"Errors encountered: {error_count}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise

def check_resources():
    """Check if required resource files exist"""
    resources_path = Path(__file__).parent / 'resources'
    required_files = ['modalities.csv', 'ICDO3Topography.csv']
    
    missing_files = []
    for file in required_files:
        if not (resources_path / file).exists():
            missing_files.append(file)
    
    return missing_files

async def main():
    """Main function"""
    # Check for required files
    missing_files = check_resources()
    if missing_files:
        logger.error("Missing required resource files:")
        for file in missing_files:
            logger.error(f"- {file}")
        return
    
    input_file = "data/Results.csv"
    output_file = "radiology_results_10.csv"
    
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    await process_reports(input_file, output_file)

if __name__ == "__main__":
    asyncio.run(main())

# Contents from: .\src\__init__.py


# Contents from: .\src\extractor.py
# extractor.py

import warnings
import json
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class ResponseCategory(str, Enum):
    """RECIST response categories"""
    COMPLETE_RESPONSE = "Complete Response"
    PARTIAL_RESPONSE = "Partial Response"
    STABLE_DISEASE = "Stable Disease"
    PROGRESSIVE_DISEASE = "Progressive Disease"
    NOT_EVALUABLE = "Not Evaluable"

class ModalityType(str, Enum):
    """Supported modality types"""
    MAMMOGRAPHY = "Mammography"
    CHEST_CT = "Chest CT"
    BRAIN_IMAGING = "Brain Imaging"
    OTHER = "Other"

# Modality-specific models
class MammographyFindings(BaseModel):
    """Mammography-specific findings"""
    birads_category: str = Field(description="BI-RADS assessment category")
    breast_density: str = Field(description="Breast density category")
    masses: List[Dict[str, str]] = Field(default_factory=list)
    calcifications: List[Dict[str, str]] = Field(default_factory=list)
    architectural_distortion: Optional[bool] = None
    asymmetries: List[Dict[str, str]] = Field(default_factory=list)

class ChestCTFungalFindings(BaseModel):
    """Chest CT findings focused on fungal infections"""
    halo_sign: List[Dict[str, str]] = Field(default_factory=list)
    cavitations: List[Dict[str, str]] = Field(default_factory=list)
    fungal_nodules: List[Dict[str, str]] = Field(default_factory=list)
    ground_glass_opacities: List[Dict[str, str]] = Field(default_factory=list)
    air_crescent_signs: List[Dict[str, str]] = Field(default_factory=list)
    other_fungal_findings: List[Dict[str, str]] = Field(default_factory=list)

class BrainTumorFindings(BaseModel):
    """Brain imaging findings focused on tumors"""
    tumor_details: Dict[str, Any] = Field(...)
    edema: Optional[Dict[str, str]] = None
    mass_effect: Optional[Dict[str, str]] = None
    enhancement_pattern: Optional[str] = None
    brain_region: str = Field(description="Specific brain region affected")
    additional_features: List[Dict[str, str]] = Field(default_factory=list)

class ModalitySpecific(BaseModel):
    """Container for modality-specific findings"""
    modality_type: ModalityType
    mammography: Optional[MammographyFindings] = None
    chest_ct: Optional[ChestCTFungalFindings] = None
    brain_tumor: Optional[BrainTumorFindings] = None

class RECISTMeasurement(BaseModel):
    """Model for RECIST measurements"""
    location: str = Field(description="Anatomical location of measurement")
    current_value: float = Field(description="Current measurement value")
    current_unit: str = Field(description="Unit of measurement (mm/cm)")
    standardized_value_mm: Optional[float] = Field(default=None)
    prior_value: Optional[float] = Field(default=None)
    prior_unit: Optional[str] = Field(default=None)
    percent_change: Optional[float] = Field(default=None)
    response_category: Optional[str] = Field(default=None)
    is_target: bool = Field(default=False)

    def __init__(self, **data):
        super().__init__(**data)
        self.standardize_measurements()
    
    def standardize_measurements(self):
        """Convert measurements to mm and calculate response"""
        if self.current_value:
            if self.current_unit == 'cm':
                self.standardized_value_mm = self.current_value * 10
            elif self.current_unit == 'mm':
                self.standardized_value_mm = self.current_value
            
            if self.prior_value and self.prior_unit:
                prior_mm = self.prior_value * 10 if self.prior_unit == 'cm' else self.prior_value
                self.percent_change = ((self.standardized_value_mm - prior_mm) / prior_mm) * 100
                
                if self.is_target:
                    if self.standardized_value_mm == 0:
                        self.response_category = ResponseCategory.COMPLETE_RESPONSE.value
                    elif self.percent_change <= -30:
                        self.response_category = ResponseCategory.PARTIAL_RESPONSE.value
                    elif self.percent_change >= 20:
                        self.response_category = ResponseCategory.PROGRESSIVE_DISEASE.value
                    else:
                        self.response_category = ResponseCategory.STABLE_DISEASE.value

class ClassificationResult(BaseModel):
    """Model for classification results"""
    class_name: str = Field(description="Classification category")
    description: str = Field(description="Description of the finding")
    confidence: Optional[float] = Field(default=None)

class OtherFinding(BaseModel):
    """Model for other findings"""
    item: str = Field(description="Type of finding")
    description: str = Field(description="Description of the finding")

class RadiologyReport(BaseModel):
    """Model for structured radiology report data"""
    report: str = Field(description="Original report text")
    modality: str = Field(description="Imaging modality (CT/MRI/PET/etc)")
    primary_location: str = Field(description="Anatomical region of primary tumor using ICDO3 description (not code)")
    study_date: Optional[str] = Field(default=None)
    comparison_date: Optional[str] = Field(default=None)
    clinical_history: Optional[str] = Field(default=None)
    indication: Optional[str] = Field(default=None)
    
    target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    non_target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    new_lesions: List[Dict[str, Any]] = Field(default_factory=list)
    
    reported_response: Optional[str] = Field(default=None)
    recist_calculated_response: Optional[str] = Field(default=None)
    
    classifications: List[ClassificationResult] = Field(default_factory=list)
    other_findings: List[OtherFinding] = Field(default_factory=list)
    
    ICDO3_site: Optional[str] = Field(default=None)
    ICDO3_site_term: Optional[str] = Field(default=None)
    ICDO3_site_similarity: Optional[float] = Field(default=None)
    
    modality_specific: Optional[ModalitySpecific] = None

    def calculate_recist_response(self) -> str:
        """Calculate overall RECIST response"""
        if self.new_lesions:
            return ResponseCategory.PROGRESSIVE_DISEASE.value
            
        target_responses = [lesion.response_category for lesion in self.target_lesions 
                          if lesion.response_category]
                          
        if not target_responses:
            return ResponseCategory.NOT_EVALUABLE.value
            
        if all(r == ResponseCategory.COMPLETE_RESPONSE.value for r in target_responses):
            return ResponseCategory.COMPLETE_RESPONSE.value
        elif any(r == ResponseCategory.PROGRESSIVE_DISEASE.value for r in target_responses):
            return ResponseCategory.PROGRESSIVE_DISEASE.value
        elif any(r == ResponseCategory.PARTIAL_RESPONSE.value for r in target_responses):
            return ResponseCategory.PARTIAL_RESPONSE.value
        else:
            return ResponseCategory.STABLE_DISEASE.value

class ModalityMapper:
    """Maps imaging modalities to standardized terminology"""
    def __init__(self, modalities_df: pd.DataFrame):
        self.modalities_df = modalities_df
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.vectors = self.vectorizer.fit_transform(modalities_df['Modality'].fillna(''))

    def find_closest_modality(self, text: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            query = self.vectorizer.transform([text])
            sims = cosine_similarity(query, self.vectors).flatten()
            idx = sims.argmax()
            
            if sims[idx] < threshold:
                return None
                
            result = {
                'standard_name': self.modalities_df.iloc[idx]['Modality'],
                'category': self.modalities_df.iloc[idx]['Category'],
                'similarity': float(sims[idx])
            }
            
            # Determine modality type
            if 'mammogram' in text.lower() or 'mammography' in text.lower():
                result['modality_type'] = ModalityType.MAMMOGRAPHY
            elif 'chest' in text.lower() and 'ct' in text.lower():
                result['modality_type'] = ModalityType.CHEST_CT
            elif any(term in text.lower() for term in ['brain', 'head']) and \
                 any(term in text.lower() for term in ['ct', 'mri', 'magnetic']):
                result['modality_type'] = ModalityType.BRAIN_IMAGING
            else:
                result['modality_type'] = ModalityType.OTHER
            
            print(f"Modality matching result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error finding modality match: {str(e)}")
            return None

class LocationMapper:
    """Maps anatomical locations to standardized terminology"""
    def __init__(self, topography_df: pd.DataFrame):
        self.topography_df = topography_df
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.vectors = self.vectorizer.fit_transform(topography_df['term'].fillna(''))

    def find_closest_location(self, text: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            query = self.vectorizer.transform([text])
            sims = cosine_similarity(query, self.vectors).flatten()
            idx = sims.argmax()
            print(f"Debug - Location input: {text}")
            print(f"Debug - Best match: {self.topography_df.iloc[idx]['term']}")
            print(f"Debug - Similarity score: {sims[idx]}")
            
            if sims[idx] < threshold:
                return None
            
            return {
                'code': self.topography_df.iloc[idx]['ICDO3'],
                'term': self.topography_df.iloc[idx]['term'],
                'similarity': float(sims[idx])
            }
        except Exception as e:
            logger.error(f"Error finding location match: {str(e)}")
            return None

@dataclass
class ExtractionDependencies:
    """Dependencies for extraction process"""
    modality_mapper: Any
    location_mapper: Any

# Initialize standard extraction agent
standard_extraction_agent = Agent(
    "openai:gpt-4o",
    retries=3,
    deps_type=ExtractionDependencies,
    result_type=RadiologyReport,
    system_prompt="""
    You are an expert radiologist specializing in structured data extraction from radiology reports.
    
    Extract all information in a single pass and return a complete RadiologyReport object:
    
    1. Study Information:
       - modality (CT/MRI/PET/etc)
       - primary_location (anatomical region of primary tumor using ICDO3)
       - dates (study and comparison)
       - clinical_history
       - indication
       
    2. Measurements and RECIST:
       - For each lesion: current size, prior size if available
       - Mark target vs non-target lesions
       - Record new lesions if any
       
    3. Response Assessment:
       - reported_response: Extract the response assessment as stated
       - recist_calculated_response will be calculated automatically
       
    4. Classifications:
       Classify findings into these categories:
       - Normal: No significant abnormality
       - Infection: Any infectious process
       - Metastasis: Evidence of metastatic disease
       - Primary tumor: Primary tumor findings
       - Effusion: Pleural effusion, pericardial effusion, or ascites
       - Trauma: Evidence of injury
       - Hemorrhage: Any bleeding
       - Thrombosis: Blood clots or emboli
       - Others: Any other significant findings
       
    5. Structure output:
       classifications: [
           {"class_name": "<category>", "description": "<detailed finding>"}
       ]
       other_findings: [
           {"item": "<finding type>", "description": "<detailed description>"}
       ]
       
    Only include explicitly stated information.
    Provide detailed, specific descriptions for each finding.
    """
)

# Initialize modality-specific extraction agent
modality_specific_agent = Agent(
    "openai:gpt-4o",
    retries=3,
    deps_type=ExtractionDependencies,
    result_type=ModalitySpecific,
    system_prompt="""
    You are an expert radiologist focusing on modality-specific findings extraction.
    Based on the identified modality type, extract relevant specialized information:

    For Mammography:
    - BI-RADS category
    - Breast density
    - Details of masses and calcifications (location, type)
    - Architectural distortion
    - Asymmetries

    For Chest CT (Fungal Infections):
    - Halo sign presence and details
    - Cavitations
    - Nodules consistent with fungal disease
    - Ground-glass opacities
    - Air-crescent signs
    - Other fungal-specific findings

    For Brain Imaging (Tumors):
    - Tumor location (specific brain region)
    - Size and characteristics
    - Edema presence and extent
    - Mass effect
    - Enhancement pattern
    - Additional features

    Return findings in the appropriate structure based on modality type.
    Only include explicitly stated findings.
    Provide detailed descriptions for each finding.
    """
)

@standard_extraction_agent.tool
async def find_modality(ctx: RunContext[ExtractionDependencies], text: str):
    """Tool for matching imaging modalities"""
    result = ctx.deps.modality_mapper.find_closest_modality(text)
    print(f"Modality mapping result: {result}")
    return result

@standard_extraction_agent.tool
async def find_location(ctx: RunContext[ExtractionDependencies], text: str):
    """Tool for matching anatomical locations"""
    result = ctx.deps.location_mapper.find_closest_location(text)
    print(f"Location mapping result: {result}")
    return result

# Add the same tools to modality-specific agent
modality_specific_agent.tool(find_modality)
modality_specific_agent.tool(find_location)

class RadiologyExtractor:
    """Main class for extracting information from radiology reports"""
    
    def __init__(self, modalities_df: pd.DataFrame, topography_df: pd.DataFrame):
        self.modality_mapper = ModalityMapper(modalities_df)
        self.location_mapper = LocationMapper(topography_df)

    async def process_report(self, text: str) -> RadiologyReport:
        """Process a single radiology report using two-pass extraction"""
        try:
            logger.info("Starting standard extraction...")
            
            # Create dependencies
            deps = ExtractionDependencies(
                modality_mapper=self.modality_mapper,
                location_mapper=self.location_mapper
            )
            
            # First pass: Standard extraction
            result = await standard_extraction_agent.run(text, deps=deps)
            
            # Store original report text
            result.data.report = text
            
            # Match location code for primary location
            if result.data.primary_location:
                logger.info(f"Matching location for: {result.data.primary_location}")
                location_match = await find_location(
                    RunContext(deps=deps, retry=0, tool_name="find_location"),
                    result.data.primary_location
                )
                
                if location_match:
                    logger.info(f"Found location match: {location_match}")
                    result.data.ICDO3_site = location_match['code']
                    result.data.ICDO3_site_term = location_match['term']
                    result.data.ICDO3_site_similarity = location_match['similarity']
                else:
                    logger.warning("No location match found")
            
            # Calculate RECIST response
            result.data.recist_calculated_response = result.data.calculate_recist_response()
            
            # Second pass: Modality-specific extraction
            modality_result = await find_modality(
                RunContext(deps=deps, retry=0, tool_name="find_modality"),
                text[:200]  # Use first part of report for modality detection
            )
            
            if modality_result and 'modality_type' in modality_result:
                modality_type = modality_result['modality_type']
                logger.info(f"Detected modality type: {modality_type}")
                
                if modality_type != ModalityType.OTHER:
                    logger.info("Performing modality-specific extraction...")
                    try:
                        modality_specific_result = await modality_specific_agent.run(
                            text,
                            context={'modality_type': modality_type},
                            deps=deps
                        )
                        result.data.modality_specific = modality_specific_result.data
                    except Exception as e:
                        logger.error(f"Error in modality-specific extraction: {str(e)}")
            
            return result.data
            
        except Exception as e:
            logger.error("Error processing report", exc_info=True)
            raise

async def process_batch(reports_df: pd.DataFrame,
                       extractor: RadiologyExtractor,
                       batch_size: int = 10) -> pd.DataFrame:
    """Process a batch of reports"""
    results = []
    total = len(reports_df)
    
    logger.info(f"Processing {total} reports in batches of {batch_size}")
    
    for idx, row in enumerate(reports_df.iterrows(), 1):
        try:
            _, report_row = row
            logger.info(f"Processing report {idx}/{total}")
            
            result = await extractor.process_report(report_row['REPORT'])
            
            # Combine with metadata
            result_dict = {
                'MRN': report_row.get('MRN'),
                'EXAM_DATE': report_row.get('EXAM DATE/TIME'),
                'PROCEDURE': report_row.get('PROCEDURE'),
                **result.dict(exclude_none=True)
            }
            results.append(result_dict)
            
        except Exception as e:
            logger.error(f"Error processing report {idx}: {str(e)}")
            results.append({
                'MRN': report_row.get('MRN'),
                'EXAM_DATE': report_row.get('EXAM DATE/TIME'),
                'PROCEDURE': report_row.get('PROCEDURE'),
                'error': str(e)
            })
            
        if idx % batch_size == 0:
            logger.info(f"Completed {idx}/{total} reports")
    
    return pd.DataFrame(results)

# main
async def main():
    """Main function for demonstration"""
    try:
        # Update the resources path resolution
        resources_path = Path(__file__).parent.parent / 'resources'
        
        logger.info("Loading reference data...")
        try:
            modalities_df = pd.read_csv(resources_path / 'modalities.csv')
            topography_df = pd.read_csv(resources_path / 'ICDO3Topography.csv')
        except FileNotFoundError as e:
            logger.error(f"Error loading resource files: {e}")
            logger.error(f"Please ensure files exist in: {resources_path}")
            return
            
        # Initialize extractor
        extractor = RadiologyExtractor(modalities_df, topography_df)
        
        # Check if we're processing a batch
        input_file = Path('Results.csv')
        if input_file.exists():
            logger.info("Processing batch from Results.csv...")
            reports_df = pd.read_csv(input_file)
            results_df = await process_batch(reports_df, extractor)
            output_file = 'radiology_results.csv'
            results_df.to_csv(output_file, index=False)
            logger.info(f"Batch processing complete. Results saved to {output_file}")
            
            # Print summary
            total_reports = len(results_df)
            error_reports = results_df['error'].notna().sum() if 'error' in results_df.columns else 0
            success_reports = total_reports - error_reports
            
            print("\nProcessing Summary:")
            print(f"Total Reports: {total_reports}")
            print(f"Successfully Processed: {success_reports}")
            print(f"Errors: {error_reports}")
            
            return
        
        # Example report for testing
        example_report = """
        CT CHEST WITH CONTRAST
        
        CLINICAL HISTORY: Follow-up of lung cancer with fungal infection.
        
        FINDINGS:
        1. Right upper lobe mass measures 3.2 x 2.8 cm (previously 4.1 x 3.5 cm),
           representing partial response to therapy.
        2. Multiple bilateral pulmonary nodules with surrounding ground-glass halos,
           largest measuring 8mm in right lower lobe.
        3. Small cavitary lesion in left lower lobe with air-crescent sign.
        4. Small right pleural effusion.
        
        IMPRESSION:
        1. Partial response of primary lung tumor.
        2. Findings consistent with pulmonary fungal infection.
        3. Small right pleural effusion.
        """
        
        # Build example metadata for the report
        example_metadata = {
            'EXAM DATE/TIME': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'PROCEDURE': 'CT CHEST WITH CONTRAST'
        }
        
        # Process single report
        result = await extractor.process_report(example_report)
        
        # Print results
        print("\nExtracted Information:")
        print("=" * 50)
        
        # Basic Information
        print("\nStudy Information:")
        print(f"Modality: {result.modality}")
        print(f"Primary Location: {result.primary_location}")
        if result.ICDO3_site:
            print(f"ICD-O Site: {result.ICDO3_site} ({result.ICDO3_site_term})")
        
        # Response Assessment
        print("\nResponse Assessment:")
        print(f"Reported Response: {result.reported_response}")
        print(f"RECIST Calculated: {result.recist_calculated_response}")
        
        # Target Lesions
        if result.target_lesions:
            print("\nTarget Lesions:")
            for lesion in result.target_lesions:
                print(f"\n- Location: {lesion.location}")
                print(f"  Current: {lesion.current_value} {lesion.current_unit}")
                if lesion.prior_value:
                    print(f"  Prior: {lesion.prior_value} {lesion.prior_unit}")
                    print(f"  Change: {lesion.percent_change:.1f}%")
                print(f"  Response: {lesion.response_category}")
        
        # Non-target Lesions
        if result.non_target_lesions:
            print("\nNon-target Lesions:")
            for lesion in result.non_target_lesions:
                print(f"\n- Location: {lesion.location}")
                print(f"  Current: {lesion.current_value} {lesion.current_unit}")
        
        # Classifications
        if result.classifications:
            print("\nClassifications:")
            for classification in result.classifications:
                print(f"\n{classification.class_name}:")
                print(f"  {classification.description}")
        
        # Modality-specific findings
        if result.modality_specific:
            print("\nModality-Specific Findings:")
            print(json.dumps(
                result.modality_specific.dict(exclude_none=True),
                indent=2
            ))
        
        # Save complete output with metadata
        output_file = Path('example_output.json')
        output_data = {
            'EXAM_DATE': example_metadata['EXAM DATE/TIME'],
            'PROCEDURE': example_metadata['PROCEDURE'],
            **result.dict(exclude_none=True)
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nComplete results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# Contents from: .\src\response_assessment.py
# response_assessment.py
"""
Module for handling response assessment criteria including RECIST 1.1
"""

from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

class ResponseCategory(Enum):
    """Enumeration of possible response categories"""
    COMPLETE_RESPONSE = "CR"
    PARTIAL_RESPONSE = "PR"
    STABLE_DISEASE = "SD"
    PROGRESSIVE_DISEASE = "PD"
    NOT_EVALUABLE = "NE"

@dataclass
class TargetLesion:
    """Class representing a target lesion measurement"""
    location: str
    current_size: float  # in mm
    prior_size: Optional[float] = None  # in mm
    measurement_axis: str = "long"  # long, short, or both
    
    @property
    def percent_change(self) -> Optional[float]:
        """Calculate percent change from prior measurement"""
        if self.prior_size is not None and self.prior_size > 0:
            return ((self.current_size - self.prior_size) / self.prior_size) * 100
        return None

@dataclass
class NonTargetLesion:
    """Class representing a non-target lesion assessment"""
    location: str
    status: str  # Complete Response, Present, Absent, Unequivocal Progression

@dataclass
class NewLesion:
    """Class representing a new lesion"""
    location: str
    description: str
    size: Optional[float] = None  # in mm

class RECISTEvaluator:
    """Class for evaluating response according to RECIST 1.1 criteria"""
    
    def __init__(self):
        self.response_thresholds = {
            'PR': -30,  # 30% decrease
            'PD': 20    # 20% increase
        }
    
    def evaluate_target_response(self, 
                               target_lesions: List[TargetLesion]) -> ResponseCategory:
        """Evaluate response based on target lesions"""
        if not target_lesions:
            return ResponseCategory.NOT_EVALUABLE
            
        # Calculate sum of diameters
        current_sum = sum(lesion.current_size for lesion in target_lesions)
        
        # If all lesions have prior measurements
        if all(lesion.prior_size is not None for lesion in target_lesions):
            prior_sum = sum(lesion.prior_size for lesion in target_lesions if lesion.prior_size)
            
            # Calculate percent change
            if prior_sum > 0:
                percent_change = ((current_sum - prior_sum) / prior_sum) * 100
                
                # Determine response category
                if current_sum == 0:
                    return ResponseCategory.COMPLETE_RESPONSE
                elif percent_change <= self.response_thresholds['PR']:
                    return ResponseCategory.PARTIAL_RESPONSE
                elif percent_change >= self.response_thresholds['PD']:
                    return ResponseCategory.PROGRESSIVE_DISEASE
                else:
                    return ResponseCategory.STABLE_DISEASE
                    
        return ResponseCategory.NOT_EVALUABLE
        
    def evaluate_non_target_response(self,
                                   non_target_lesions: List[NonTargetLesion]) -> ResponseCategory:
        """Evaluate response based on non-target lesions"""
        if not non_target_lesions:
            return ResponseCategory.NOT_EVALUABLE
            
        # Check for unequivocal progression
        if any(lesion.status == "Unequivocal Progression" for lesion in non_target_lesions):
            return ResponseCategory.PROGRESSIVE_DISEASE
            
        # Check for complete response
        if all(lesion.status == "Complete Response" or lesion.status == "Absent" 
               for lesion in non_target_lesions):
            return ResponseCategory.COMPLETE_RESPONSE
            
        # If all lesions are present but stable
        if all(lesion.status == "Present" for lesion in non_target_lesions):
            return ResponseCategory.NON_CR_NON_PD
            
        return ResponseCategory.NOT_EVALUABLE
        
    def evaluate_overall_response(self,
                                target_lesions: List[TargetLesion],
                                non_target_lesions: List[NonTargetLesion],
                                new_lesions: List[NewLesion]) -> ResponseCategory:
        """Determine overall response assessment"""
        # If there are new lesions, it's PD
        if new_lesions:
            return ResponseCategory.PROGRESSIVE_DISEASE
            
        target_response = self.evaluate_target_response(target_lesions)
        non_target_response = self.evaluate_non_target_response(non_target_lesions)
        
        # Logic for overall response
        if target_response == ResponseCategory.COMPLETE_RESPONSE:
            if non_target_response == ResponseCategory.COMPLETE_RESPONSE:
                return ResponseCategory.COMPLETE_RESPONSE
            elif non_target_response == ResponseCategory.NON_CR_NON_PD:
                return ResponseCategory.PARTIAL_RESPONSE
                
        elif target_response == ResponseCategory.PROGRESSIVE_DISEASE:
            return ResponseCategory.PROGRESSIVE_DISEASE
            
        elif target_response == ResponseCategory.STABLE_DISEASE:
            if non_target_response != ResponseCategory.PROGRESSIVE_DISEASE:
                return ResponseCategory.STABLE_DISEASE
                
        return ResponseCategory.NOT_EVALUABLE

def convert_measurements_to_recist(measurements: List[Dict]) -> List[TargetLesion]:
    """Convert raw measurements to RECIST target lesions"""
    target_lesions = []
    
    for measurement in measurements:
        # Convert measurements to mm if needed
        current_size = measurement['current_size']['value']
        if measurement['current_size']['unit'] == 'cm':
            current_size *= 10
            
        prior_size = None
        if 'prior_size' in measurement:
            prior_size = measurement['prior_size']['value']
            if measurement['prior_size']['unit'] == 'cm':
                prior_size *= 10
                
        target_lesions.append(TargetLesion(
            location=measurement['location'],
            current_size=current_size,
            prior_size=prior_size
        ))
        
    return target_lesions

def assess_response(preprocessed_data: Dict) -> Dict:
    """Assess response using preprocessed report data"""
    evaluator = RECISTEvaluator()
    
    # Convert measurements to target lesions
    target_lesions = convert_measurements_to_recist(
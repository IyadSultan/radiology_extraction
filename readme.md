# Radiology Report Evaluation System

A Flask-based web application for evaluating and validating extracted information from radiology reports. This system provides an interface for human reviewers to verify automatically extracted data from medical imaging reports.

## Features

- Interactive web interface for report evaluation
- Support for multiple report reviewers
- Keyboard shortcuts for efficient reviewing
- Progress tracking
- Automatic data saving
- Structured validation of:
  - Clinical information
  - RECIST measurements
  - Target and non-target lesions
  - Response assessments
  - Anatomical classifications
  - ICD-O-3 site codes

## Prerequisites

- Python 3.8+
- Flask
- Pandas
- Bootstrap 5.1.3 (included via CDN)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd radiology-report-evaluation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install flask pandas
```

## Project Structure

```
radiology-report-evaluation/
├── app.py
├── templates/
│   ├── evaluate.html
│   ├── set_evaluator.html
│   └── no_reports.html
├── radiology_results_10.csv    # Input data file
└── human_evaluation_results.csv # Generated results file
```

## Running the Application

1. Ensure your radiology report data is in `radiology_results_10.csv`

2. Start the Flask application:
```bash
python app.py
```

3. Access the application at `http://localhost:5000`

## Usage

1. Enter your name as the evaluator when prompted

2. For each report:
   - Review the original report text on the left
   - Verify extracted fields on the right
   - Check/uncheck boxes to indicate correct/incorrect extractions
   - Add comments if needed
   - Submit or skip the current report

### Keyboard Shortcuts

- `1-9`: Toggle field checkboxes
- `Enter`: Submit evaluation
- `Esc`: Skip current report

## Data Format

### Input Data (radiology_results_10.csv)

Required columns:
- MRN
- EXAM_DATE
- PROCEDURE
- REPORT
- Additional extracted fields (target_lesions, non_target_lesions, etc.)

### Output Data (human_evaluation_results.csv)

Generated file containing:
- All verified fields
- Evaluator information
- Timestamp
- Verification status for each field

## Error Handling

- Missing reports display appropriate error messages
- Failed evaluations are logged and can be retried
- Invalid JSON data is handled gracefully

## Development

For development mode:
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

## Security Notes

- Change the `app.secret_key` in production
- Use proper session management
- Implement appropriate access controls
- Handle PHI according to HIPAA guidelines

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see the LICENSE file for details.
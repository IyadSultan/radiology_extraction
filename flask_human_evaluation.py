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
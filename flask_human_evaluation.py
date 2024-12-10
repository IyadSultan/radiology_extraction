from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

def load_data(current_index=0):
    try:
        # Load extraction results
        results_df = pd.read_csv('radiology_results_10.csv')
        
        # Drop rows where REPORT is missing
        results_df = results_df.dropna(subset=['REPORT'])
        
        # Load previous evaluations if they exist
        submitted_status = {}
        previous_evaluations = {}
        if Path('human_evaluation_results.csv').exists():
            evaluated_df = pd.read_csv('human_evaluation_results.csv')
            for _, row in evaluated_df.iterrows():
                mrn = str(row['MRN'])
                submitted_status[mrn] = True
                # Store field evaluations
                previous_evaluations[mrn] = {col: row[col] 
                                           for col in row.index 
                                           if not col in ['MRN', 'EXAM_DATE', 'evaluator_name', 'evaluation_timestamp']}
            
        if results_df.empty:
            print("No reports found.")
            return pd.DataFrame(), submitted_status, previous_evaluations
            
        return results_df, submitted_status, previous_evaluations
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame(), {}, {}

def save_evaluation(evaluation_data):
    try:
        # Convert evaluation data to DataFrame
        eval_df = pd.DataFrame([evaluation_data])
        
        # Add timestamp
        eval_df['evaluation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Append to existing file or create new one
        if Path('human_evaluation_results.csv').exists():
            # Read existing evaluations
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
    
    # Extract fields to evaluate
    fields_to_evaluate = {col: current_report[col] for col in current_report.index 
                         if col not in ['MRN', 'EXAM_DATE', 'PROCEDURE', 'REPORT']}
    
    # Get previous evaluation results if they exist
    field_results = {}
    if mrn in previous_evaluations:
        prev_eval = previous_evaluations[mrn]
        field_results = {field: prev_eval.get(field, True) for field in fields_to_evaluate.keys()}
    
    # Handle potential missing or NaN values
    exam_date = str(current_report['EXAM_DATE']) if 'EXAM_DATE' in current_report else 'N/A'
    procedure = str(current_report['PROCEDURE']) if 'PROCEDURE' in current_report else 'N/A'
    report = str(current_report['REPORT']) if 'REPORT' in current_report else 'Report not found'
    
    return render_template('evaluate.html',
                          mrn=mrn,
                          exam_date=exam_date,
                          procedure=procedure,
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
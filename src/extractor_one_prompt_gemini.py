import google.generativeai as genai
from google.generativeai import types
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import re

class RadiologyReportProcessor:
    def __init__(self, api_key: str):
        """Initialize the Gemini client with API key."""
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.0-flash-exp"

    def create_system_prompt(self) -> str:
        """Create the system prompt for report processing."""
        return """You are an expert radiologist specializing in structured data extraction from radiology reports.
        Extract all information in a single pass and return a COMPLETE JSON object with the following structure:
        
        {
            "analysis_timestamp": "<current_datetime>",
            "report_data": {
                "report": "<original_report_text>",
                "modality": "<imaging_modality>",
                "primary_location": "<anatomical_region>",
                "comparison_date": "<YYYY-MM-DD>",
                "clinical_history": "<history>",
                "indication": "<indication>",
                "target_lesions": [
                    {
                        "location": "<anatomical_location>",
                        "current_value": <numeric_value>,
                        "current_unit": "mm",
                        "standardized_value_mm": <value_in_mm>,
                        "prior_value": <numeric_value>,
                        "prior_unit": "mm",
                        "percent_change": <numeric_value>,
                        "response_category": "<response>",
                        "is_target": true
                    }
                ],
                "non_target_lesions": [
                    {
                        "location": "<anatomical_location>",
                        "current_value": <numeric_value>,
                        "current_unit": "mm",
                        "standardized_value_mm": <value_in_mm>,
                        "prior_value": <numeric_value>,
                        "prior_unit": "mm",
                        "percent_change": <numeric_value>,
                        "response_category": "<response>",
                        "is_target": false
                    }
                ],
                "new_lesions": [],
                "reported_response": "<response_text>",
                "recist_calculated_response": "<calculated_response>",
                "classifications": [
                    {
                        "class_name": "<category>",
                        "description": "<detailed_finding>"
                    }
                ],
                "other_findings": [
                    {
                        "item": "<finding_type>",
                        "description": "<detailed_description>"
                    }
                ]
            }
        }
        
        Follow these rules:
        1. Maintain exact JSON structure
        2. Convert all measurements to mm if given in cm
        3. Calculate percent change for lesions with prior measurements
        4. Classify findings into predefined categories
        5. Include only explicitly stated information
        6. Provide detailed descriptions for each finding
        """

    def preprocess_report(self, report_text: str) -> str:
        """Clean and standardize the report text."""
        # Remove extra whitespace and normalize line endings
        report_text = re.sub(r'\s+', ' ', report_text)
        report_text = report_text.replace('\n', ' ')
        return report_text.strip()

    def process_report(self, report_text: str) -> Dict[str, Any]:
        """Process the radiology report and return structured data."""
        try:
            # Prepare the prompt
            system_prompt = self.create_system_prompt()
            full_prompt = f"{system_prompt}\n\nReport Text:\n{report_text}"
            
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for consistent output
                    top_p=0.95,
                    top_k=20,
                    candidate_count=1,
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )
            )
            
            # Parse the JSON response
            result = json.loads(response.text)
            
            # Add analysis timestamp if not present
            if 'analysis_timestamp' not in result:
                result['analysis_timestamp'] = datetime.now().isoformat()
                
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "report_data": None
            }

    def validate_response(self, response: Dict[str, Any]) -> bool:
        """Validate the structure of the processed report."""
        required_fields = [
            'analysis_timestamp',
            'report_data',
            'report_data.modality',
            'report_data.primary_location',
            'report_data.classifications'
        ]
        
        try:
            for field in required_fields:
                if '.' in field:
                    parent, child = field.split('.')
                    if parent not in response or child not in response[parent]:
                        return False
                elif field not in response:
                    return False
            return True
        except Exception:
            return False

# Example usage
def main():
    # Initialize processor
    processor = RadiologyReportProcessor("YOUR_API_KEY")
    
    # Example report text
    report_text = """
 Diagnosis: Lymphoma.
Reason: Follow-up.
Comparison: Comparison made to the CT part of PET CT scan dated 2 February 2021.
Findings:
Neck CT with IV contrast:
There are unchanged enlarged hypoattenuating left cervical lymph nodes the large
st seen at levels III measuring 1 cm in short axis.
There are unchanged mildly enlarged right supraclavicular lymph nodes, none exce
eding 1 cm in short axis.
There is no new cervical lymph node enlargement.
Chest CT with IV contrast:
There are unchanged multiple hypoattenuating conglomerate mediastinal lymph node
s are encasing the aortic arch and its major branches with no significant narrow
ing, for example the subcarinal lymph node measures 2 cm in short axis with no s
ignificant change.
There is no new intrathoracic lymph node enlargement.
There is no significant axillary lymph node enlargement.
The mild pericardial thickening/effusion appears less significant.
There is a new trace amount of right pleural effusion.
There is no significant left pleural effusion.
There is no pulmonary nodule/mass.
There is no pulmonary consolidation.
Abdomen and pelvis CT scan with contrast:
Normal liver, spleen, pancreas, adrenals and both kidneys.
There is no hydronephrosis bilaterally.
Unremarkable gallbladder.
There is no intra or extrahepatic biliary tree dilatation.
There is no significant retroperitoneal or pelvic lymph node enlargement.
There is no ascites.
Unremarkable urinary bladder outline.
There is no vertebral collapse.
Impression:
There has been no significant interval change regarding the hypoattenuating resi
dual lymph nodes seen above the diaphragm compared to the CT part of PET/CT scan
dated 2 February 2021.
Dr. Mohammad Mujalli
Radiologist
    """
    
    # Process report
    result = processor.process_report(report_text)
    
    # Validate and print result
    if processor.validate_response(result):
        print(json.dumps(result, indent=2))
    else:
        print("Error: Invalid response structure")

if __name__ == "__main__":
    main()
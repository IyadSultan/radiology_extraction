import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

prompt = """
You are an expert radiologist specializing in structured data extraction from radiology reports. You must parse the following text as a radiology report and produce a single JSON object matching this schema:

{
  "report": string,               // Original report text
  "modality": string,             // Overall imaging modality (e.g., CT/MRI/PET, or specific modalities)
  "primary_location": string,     // Anatomical region of primary tumor (ICDO3 description, not the code)
  "study_date": string,           // Date of current study (if present in the text)
  "comparison_date": string,      // Date of any comparison study (if present)
  "clinical_history": string,     // Relevant clinical history
  "indication": string,           // Reason for the study
  "normal_study": boolean,        // True if normal, False if not, only if explicitly stated
  "target_lesions": [            // Array of target lesion measurements
    {
      "location": string,             
      "current_value": number,        
      "current_unit": string,         
      "standardized_value_mm": number,
      "prior_value": number,          
      "prior_unit": string,           
      "percent_change": number,       
      "response_category": string,    
      "is_target": boolean            
    }
  ],
  "non_target_lesions": [         // Array of non-target lesion measurements
    {
      "location": string,
      "current_value": number,
      "current_unit": string,
      "standardized_value_mm": number,
      "prior_value": number,
      "prior_unit": string,
      "percent_change": number,
      "response_category": string,
      "is_target": boolean
    }
  ],
  "new_lesions": [                // Array describing any new lesions
    {
      "location": string,
      "description": string
    }
  ],
  "reported_response": string,           // Response assessment as stated in the report
  "recist_calculated_response": string,  // Overall calculated RECIST response
  "classifications": [           // Classification categories identified in the report
    {
      "class_name": string,      // e.g. "Infection", "Metastasis", "Normal", etc.
      "description": string,     // Description or notes about this finding
      "confidence": number       // Confidence score if available (otherwise omit or set null)
    }
  ],
  "other_findings": [            // Miscellaneous findings not captured elsewhere
    {
      "item": string,            // Type or name of the finding
      "description": string      // Detailed description
    }
  ],
  "ICDO3_site": string,          // ICD-O-3 code for the primary location
  "ICDO3_site_term": string,     // Exact matched term for the primary location
  "ICDO3_site_similarity": number,
  "modality_specific": {         // Nested object for modality-specific details
    "modality_type": string,     // One of: "Mammography", "Chest CT", "Brain Imaging", "PET/CT", "Other"
    "mammography": {
      "birads_category": string,
      "breast_density": string,
      "masses": [ { "location": string, "description": string } ],
      "calcifications": [ { "location": string, "description": string } ],
      "architectural_distortion": boolean,
      "asymmetries": [ { "location": string, "description": string } ]
    },
    "chest_ct": {
      "halo_sign": [ { "location": string, "description": string } ],
      "cavitations": [ { "location": string, "description": string } ],
      "fungal_nodules": [ { "location": string, "description": string } ],
      "ground_glass_opacities": [ { "location": string, "description": string } ],
      "air_crescent_signs": [ { "location": string, "description": string } ],
      "other_fungal_findings": [ { "location": string, "description": string } ]
    },
    "brain_tumor": {
      "tumor_details": { "size": string, "description": string },
      "edema": { "description": string },
      "mass_effect": { "description": string },
      "enhancement_pattern": string,
      "brain_region": string,
      "additional_features": [ { "feature": string, "description": string } ]
    },
    "pet_ct": {
      "radiopharmaceutical": string,
      "injection_time": string,
      "blood_glucose": number,
      "uptake_time": string,
      "suv_measurements": [
        {
          "location": string,
          "suv_max": number,
          "suv_mean": number,
          "suv_peak": number,
          "prior_suv_max": number,
          "percent_change": number,
          "metabolic_volume": number,
          "measurement_time": string
        }
      ],
      "total_lesion_glycolysis": number,
      "background_suv": number,
      "metabolic_response": string,
      "uptake_pattern": string,
      "additional_findings": [ { "finding": string, "description": string } ]
    }
  }
}

Instructions:
1. **Read the provided radiology report** (which follows this instruction block).
2. **Extract** all data that fits into the above JSON structure:
   - **Study information:** modality, primary location, study/comparison dates, clinical history, indication, normal_study (if stated).
   - **Lesions (RECIST):** target and non-target lesions, new lesions. Each lesion should include size measurements, prior sizes, % change, response category, and whether it is a target lesion or not.
   - **Response assessment:** any explicitly stated assessment (reported_response), plus do not forget to assign the `recist_calculated_response` if possible (based on the logic in the code, e.g., presence of new lesions => "Progressive Disease", etc.).
   - **Classification results:** e.g. Normal, Infection, Metastasis, Primary tumor, Effusion, Trauma, Hemorrhage, Thrombosis, Lymphadenopathy, Others. Provide a description for each classification. If the confidence is not stated, either omit or set to null.
   - **Other findings:** anything not covered by the above categories but still reported. 
   - If the text implies a specific modality (like "Mammogram," "PET/CT," "Brain MRI," etc.), create the corresponding nested object (`mammography`, `pet_ct`, `chest_ct`, or `brain_tumor`) within `"modality_specific"`. 
   - Omit any sub-structures that do not apply. (For instance, if the modality is CT chest for fungal infections, fill `"chest_ct"`; if the modality is PET/CT, fill `"pet_ct"`, etc.)
3. **Output** must be strictly in valid JSON format, with no extra keys or commentary. No markdown. No additional explanation.
4. Do **not** leave any required JSON fields blank or missing—if something isn't mentioned, use `null`, an empty string, or an empty list (depending on the field type) rather than omitting the key.

Now, here is the radiology report text that you need to parse and convert into JSON. Your final output must be **only** one JSON object with the structure described above.

[BEGIN REPORT TEXT]
{{YOUR_Radiology_Report_HERE}}
[END REPORT TEXT]

"""

case= """
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

prompt = prompt.replace("{{YOUR_Radiology_Report_HERE}}", case)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [prompt]
               
    },
    
  ]
)

response = chat_session.send_message(prompt)

# Add debugging print statements
print("Raw response:")
print(response.text)

# Save the response to a JSON file
try:
    # Clean the response text by removing markdown code block markers
    cleaned_response = response.text
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response.replace('```json', '', 1)
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response.rsplit('```', 1)[0]
    cleaned_response = cleaned_response.strip()

     # Use regular expression to extract only the JSON object
    match = re.search(r'{.*}', cleaned_response, re.DOTALL)
    if match:
        cleaned_response = match.group(0)
    
    # Parse the cleaned response text as JSON
    json_data = json.loads(cleaned_response)
    
    # Save to a file with proper formatting
    with open('extractor_gemini_output.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print("Successfully saved response to extractor_gemini_output.json")
except json.JSONDecodeError as e:
    print(f"Error: Could not parse response as JSON: {e}")
    print("Response content:")
    print(response.text)
except IOError as e:
    print(f"Error: Could not save file: {e}")
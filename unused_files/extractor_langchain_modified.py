import re
import warnings
import json
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

# LangChain imports
from langchain.chains import create_extraction_chain_pydantic
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Enums, Models, and other classes remain largely unchanged
class ResponseCategory(str, Enum):
    """RECIST response categories"""
    COMPLETE_RESPONSE = "Complete Response"
    PARTIAL_RESPONSE = "Partial Response"
    STABLE_DISEASE = "Stable Disease"
    PROGRESSIVE_DISEASE = "Progressive Disease"
    NOT_EVALUABLE = "Not Evaluable"

class ModalityType(str, Enum):
    MAMMOGRAPHY = "Mammography"
    CHEST_CT = "Chest CT"
    BRAIN_IMAGING = "Brain Imaging"
    PET_CT = "PET/CT"
    OTHER = "Other"

class Classification(BaseModel):
    """Model for classification results"""
    class_name: str = Field(description="Classification category", alias="category")
    description: str = Field(description="Description of the finding")
    confidence: Optional[float] = Field(default=None)

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True

class OtherFinding(BaseModel):
    """Model for other findings"""
    item: str = Field(description="Type of finding")
    description: str = Field(description="Description of the finding")

class SUVMeasurement(BaseModel):
    location: str = Field(description="Anatomical location of measurement")
    suv_max: float = Field(description="Maximum SUV value")
    suv_mean: Optional[float] = None
    suv_peak: Optional[float] = None
    prior_suv_max: Optional[float] = None
    percent_change: Optional[float] = None
    metabolic_volume: Optional[float] = None
    measurement_time: Optional[str] = None

    def calculate_change(self):
        if self.prior_suv_max and self.prior_suv_max > 0:
            self.percent_change = ((self.suv_max - self.prior_suv_max) / self.prior_suv_max) * 100

class PETCTFindings(BaseModel):
    radiopharmaceutical: str
    injection_time: Optional[str] = None
    blood_glucose: Optional[float] = None
    uptake_time: Optional[str] = None
    suv_measurements: List[SUVMeasurement] = Field(default_factory=list)
    total_lesion_glycolysis: Optional[float] = None
    background_suv: Optional[float] = None
    metabolic_response: Optional[str] = None
    uptake_pattern: Optional[str] = None
    additional_findings: List[Dict[str, str]] = Field(default_factory=list)

class MammographyFindings(BaseModel):
    birads_category: str
    breast_density: str
    masses: List[Dict[str, str]] = Field(default_factory=list)
    calcifications: List[Dict[str, str]] = Field(default_factory=list)
    architectural_distortion: Optional[bool] = None
    asymmetries: List[Dict[str, str]] = Field(default_factory=list)

class ChestCTFungalFindings(BaseModel):
    halo_sign: List[Dict[str, str]] = Field(default_factory=list)
    cavitations: List[Dict[str, str]] = Field(default_factory=list)
    fungal_nodules: List[Dict[str, str]] = Field(default_factory=list)
    ground_glass_opacities: List[Dict[str, str]] = Field(default_factory=list)
    air_crescent_signs: List[Dict[str, str]] = Field(default_factory=list)
    other_fungal_findings: List[Dict[str, str]] = Field(default_factory=list)

class BrainTumorFindings(BaseModel):
    tumor_details: Dict[str, Any]
    edema: Optional[Dict[str, str]] = None
    mass_effect: Optional[Dict[str, str]] = None
    enhancement_pattern: Optional[str] = None
    brain_region: str
    additional_features: List[Dict[str, str]] = Field(default_factory=list)

class ModalitySpecific(BaseModel):
    modality_type: ModalityType
    mammography: Optional[MammographyFindings] = None
    chest_ct: Optional[ChestCTFungalFindings] = None
    brain_tumor: Optional[BrainTumorFindings] = None
    pet_ct: Optional[PETCTFindings] = None

class Lesion(BaseModel):
    """Model for lesion measurements"""
    location: str = Field(description="Anatomical location of lesion")
    current_value: float = Field(description="Current measurement value")
    current_unit: str = Field(description="Unit of measurement (mm/cm)")
    standardized_value_mm: float = Field(description="Measurement standardized to mm")
    prior_value: Optional[float] = Field(None, description="Previous measurement value")
    prior_unit: Optional[str] = Field(None, description="Unit of previous measurement")
    percent_change: Optional[float] = Field(None, description="Percent change from prior")
    response_category: str = Field(description="Response category for this lesion")
    is_target: bool = Field(description="Whether this is a target lesion")

    def __init__(self, **data):
        super().__init__(**data)
        self.standardize_measurements()
    
    def standardize_measurements(self):
        """Standardize measurements to mm and calculate percent change"""
        # Convert current value to mm
        if self.current_unit == 'cm':
            self.standardized_value_mm = self.current_value * 10
        elif self.current_unit == 'mm':
            self.standardized_value_mm = self.current_value
        
        # Calculate percent change if prior measurements exist
        if self.prior_value is not None and self.prior_unit is not None:
            prior_mm = self.prior_value * 10 if self.prior_unit == 'cm' else self.prior_value
            
            if prior_mm > 0:
                self.percent_change = ((self.standardized_value_mm - prior_mm) / prior_mm) * 100
                
                # Set response category based on RECIST criteria
                if self.is_target:
                    if self.standardized_value_mm == 0:
                        self.response_category = "Complete Response"
                    elif self.percent_change <= -30:
                        self.response_category = "Partial Response"
                    elif self.percent_change >= 20:
                        self.response_category = "Progressive Disease"
                    else:
                        self.response_category = "Stable Disease"

class RadiologyReport(BaseModel):
    """Main model for radiology report extraction"""
    report: str = Field(description="Full report text")
    modality: str = Field(description="Imaging modality")
    primary_location: str = Field(description="Primary anatomical location")
    language_model: str = Field(description="Name of language model used for extraction")
    study_date: Optional[str] = Field(None, description="Date of current study")
    comparison_date: Optional[str] = Field(None, description="Date of comparison study")
    clinical_history: Optional[str] = Field(None, description="Clinical history")
    indication: Optional[str] = Field(None, description="Study indication")
    normal_study: Optional[bool] = Field(None, description="True if the study is normal, False if not")
    target_lesions: List[Lesion] = Field(default_factory=list)
    non_target_lesions: List[Lesion] = Field(default_factory=list)
    new_lesions: List[Lesion] = Field(default_factory=list)
    classifications: List[Classification] = Field(default_factory=list)
    other_findings: List[OtherFinding] = Field(default_factory=list)
    reported_response: Optional[str] = Field(None, description="Response assessment as stated")
    recist_calculated_response: Optional[str] = Field(None, description="Calculated RECIST response")
    ICDO3_site: Optional[str] = None
    ICDO3_site_term: Optional[str] = None
    ICDO3_site_similarity: Optional[float] = None
    modality_specific: Optional[Dict] = None

    def calculate_recist_response(self) -> str:
        """Calculate RECIST response based on measurements"""
        try:
            # Check for new lesions first
            if self.new_lesions:
                return ResponseCategory.PROGRESSIVE_DISEASE.value
            
            # Get sum of target lesions
            current_sum = sum(
                lesion.current_value 
                for lesion in self.target_lesions 
                if lesion.current_value is not None
            )
            prior_sum = sum(
                lesion.prior_value 
                for lesion in self.target_lesions 
                if lesion.prior_value is not None
            )
            
            # If no measurable disease
            if not current_sum:
                return ResponseCategory.NOT_EVALUABLE.value
            
            # If complete disappearance of all target lesions
            if current_sum == 0:
                return ResponseCategory.COMPLETE_RESPONSE.value
            
            # Calculate percent change if we have prior measurements
            if prior_sum:
                percent_change = ((current_sum - prior_sum) / prior_sum) * 100
            else:
                return ResponseCategory.NOT_EVALUABLE.value

            # Apply RECIST criteria
            if percent_change <= -30:
                return ResponseCategory.PARTIAL_RESPONSE.value
            elif percent_change >= 20:
                return ResponseCategory.PROGRESSIVE_DISEASE.value
            else:
                return ResponseCategory.STABLE_DISEASE.value
                
        except Exception as e:
            logger.error(f"Error calculating RECIST response: {e}")
            return ResponseCategory.NOT_EVALUABLE.value

    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True

import re
from sentence_transformers import SentenceTransformer

# Other imports and definitions...

class ResponseExtractor:
    def __init__(self):
        # Predefined templates for response categories
        self.templates = {
            "Stable Disease": [
                "no significant interval change",
                "findings are stable",
                "unchanged compared to prior study"
            ],
            "Progressive Disease": [
                "evidence of progression",
                "increased size or number of lesions",
                "new lesions identified"
            ],
            "Partial Response": [
                "partial decrease in size",
                "reduction in lesion size",
                "significant interval decrease"
            ],
            "Complete Response": [
                "complete resolution of lesions",
                "no evidence of residual disease"
            ]
        }
        # Initialize the semantic similarity model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Precompute embeddings for templates
        self.template_embeddings = {
            category: self.model.encode(phrases, convert_to_tensor=True)
            for category, phrases in self.templates.items()
        }

    def extract_response(self, impression_text: str) -> Optional[str]:
        # Preprocess the impression section
        impression_text = impression_text.strip().lower()

        # Check for explicit matches using regex patterns
        for category, phrases in self.templates.items():
            for phrase in phrases:
                if re.search(re.escape(phrase), impression_text):
                    return category

        # Use semantic similarity as a fallback
        impression_embedding = self.model.encode([impression_text], convert_to_tensor=True)
        max_similarity = 0
        best_match = None

        for category, embeddings in self.template_embeddings.items():
            similarities = torch.nn.functional.cosine_similarity(impression_embedding, embeddings)
            max_sim = similarities.max().item()
            if max_sim > max_similarity:
                max_similarity = max_sim
                best_match = category

        # Apply a similarity threshold (e.g., 0.7) to ensure confidence
        return best_match if max_similarity > 0.7 else None




class EnhancedSimilarityMapper:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._embeddings = None
        self._texts = None
        
    def _compute_embeddings(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
    
    def _calculate_similarity(self, query_embedding, stored_embeddings):
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.cpu().numpy()
        if torch.is_tensor(stored_embeddings):
            stored_embeddings = stored_embeddings.cpu().numpy()

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if len(stored_embeddings.shape) == 1:
            stored_embeddings = stored_embeddings.reshape(1, -1)
            
        return cosine_similarity(query_embedding, stored_embeddings)[0]

class ModalityMapper(EnhancedSimilarityMapper):
    def __init__(self, modalities_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.modalities_df = modalities_df
        combined_texts = (modalities_df['Modality'].fillna('') + ' ' + modalities_df['Description'].fillna(''))
        self._texts = combined_texts.tolist()
        self._embeddings = self._compute_embeddings(self._texts)
        self.modalities_df['processed_text'] = combined_texts.str.lower()

    def find_closest_modality(self, text: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            query_embedding = self._compute_embeddings([text])
            similarities = self._calculate_similarity(query_embedding, self._embeddings)
            idx = similarities.argmax()

            if similarities[idx] < threshold:
                return None

            result = {
                'standard_name': self.modalities_df.iloc[idx]['Modality'],
                'category': self.modalities_df.iloc[idx]['Category'],
                'similarity': float(similarities[idx])
            }

            text_lower = text.lower()
            if ('pet' in text_lower and 'ct' in text_lower) or 'pet/ct' in text_lower:
                result['modality_type'] = ModalityType.PET_CT
            elif any(term in text_lower for term in ['mammogram', 'mammography', 'breast imaging']):
                result['modality_type'] = ModalityType.MAMMOGRAPHY
            elif ('chest' in text_lower and 'ct' in text_lower) or 'thoracic ct' in text_lower:
                result['modality_type'] = ModalityType.CHEST_CT
            elif any(brain_term in text_lower for brain_term in ['brain', 'head', 'cranial']) and \
                 any(modal_term in text_lower for modal_term in ['ct', 'mri', 'magnetic']):
                result['modality_type'] = ModalityType.BRAIN_IMAGING
            else:
                result['modality_type'] = ModalityType.OTHER

            return result

        except Exception as e:
            logger.error(f"Error in modality matching: {str(e)}")
            return None

class LocationMapper(EnhancedSimilarityMapper):
    def __init__(self, topography_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.topography_df = topography_df
        self.topography_df['combined_terms'] = topography_df['term'].fillna('') + ' ' + \
                                               topography_df['synonyms'].fillna('')
        self._texts = self.topography_df['combined_terms'].tolist()
        self._embeddings = self._compute_embeddings(self._texts)
        
    def find_closest_location(self, text: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            query_embedding = self._compute_embeddings([text])
            similarities = self._calculate_similarity(query_embedding, self._embeddings)
            top_k = 3
            top_indices = similarities.argsort()[-top_k:][::-1]
            top_similarities = similarities[top_indices]
            
            valid_matches = top_similarities >= threshold
            if not any(valid_matches):
                return None
                
            best_idx = top_indices[0]
            best_similarity = top_similarities[0]
            
            return {
                'code': self.topography_df.iloc[best_idx]['ICDO3'],
                'term': self.topography_df.iloc[best_idx]['term'],
                'similarity': float(best_similarity),
                'alternative_matches': [
                    {
                        'term': self.topography_df.iloc[idx]['term'],
                        'similarity': float(similarities[idx])
                    }
                    for idx in top_indices[1:] if similarities[idx] >= threshold
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in location matching: {str(e)}")
            return None

@dataclass
class ExtractionDependencies:
    modality_mapper: ModalityMapper
    location_mapper: LocationMapper

# Update the standard system prompt to be more explicit about the required JSON structure
# standard_system_prompt = """You are an expert radiologist specializing in structured data extraction from radiology reports.

# Extract information and return it in EXACTLY this JSON format:
# {
#     "report": "<full report text>",
#     "modality": "<imaging modality>",
#     "primary_location": "<anatomical region>",
#     "study_date": "<study date if available>",
#     "comparison_date": "<comparison date if available>",
#     "clinical_history": "<clinical history>",
#     "indication": "<study indication>",
#     "target_lesions": [
#         {
#             "location": "<anatomical location>",
#             "current_value": <numeric value>,
#             "current_unit": "<unit>",
#             "prior_value": <numeric value or null>,
#             "prior_unit": "<unit or null>",
#             "is_target": true
#         }
#     ],
#     "non_target_lesions": [
#         {
#             "location": "<anatomical location>",
#             "current_value": <numeric value>,
#             "current_unit": "<unit>",
#             "prior_value": <numeric value or null>,
#             "prior_unit": "<unit or null>",
#             "is_target": false
#         }
#     ],
#     "new_lesions": [],
#     "reported_response": "<response assessment if stated>",
#     "classifications": [
#         {
#             "class_name": "<category>",
#             "description": "<finding description>"
#         }
#     ],
#     "other_findings": [
#         {
#             "item": "<finding name>",
#             "description": "<finding description>"
#         }
#     ]
# }

# Important:
# 1. ALL fields "report", "modality", and "primary_location" are REQUIRED
# 2. Convert all measurements to numeric values (e.g., "1 cm" should be {"current_value": 1, "current_unit": "cm"})
# 3. Return valid JSON only
# 4. Ensure all arrays are present even if empty
# 5. Use null for optional fields if information is not available

# Extract from these categories:
# - Normal findings
# - Infection
# - Metastasis
# - Primary tumor
# - Effusion
# - Trauma
# - Hemorrhage
# - Thrombosis
# - Lymphadenopathy
# - Others"""

standard_system_prompt = """
You are an expert radiologist specializing in structured data extraction from radiology reports.
    
    Extract all information in a single pass and return a complete RadiologyReport object:
    
    1. Study Information:
       - modality (CT/MRI/PET/etc; mention specific modalities for Mammography, Chest CT (Fungal Infections), Brain Imaging)
       - primary_location (anatomical region of primary tumor using ICDO3 description (not code), not the imaging site)
       - dates (study and comparison)
       - clinical_history
       - indication
       - normal_study (True if the study is normal, False if not)
       
    2. Measurements and RECIST:
       - For each lesion: current size, prior size if available
       - Mark target vs non-target lesions
       - Record new lesions if any
       
    3. Response Assessment:
       - reported_response: Extract the response assessment as stated
       - recist_calculated_response will be calculated automatically
       = RECIST Calculation Instructions:
            A. RECIST responses should be calculated as follows:
            - **Complete Response (CR):** Complete disappearance of all target lesions.
            - **Partial Response (PR):** A decrease of 30% or more in the sum of the diameters of target lesions compared to the baseline or prior measurements.
            - **Stable Disease (SD):** A change in the sum of diameters of target lesions between a 30% decrease and a 20% increase.
            - **Progressive Disease (PD):** An increase of 20% or more in the sum of diameters of target lesions, or the appearance of new lesions.

            B. Ensure the percentage change is calculated as:
            \[
            \text{Percent Change} = \frac{\text{Current Sum of Target Lesions} - \text{Prior Sum of Target Lesions}}{\text{Prior Sum of Target Lesions}} \times 100
            \]

            C. If there are new lesions, the RECIST response should always be **Progressive Disease** (PD), irrespective of size changes in target lesions.

            D. If no measurable disease is present, classify as **Not Evaluable** (NE).

            Return the `recist_calculated_response` as one of:
            - "Complete Response"
            - "Partial Response"
            - "Stable Disease"
            - "Progressive Disease"
            - "Not Evaluable"

       
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
       - Lymphadenopathy: Enlarged lymph nodes
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

# Modality-specific prompt
modality_system_prompt = """
You are an expert radiologist focusing on modality-specific findings extraction.
    Based on the identified modality type, extract relevant specialized information:

    For Mammography:
    - BI-RADS category
    - Breast density
    - Details of masses and calcifications
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

    For PET/CT:
    - Radiopharmaceutical type (e.g., FDG)
    - Injection and uptake times
    - Blood glucose level
    - SUV measurements for each lesion:
        * SUV max
        * SUV mean (if available)
        * SUV peak (if available)
        * Prior SUV values for comparison
        * Metabolic volume
    - Background SUV
    - Total lesion glycolysis
    - Overall metabolic response assessment
    - Uptake patterns
    - Additional metabolic findings

    Return findings in the appropriate structure based on modality type.
    Only include explicitly stated findings.
    Provide detailed descriptions for each finding.
    For PET/CT specifically, ensure all SUV measurements are properly recorded with anatomical locations.
"""

class EnhancedRadiologyExtractor:
    def __init__(self, modalities_df: pd.DataFrame, topography_df: pd.DataFrame):
        # Initialize modality and location mappers
        self.modality_mapper = ModalityMapper(modalities_df)
        self.location_mapper = LocationMapper(topography_df)

        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.model_name = "gpt-4o"

        # Create parser
        self.parser = PydanticOutputParser(pydantic_object=RadiologyReport)

        # Create prompt template with format instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert radiologist. Extract structured information from this radiology report.
            Follow these guidelines strictly:

            1. For measurements:
               - Extract numeric values and units separately
               - Convert all measurements to standardized mm values
               - Include location for each measurement
               - Mark target vs non-target lesions
               - Calculate percent change if prior measurements exist
               - Set response category based on RECIST criteria

            2. For target lesions:
               - Location must be specific (e.g., "Mediastinal lymph nodes (subcarinal)")
               - Current value must be in numeric form
               - Units must be either 'mm' or 'cm'
               - Include standardized value in mm
               - Calculate percent change if prior measurements exist
               - Set is_target to true

            3. For non-target lesions:
               - Follow same format as target lesions
               - Set is_target to false

            4. Response categories must be one of:
               - "Complete Response"
               - "Partial Response"
               - "Stable Disease"
               - "Progressive Disease"

            {format_instructions}
            """),
            ("human", "{text}")
        ])

        # Create chain
        self.chain = self.prompt | self.llm | self.parser

        # Add classification keywords
        self.classification_keywords = {
            "Normal": [
                "normal",
                "unremarkable",
                "no significant abnormality",
                "within normal limits"
            ],
            "Infection": [
                "infection",
                "abscess",
                "empyema",
                "pneumonia",
                "septic emboli",
                "fungal infection",
                "bacterial infection",
                "viral infection",
                "tuberculosis"
            ],
            "Metastasis": [
                "metastasis",
                "metastatic disease",
                "secondary lesion",
                "spread to",
                "distant metastasis",
                "metastatic involvement"
            ],
            "Primary Tumor": [
                "primary tumor",
                "neoplasm",
                "carcinoma",
                "adenocarcinoma",
                "squamous cell carcinoma",
                "tumor origin"
            ],
            "Edema": [
                "edema",
                "swelling",
                "peritumoral edema",
                "cerebral edema",
                "soft tissue edema"
            ],
            "Atelectasis": [
                "atelectasis",
                "collapsed lung",
                "lobar collapse",
                "segmental collapse",
                "partial collapse"
            ],
            "Calcification": [
                "calcifications",
                "arterial calcifications",
                "soft tissue calcifications",
                "calcified mass",
                "calcified plaque"
            ],
            "Obstruction/Stenosis": [
                "obstruction",
                "stenosis",
                "narrowing",
                "stricture",
                "blockage",
                "compressive effect",
                "encasement"
            ],
            "Ascites": [
                "ascites",
                "peritoneal fluid",
                "fluid in abdomen",
                "abdominal distention",
                "peritoneal effusion"
            ],
            "Other": [
                "abnormality",
                "changes consistent with",
                "evidence of",
                "findings suggestive of"
            ]
        }

        # Initialize the ResponseExtractor
        self.response_extractor = ResponseExtractor()

    def classify_findings(self, report_text: str) -> List[Classification]:
        """
        Dynamically classify findings in the radiology report.
        """
        classifications = []
        lower_text = report_text.lower()

        for class_name, keywords in self.classification_keywords.items():
            for keyword in keywords:
                if keyword in lower_text:
                    # Extract a relevant description from the report
                    match_index = lower_text.find(keyword)
                    snippet = report_text[max(0, match_index - 50): match_index + 50]
                    classifications.append(
                        Classification(
                            class_name=class_name,
                            description=f"Detected finding related to {class_name}: '{snippet.strip()}'."
                        )
                    )
                    break  # Avoid duplicate classifications for the same category

        return classifications

    def process_report(self, text: str) -> RadiologyReport:
        """Process a single radiology report"""
        logger.info("Starting extraction process with LangChain...")
        
        try:
            # Extract information using the chain
            extracted = self.chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })

            if not extracted:
                raise ValueError("No data extracted from report")

            # Ensure report text and language model are set
            extracted.report = text
            extracted.language_model = self.model_name

            # Enhanced location matching
            if extracted.primary_location:
                location_match = self.location_mapper.find_closest_location(
                    extracted.primary_location,
                    threshold=0.7
                )
                
                if location_match:
                    extracted.ICDO3_site = location_match['code']
                    extracted.ICDO3_site_term = location_match['term']
                    extracted.ICDO3_site_similarity = location_match['similarity']
                    logger.info(f"Location match found: {location_match['term']} "
                              f"(similarity: {location_match['similarity']:.3f})")
                else:
                    extracted.ICDO3_site = None
                    extracted.ICDO3_site_term = "Unknown"
                    extracted.ICDO3_site_similarity = None
                    logger.warning("No matching location found above threshold")

            # Calculate RECIST response
            extracted.recist_calculated_response = extracted.calculate_recist_response()
            
            # Add dynamic classification
            extracted.classifications = self.classify_findings(text)

            return extracted

        except Exception as e:
            logger.error(f"Error processing report: {str(e)}")
            raise




example_report_text = """
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
# # Usage example:
# resources_path = Path(__file__).parent.parent / 'resources'
# modalities_df = pd.read_csv(resources_path / 'modalities.csv')
# topography_df = pd.read_csv(resources_path / 'ICDO3Topography.csv')
# extractor = EnhancedRadiologyExtractor(modalities_df, topography_df)
# result = extractor.process_report(example_report_text)
# print(result)

# Add at the end of the file, replacing the simple print(result)
if __name__ == "__main__":
    try:
        resources_path = Path(__file__).parent.parent / 'resources'
        modalities_df = pd.read_csv(resources_path / 'modalities.csv')
        topography_df = pd.read_csv(resources_path / 'ICDO3Topography.csv')
        
        extractor = EnhancedRadiologyExtractor(modalities_df, topography_df)
        result = extractor.process_report(example_report_text)
        
        # Enhanced output formatting
        print("\nExtracted Information:")
        print("=" * 50)
        
        # Basic Information
        print("\nStudy Information:")
        print(f"Modality: {result.modality}")
        print(f"Primary Location: {result.primary_location}")
        if result.ICDO3_site:
            print(f"ICD-O Site: {result.ICDO3_site} ({result.ICDO3_site_term})")
            print(f"Location Match Similarity: {result.ICDO3_site_similarity:.3f}")
        
        # Response Assessment
        print("\nResponse Assessment:")
        print(f"Reported Response: {result.reported_response}")
        print(f"RECIST Calculated: {result.recist_calculated_response}")
        
        # Measurements with enhanced formatting
        if result.target_lesions:
            print("\nTarget Lesions:")
            for i, lesion in enumerate(result.target_lesions, 1):
                print(f"\n{i}. Location: {lesion.location}")
                print(f"   Current: {lesion.current_value} {lesion.current_unit}")
                if lesion.prior_value:
                    print(f"   Prior: {lesion.prior_value} {lesion.prior_unit}")
                    print(f"   Change: {lesion.percent_change:.1f}%")
                print(f"   Response: {lesion.response_category}")
        
        # Enhanced visualization of findings
        if result.classifications:
            print("\nClassified Findings:")
            for classification in result.classifications:
                print(f"\n{classification.class_name}:")
                print(f"  {classification.description}")
        
        # Save detailed output
        output_file = Path(f'example_output_langchain_{extractor.model_name}.json')
        output_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'report_data': result.dict(exclude_none=True)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

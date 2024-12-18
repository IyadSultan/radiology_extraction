# LangChain imports
from langchain.chains import create_extraction_chain_pydantic
from langchain_openai import ChatOpenAI

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


class SUVMeasurement(BaseModel):
    """Model for SUV measurements"""
    location: str = Field(description="Anatomical location of measurement")
    suv_max: float = Field(description="Maximum SUV value")
    suv_mean: Optional[float] = Field(description="Mean SUV value")
    suv_peak: Optional[float] = Field(description="Peak SUV value")
    prior_suv_max: Optional[float] = Field(description="Previous maximum SUV value")
    percent_change: Optional[float] = Field(default=None)
    metabolic_volume: Optional[float] = Field(description="Metabolic tumor volume (MTV)")
    measurement_time: Optional[str] = Field(description="Time post-injection")

    def calculate_change(self):
        if self.prior_suv_max and self.prior_suv_max > 0:
            self.percent_change = ((self.suv_max - self.prior_suv_max) / self.prior_suv_max) * 100

class PETCTFindings(BaseModel):
    """PET/CT specific findings"""
    radiopharmaceutical: str = Field(description="Type of tracer used (e.g., FDG)")
    injection_time: Optional[str] = Field(description="Time of tracer injection")
    blood_glucose: Optional[float] = Field(description="Blood glucose level at time of scan")
    uptake_time: Optional[str] = Field(description="Uptake time post-injection")
    suv_measurements: List[SUVMeasurement] = Field(default_factory=list)
    total_lesion_glycolysis: Optional[float] = Field(description="Total lesion glycolysis (TLG)")
    background_suv: Optional[float] = Field(description="Background SUV measurement")
    metabolic_response: Optional[str] = Field(description="Overall metabolic response assessment")
    uptake_pattern: Optional[str] = Field(description="Pattern of tracer uptake")
    additional_findings: List[Dict[str, str]] = Field(default_factory=list)

class MammographyFindings(BaseModel):
    """Mammography-specific findings"""
    birads_category: str = Field(description="BI-RADS assessment category")
    breast_density: str = Field(description="Breast density category")
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
    """Brain imaging findings focused on tumors"""
    tumor_details: Dict[str, Any] = Field(...)
    edema: Optional[Dict[str, str]] = None
    mass_effect: Optional[Dict[str, str]] = None
    enhancement_pattern: Optional[str] = None
    brain_region: str = Field(description="Specific brain region affected")
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

class RadiologyReport(BaseModel):
    """Main model for radiology report extraction"""
    report: str = Field(description="Full report text")
    modality: str = Field(description="Imaging modality")
    primary_location: str = Field(description="Primary anatomical location")
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

class EnhancedSimilarityMapper:
    """Base class for enhanced similarity mapping using Sentence Transformers"""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._embeddings = None
        self._texts = None
        
    def _compute_embeddings(self, texts):
        """Compute embeddings for a list of texts"""
        return self.model.encode(texts, convert_to_tensor=True)
    
    def _calculate_similarity(self, query_embedding, stored_embeddings):
        """Calculate cosine similarity between query and stored embeddings"""
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
    """Maps imaging modalities using enhanced similarity search"""
    def __init__(self, modalities_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.modalities_df = modalities_df
        
        # Combine Modality and Description into a single text field
        combined_texts = (modalities_df['Modality'].fillna('') + ' ' + modalities_df['Description'].fillna(''))
        self._texts = combined_texts.tolist()
        self._embeddings = self._compute_embeddings(self._texts)
        
        # Create processed text for further usage
        self.modalities_df['processed_text'] = combined_texts.str.lower()
        
    def find_closest_modality(self, text: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """Find closest modality using semantic similarity"""
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

            # Enhanced modality type detection logic
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
    """Maps anatomical locations using enhanced similarity search"""
    def __init__(self, topography_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.topography_df = topography_df
        
        # Preprocess and combine terms for better matching
        self.topography_df['combined_terms'] = topography_df['term'].fillna('') + ' ' + \
                                             topography_df['synonyms'].fillna('')
        self._texts = self.topography_df['combined_terms'].tolist()
        self._embeddings = self._compute_embeddings(self._texts)
        
    def find_closest_location(self, text: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """Find closest location using semantic similarity"""
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

    def batch_find_closest_locations(self, texts: list, threshold: float = 0.3) -> list:
        """Batch process multiple location queries efficiently"""
        try:
            query_embeddings = self._compute_embeddings(texts)
            similarities = self._calculate_similarity(query_embeddings, self._embeddings)
            
            results = []
            for i, sims in enumerate(similarities):
                idx = sims.argmax()
                if sims[idx] < threshold:
                    results.append(None)
                    continue
                    
                results.append({
                    'code': self.topography_df.iloc[idx]['ICDO3'],
                    'term': self.topography_df.iloc[idx]['term'],
                    'similarity': float(sims[idx])
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in batch location matching: {str(e)}")
            return [None] * len(texts)

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
           {{"class_name": "<category>", "description": "<detailed finding>"}}
       ]
       other_findings: [
           {{"item": "<finding type>", "description": "<detailed description>"}}
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
    def __init__(self, modalities_df: pd.DataFrame, topography_df: pd.DataFrame, model_name: str = "gpt-4o-mini"):
        self.modality_mapper = ModalityMapper(modalities_df)
        self.location_mapper = LocationMapper(topography_df)
        self.model_name = model_name
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.5)
        
        # Create standard extraction chain
        self.standard_prompt = ChatPromptTemplate.from_messages([
            ("system", standard_system_prompt),
            ("human", "{text}")
        ])
        self.standard_chain = (
            self.standard_prompt 
            | self.llm.with_structured_output(RadiologyReport)
        )
        
        # Create modality-specific chain
        self.modality_prompt = ChatPromptTemplate.from_messages([
            ("system", modality_specific_prompt),
            ("human", "Modality Type: {modality_type}\n\nReport Text: {text}")
        ])
        self.modality_chain = (
            self.modality_prompt 
            | self.llm.with_structured_output(ModalitySpecific)
        )

    def process_report(self, text: str) -> RadiologyReport:
        """Process a single radiology report using enhanced two-pass extraction"""
        logger.info(f"Starting extraction process with LangChain using {self.model_name}...")
        
        try:
            # First pass: Standard extraction
            extracted = self.standard_chain.invoke({
                "text": text
            })
            
            # Add the original report text
            extracted.report = text
            
            # Location matching
            if extracted.primary_location:
                loc_match = self.location_mapper.find_closest_location(extracted.primary_location)
                if loc_match:
                    extracted.ICDO3_site = loc_match['code']
                    extracted.ICDO3_site_term = loc_match['term']
                    extracted.ICDO3_site_similarity = loc_match['similarity']
            
            # Calculate RECIST response
            extracted.recist_calculated_response = extracted.calculate_recist_response()
            
            # Second pass: Modality-specific extraction
            try:
                # Detect modality type from first 200 characters
                modality_result = self.modality_mapper.find_closest_modality(text[:200])
                
                if modality_result and 'modality_type' in modality_result:
                    modality_type = modality_result['modality_type']
                    logger.info(f"Detected modality type: {modality_type}")
                    
                    if modality_type != ModalityType.OTHER:
                        modality_specific_result = self.modality_chain.invoke({
                            "text": text,
                            "modality_type": modality_type.value  # Use the string value of the enum
                        })
                        extracted.modality_specific = modality_specific_result
                
            except Exception as e:
                logger.error(f"Error in modality-specific extraction: {str(e)}")
                # Continue with standard extraction result even if modality-specific fails
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error processing report: {e}")
            raise

# Add the modality-specific system prompt
modality_specific_prompt = """
You are an expert radiologist focusing on modality-specific findings extraction.
Based on the identified modality type, extract relevant specialized information in the following format:

For Mammography:
{
    "modality_type": "MAMMOGRAPHY",
    "birads_category": "string",
    "breast_density": "string",
    "masses": [
        {
            "location": "string",
            "size": "string",
            "characteristics": "string"
        }
    ],
    "calcifications": [
        {
            "location": "string",
            "type": "string",
            "distribution": "string"
        }
    ],
    "architectural_distortion": "string",
    "asymmetries": "string"
}

For Chest CT (Fungal Infections):
{
    "modality_type": "CHEST_CT",
    "halo_sign": {
        "present": false,
        "details": "string"
    },
    "cavitations": [
        {
            "location": "string",
            "size": "string",
            "characteristics": "string"
        }
    ],
    "nodules": [
        {
            "location": "string",
            "size": "string",
            "characteristics": "string"
        }
    ],
    "ground_glass_opacities": [
        {
            "location": "string",
            "extent": "string",
            "pattern": "string"
        }
    ],
    "air_crescent_signs": [
        {
            "location": "string",
            "details": "string"
        }
    ]
}

For Brain Imaging:
{
    "modality_type": "BRAIN_IMAGING",
    "tumor": {
        "location": "string",
        "size": "string",
        "characteristics": "string"
    },
    "edema": {
        "present": false,
        "extent": "string",
        "location": "string"
    },
    "mass_effect": {
        "present": false,
        "details": "string"
    },
    "enhancement_pattern": "string",
    "additional_features": [
        {
            "feature": "string",
            "description": "string"
        }
    ]
}

For PET/CT:
{
    "modality_type": "PET_CT",
    "radiopharmaceutical": {
        "type": "string",
        "dose": "string",
        "injection_time": "string",
        "uptake_time": "string"
    },
    "blood_glucose": "string",
    "lesions": [
        {
            "location": "string",
            "suv_max": 0.0,
            "suv_mean": 0.0,
            "suv_peak": 0.0,
            "prior_suv_max": 0.0,
            "metabolic_volume": "string"
        }
    ],
    "background_suv": 0.0,
    "total_lesion_glycolysis": "string",
    "metabolic_response": "string",
    "additional_findings": [
        {
            "location": "string",
            "description": "string"
        }
    ]
}

Important:
1. Return VALID JSON only
2. Include all numeric measurements as numbers
3. Use null for missing optional values
4. Include all arrays even if empty
5. Only extract information explicitly stated in the report
6. Ensure output matches the exact structure for the specified modality type

The modality type will be provided in the input. Extract only the relevant information for that modality type.
"""

# Read example report from file
with open('C:\\Users\\USER\\Documents\\radiologyExtraction\\examples\\example1.txt', 'r') as f:
    example_report_text = f.read()
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
        
        model_name = "gpt-4o-mini"  # Specify the model name
        extractor = EnhancedRadiologyExtractor(modalities_df, topography_df, model_name)
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
        
        # Save detailed output with model name in filename
        output_file = Path(f'example_output_langchain_{model_name.replace("-", "_")}.json')
        output_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'model_name': model_name,  # Include model name in output
            'report_data': result.dict(exclude_none=True)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nDetailed results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

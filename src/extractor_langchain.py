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
    current_value: Optional[float] = Field(None, description="Current measurement value")
    current_unit: Optional[str] = Field(None, description="Unit of measurement (e.g., mm, cm)")
    prior_value: Optional[float] = Field(None, description="Previous measurement value")
    prior_unit: Optional[str] = Field(None, description="Unit of previous measurement")
    response_category: Optional[str] = Field(None, description="Response category for this lesion")
    percent_change: Optional[float] = Field(None, description="Percent change from prior")

class RadiologyReport(BaseModel):
    """Main model for radiology report extraction"""
    report: str = Field(description="Full report text")
    modality: str = Field(description="Imaging modality")
    primary_location: str = Field(description="Primary anatomical location")
    study_date: Optional[str] = Field(None, description="Date of current study")
    comparison_date: Optional[str] = Field(None, description="Date of comparison study")
    clinical_history: Optional[str] = Field(None, description="Clinical history")
    indication: Optional[str] = Field(None, description="Study indication")
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
standard_system_prompt = """You are an expert radiologist specializing in structured data extraction from radiology reports.

Extract information and return it in EXACTLY this JSON format:
{
    "report": "<full report text>",
    "modality": "<imaging modality>",
    "primary_location": "<anatomical region>",
    "study_date": "<study date if available>",
    "comparison_date": "<comparison date if available>",
    "clinical_history": "<clinical history>",
    "indication": "<study indication>",
    "target_lesions": [
        {
            "location": "<anatomical location>",
            "current_value": <numeric value>,
            "current_unit": "<unit>",
            "prior_value": <numeric value or null>,
            "prior_unit": "<unit or null>",
            "is_target": true
        }
    ],
    "non_target_lesions": [
        {
            "location": "<anatomical location>",
            "current_value": <numeric value>,
            "current_unit": "<unit>",
            "prior_value": <numeric value or null>,
            "prior_unit": "<unit or null>",
            "is_target": false
        }
    ],
    "new_lesions": [],
    "reported_response": "<response assessment if stated>",
    "classifications": [
        {
            "class_name": "<category>",
            "description": "<finding description>"
        }
    ],
    "other_findings": [
        {
            "item": "<finding name>",
            "description": "<finding description>"
        }
    ]
}

Important:
1. ALL fields "report", "modality", and "primary_location" are REQUIRED
2. Convert all measurements to numeric values (e.g., "1 cm" should be {"current_value": 1, "current_unit": "cm"})
3. Return valid JSON only
4. Ensure all arrays are present even if empty
5. Use null for optional fields if information is not available

Extract from these categories:
- Normal findings
- Infection
- Metastasis
- Primary tumor
- Effusion
- Trauma
- Hemorrhage
- Thrombosis
- Lymphadenopathy
- Others"""

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
        self.modality_mapper = ModalityMapper(modalities_df)
        self.location_mapper = LocationMapper(topography_df)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
        # Create parser
        self.parser = PydanticOutputParser(pydantic_object=RadiologyReport)
        
        # Create prompt template with format instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert radiologist. Extract structured information from this radiology report.
            Follow these guidelines strictly:
            
            1. For measurements:
               - Extract numeric values and units separately
               - Convert all measurements to cm if given in mm
               - Include location for each measurement
               
            2. For classifications:
               - Categorize findings into appropriate classes (e.g., Lymphadenopathy, Effusion, Normal)
               - Each classification must have a class_name and description
               - Common categories include: Normal, Infection, Metastasis, Primary tumor, Effusion, etc.
               
            3. For lesions:
               - Record location, size, and units
               - Note any changes from prior studies
               
            4. For other findings:
               - Each finding must have an item and description
               - Format as objects with these fields
               
            {format_instructions}
            """),
            ("human", "{text}")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm | self.parser

    def process_report(self, text: str) -> RadiologyReport:
        """Process a single radiology report"""
        logger.info("Starting extraction process with LangChain...")
        
        try:
            # Extract information using the chain with invoke instead of run
            extracted = self.chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            if not extracted:
                raise ValueError("No data extracted from report")
            
            # Ensure report text is set
            extracted.report = text
            
            # Location matching
            if extracted.primary_location:
                loc_match = self.location_mapper.find_closest_location(extracted.primary_location)
                if loc_match:
                    extracted.ICDO3_site = loc_match['code']
                    extracted.ICDO3_site_term = loc_match['term']
                    extracted.ICDO3_site_similarity = loc_match['similarity']

            # Convert measurements if needed
            for lesion_list in [extracted.target_lesions, extracted.non_target_lesions, extracted.new_lesions]:
                for lesion in lesion_list:
                    if lesion.current_unit == 'mm':
                        lesion.current_value = lesion.current_value / 10
                        lesion.current_unit = 'cm'
            
            # Calculate RECIST response
            extracted.recist_calculated_response = extracted.calculate_recist_response()
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error processing report: {e}")
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
        output_file = Path('example_output_langchain_gpt4omini.json')
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

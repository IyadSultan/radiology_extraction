import warnings
import json
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
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
    PET_CT = "PET/CT"
    OTHER = "Other"

# Model classes remain the same as they define the data structure
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
        """Calculate percent change in SUV max if prior value exists"""
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

# Update ModalitySpecific model to include PET/CT
class ModalitySpecific(BaseModel):
    """Container for modality-specific findings"""
    modality_type: ModalityType
    mammography: Optional[MammographyFindings] = None
    chest_ct: Optional[ChestCTFungalFindings] = None
    brain_tumor: Optional[BrainTumorFindings] = None
    pet_ct: Optional[PETCTFindings] = None


class NonTargetResponse(str, Enum):
    COMPLETE_RESPONSE = "Complete Response"
    NON_CR_NON_PD = "Non-CR/Non-PD"  
    PROGRESSIVE_DISEASE = "Progressive Disease"
    NOT_EVALUABLE = "Not Evaluable"


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
    def assess_non_target_response(self):
        if not self.is_target:
            if self.standardized_value_mm == 0:
                self.response_category = NonTargetResponse.COMPLETE_RESPONSE.value
            elif self.unequivocal_progression:  # Need to add this flag
                self.response_category = NonTargetResponse.PROGRESSIVE_DISEASE.value
            else:
                self.response_category = NonTargetResponse.NON_CR_NON_PD.value
    
    def standardize_measurements(self):
        if self.current_value is not None:
            if self.current_unit == 'cm':
                self.standardized_value_mm = self.current_value * 10
            elif self.current_unit == 'mm':
                self.standardized_value_mm = self.current_value
            else:
                # If unit is unknown, skip calculation or handle gracefully
                self.standardized_value_mm = None

        if self.prior_value is not None and self.prior_unit is not None:
            if self.prior_unit == 'cm':
                prior_mm = self.prior_value * 10
            elif self.prior_unit == 'mm':
                prior_mm = self.prior_value
            else:
                prior_mm = None

            # Only compute if both standardized_value_mm and prior_mm are available and non-zero
            if self.standardized_value_mm is not None and prior_mm is not None and prior_mm > 0:
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
    primary_location: str = Field(description="Anatomical region of primary tumor using ICDO3 description (not code)")
    comparison_date: Optional[str] = Field(default=None)
    clinical_history: Optional[str] = Field(default=None)
    indication: Optional[str] = Field(default=None)
    normal_study: Optional[bool] = Field(default=None, description="True if the study is normal, False if not")
    
    target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    non_target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    new_lesions: List[Dict[str, Any]] = Field(default_factory=list)
    
    reported_response: Optional[str] = Field(default=None)
    response_category: Optional[str] = Field(
        default=None, 
        description="Choose from COMPLETE_RESPONSE, PARTIAL_RESPONSE, STABLE_DISEASE, PROGRESSIVE_DISEASE, NOT_EVALUABLE"
    )
    recist_calculated_response: Optional[str] = Field(default=None)
    
    classifications: List[ClassificationResult] = Field(default_factory=list)
    other_findings: List[OtherFinding] = Field(default_factory=list)
    
    ICDO3_site: Optional[str] = Field(default=None)
    ICDO3_site_term: Optional[str] = Field(default=None)
    ICDO3_site_similarity: Optional[float] = Field(default=None)
    
    modality_specific: Optional[ModalitySpecific] = None

    def calculate_sum_of_diameters(self) -> float:
        return sum(l.standardized_value_mm for l in self.target_lesions 
                  if l.standardized_value_mm is not None)

    def calculate_recist_response(self) -> str:
        if not self.target_lesions and not self.non_target_lesions:
            return ResponseCategory.NOT_EVALUABLE.value
            
        if self.new_lesions:
            return ResponseCategory.PROGRESSIVE_DISEASE.value

        target_responses = [l.response_category for l in self.target_lesions if l.response_category]
        non_target_responses = [l.response_category for l in self.non_target_lesions if l.response_category]
        
        # Handle missing measurements
        if not all(target_responses) or not all(non_target_responses):
            return ResponseCategory.NOT_EVALUABLE.value

        # Complete Response: All targets CR, all non-targets CR, no new lesions
        if (all(r == ResponseCategory.COMPLETE_RESPONSE.value for r in target_responses) and
            all(r == NonTargetResponse.COMPLETE_RESPONSE.value for r in non_target_responses)):
            return ResponseCategory.COMPLETE_RESPONSE.value
            
        # Progressive Disease: Any target PD, any non-target unequivocal PD, or new lesions
        if (any(r == ResponseCategory.PROGRESSIVE_DISEASE.value for r in target_responses) or
            any(r == NonTargetResponse.PROGRESSIVE_DISEASE.value for r in non_target_responses)):
            return ResponseCategory.PROGRESSIVE_DISEASE.value
            
        # Partial Response: >=30% decrease from baseline in target lesions, no PD in non-targets
        if (all(r == ResponseCategory.PARTIAL_RESPONSE.value for r in target_responses) and
            not any(r == NonTargetResponse.PROGRESSIVE_DISEASE.value for r in non_target_responses)):
            return ResponseCategory.PARTIAL_RESPONSE.value
            
        # Stable Disease: Neither PR nor PD criteria met
        return ResponseCategory.STABLE_DISEASE.value

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

def get_modality_type(procedure: str = "CHEST CT") -> ModalityType:
    """
    Determine modality type from procedure string
    """
    if not procedure:
        return ModalityType.OTHER
        
    procedure_lower = procedure.lower()
    
    if 'pet' in procedure_lower and 'ct' in procedure_lower:
        return ModalityType.PET_CT
    elif 'chest' in procedure_lower and 'ct' in procedure_lower:
        return ModalityType.CHEST_CT
    elif 'mammogra' in procedure_lower:
        return ModalityType.MAMMOGRAPHY
    elif any(brain_term in procedure_lower for brain_term in ['brain', 'head', 'cranial']) and \
         any(modal_term in procedure_lower for modal_term in ['ct', 'mri', 'magnetic']):
        return ModalityType.BRAIN_IMAGING
    else:
        return ModalityType.OTHER

def validate_measurements(self):
    if self.standardized_value_mm is not None:
        if self.is_target and self.standardized_value_mm < 10:
            raise ValueError("Target lesions must be ≥10mm in longest diameter")
        if self.is_target and self.anatomical_site == "lymph_node" and self.standardized_value_mm < 15:
            raise ValueError("Target lymph nodes must be ≥15mm in short axis")


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
    """Dependencies for extraction process"""
    procedure: str  # Add procedure as a dependency
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
       - primary_location (anatomical region of primary tumor using ICDO3 description (not code), not the imaging site)
       - date (comparison study date)
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
    - Details of masses and calcifications
    - Architectural distortion
    - Asymmetries

    For Chest CT (Fungal Infections):
    - Halo sign presence and details
    - Cavitations
    - Nodules consistent with fungal disease
    - Ground-glass opacities
    - Air-crescent signs
    - Other fungal-specific findings, e.g.  Tree-in-bud opacities, Consolidation patterns, Any other relevant findings

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

    For each finding, provide:
    {
        "location": "specific anatomical location",
        "description": "detailed characteristics",
        "size": "measurements if available",
        "additional_features": "any other relevant details"
    }

    Even if findings are negative, explicitly state this in the appropriate categories.
    Document both presence AND absence of key findings.
    Include measurements when available.
    For PET/CT specifically, ensure all SUV measurements are properly recorded with anatomical locations.
    """
)

# Tool definitions for agents
@standard_extraction_agent.tool
async def find_modality(ctx: RunContext[ExtractionDependencies], text: str):
    """Get modality type from procedure"""
    modality_type = get_modality_type(ctx.deps.procedure)
    result = {
        'standard_name': modality_type.value,
        'modality_type': modality_type
    }
    logger.info(f"Modality detection result: {result}")
    return result

@standard_extraction_agent.tool
async def find_location(ctx: RunContext[ExtractionDependencies], text: str):
    """Enhanced location matching tool"""
    result = ctx.deps.location_mapper.find_closest_location(text)
    logger.info(f"Location mapping result: {result}")
    return result

# Add tools to modality-specific agent
modality_specific_agent.tool(find_modality)
modality_specific_agent.tool(find_location)

class EnhancedRadiologyExtractor:
    """Enhanced main class for extracting information from radiology reports"""
    
    def __init__(self, topography_df: pd.DataFrame):
        self.location_mapper = LocationMapper(topography_df)

    async def process_report(self, text: str, procedure: str) -> RadiologyReport:
        """Process a single radiology report using enhanced two-pass extraction"""
        try:
            logger.info("Starting enhanced extraction process...")
            
            deps = ExtractionDependencies(
                procedure=procedure,
                location_mapper=self.location_mapper
            )
            
            # First pass: Standard extraction
            result = await standard_extraction_agent.run(text, deps=deps)
            result.data.report = text
            
            # ... (previous location matching and RECIST calculation code) ...
            
            # Get modality type from procedure
            modality_type = get_modality_type(procedure)
            logger.info(f"Detected modality type: {modality_type}")
            
            if modality_type != ModalityType.OTHER:
                try:
                    modality_specific_result = await modality_specific_agent.run(
                        text,
                        context={
                            'modality_type': modality_type,
                            'procedure': procedure
                        },
                        deps=deps
                    )
                    
                    # Initialize the modality_specific structure
                    result.data.modality_specific = ModalitySpecific(modality_type=modality_type)
                    
                    if modality_type == ModalityType.CHEST_CT:
                        # Extract findings specifically mentioned in the report
                        findings_dict = {
                            'halo_sign': [],
                            'cavitations': [],
                            'fungal_nodules': [],
                            'ground_glass_opacities': [],
                            'air_crescent_signs': [],
                            'other_fungal_findings': []
                        }
                        
                        # Look for explicit negative findings
                        if "no pulmonary nodule" in text.lower():
                            findings_dict['fungal_nodules'].append({
                                "location": "pulmonary",
                                "description": "No pulmonary nodules identified",
                                "size": "N/A",
                                "additional_features": "Negative finding"
                            })
                            
                        if "no pulmonary consolidation" in text.lower():
                            findings_dict['other_fungal_findings'].append({
                                "location": "pulmonary",
                                "description": "No consolidation identified",
                                "size": "N/A",
                                "additional_features": "Negative finding"
                            })
                            
                        # Add the findings to the result
                        result.data.modality_specific.chest_ct = ChestCTFungalFindings(**findings_dict)
                    
                except Exception as e:
                    logger.error(f"Error in modality-specific extraction: {str(e)}")
                    
            return result.data
            
        except Exception as e:
            logger.error("Error processing report", exc_info=True)
            raise

async def process_batch(reports_df: pd.DataFrame,
                       extractor: EnhancedRadiologyExtractor,
                       batch_size: int = 10) -> pd.DataFrame:
    """Process a batch of reports with enhanced extraction"""
    results = []
    total = len(reports_df)
    
    logger.info(f"Processing {total} reports in batches of {batch_size}")
    
    for idx, row in enumerate(reports_df.iterrows(), 1):
        try:
            _, report_row = row
            logger.info(f"Processing report {idx}/{total}")
            
            result = await extractor.process_report(
                report_row['REPORT'],
                report_row['PROCEDURE']  # Pass procedure here
            )
            
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

async def main():
    """Main function with enhanced processing capabilities"""
    try:
        # Update the resources path resolution
        resources_path = Path(__file__).parent.parent / 'resources'
        
        logger.info("Loading reference data...")
        try:
           
            topography_df = pd.read_csv(resources_path / 'ICDO3Topography.csv')
        except FileNotFoundError as e:
            logger.error(f"Error loading resource files: {e}")
            logger.error(f"Please ensure files exist in: {resources_path}")
            return
            
        # Initialize enhanced extractor
        extractor = EnhancedRadiologyExtractor(topography_df)
        
        # Check if processing a batch
        input_file = Path('Results.csv')
        if input_file.exists():
            logger.info("Processing batch from Results.csv...")
            reports_df = pd.read_csv(input_file)
            results_df = await process_batch(reports_df, extractor)
            
            # Save results with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'radiology_results_{timestamp}.csv'
            results_df.to_csv(output_file, index=False)
            
            # Generate and save analysis summary
            summary_data = {
                'total_reports': len(results_df),
                'successful_reports': len(results_df[~results_df['error'].notna()]),
                'failed_reports': len(results_df[results_df['error'].notna()]),
                'modalities_found': results_df['modality'].value_counts().to_dict(),
                'average_similarity_scores': {
                    'location': results_df['ICDO3_site_similarity'].mean()
                }
            }
            
            with open(f'analysis_summary_{timestamp}.json', 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            logger.info(f"Batch processing complete. Results saved to {output_file}")
            return
            # Process single example report
        # resources_path = Path(__file__).parent.parent / 'examples'
        # with open(resources_path / 'example1.txt', 'r') as f:
        #     example_report = f.read()
        # Example report processing for testing
        example_report = """
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
        
        # Process example report
        procedure = "Chest CT"  # Default value

        
        # Process example report with procedure
        result = await extractor.process_report(example_report, procedure=procedure)
        
        # Enhanced output formatting
        print("\nExtracted Information:")
        print("=" * 50)
        
        # Basic Information
        print("\nStudy Information:")
        print(f"Procedure: {procedure}")
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
        output_file = Path('example_output.json')
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

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
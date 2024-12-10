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
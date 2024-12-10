# extractor.py

import warnings
import json
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class ResponseCategory(str, Enum):
    """RECIST response categories"""
    COMPLETE_RESPONSE = "Complete Response"
    PARTIAL_RESPONSE = "Partial Response"
    STABLE_DISEASE = "Stable Disease"
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
    is_target: bool = Field(default=False, description="Whether this is a target lesion")

    def __init__(self, **data):
        super().__init__(**data)
        self.standardize_measurements()
    
    def standardize_measurements(self):
        """Convert measurements to mm and calculate response"""
        if self.current_value:
            # Convert to mm
            if self.current_unit == 'cm':
                self.standardized_value_mm = self.current_value * 10
            elif self.current_unit == 'mm':
                self.standardized_value_mm = self.current_value
            
            # Calculate change if prior measurement exists
            if self.prior_value and self.prior_unit:
                prior_mm = self.prior_value * 10 if self.prior_unit == 'cm' else self.prior_value
                self.percent_change = ((self.standardized_value_mm - prior_mm) / prior_mm) * 100
                
                # Determine response category for target lesions
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
    # Study Information
    modality: str = Field(description="Imaging modality (CT/MRI/PET/etc)")
    primary_location: str = Field(description="Anatomical region of primary tumor using ICDO3")
    study_date: Optional[str] = Field(default=None)
    comparison_date: Optional[str] = Field(default=None)
    clinical_history: Optional[str] = Field(default=None)
    indication: Optional[str] = Field(default=None, description="Indication for imaging study")
    
    # Measurements and RECIST
    target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    non_target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    new_lesions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Response Assessment
    reported_response: Optional[str] = Field(
        default=None,
        description="Response assessment as stated in the report, using standardized RECIST response categories"
    )
    recist_calculated_response: Optional[str] = Field(
        default=None,
        description="Response calculated using RECIST 1.1 criteria"
    )
    
    # Classifications
    classifications: List[ClassificationResult] = Field(default_factory=list)
    other_findings: List[OtherFinding] = Field(default_factory=list)
    
    # Location Coding
    ICDO3_site: Optional[str] = Field(default=None)
    ICDO3_site_term: Optional[str] = Field(default=None)
    ICDO3_site_similarity: Optional[float] = Field(default=None)

    def calculate_recist_response(self) -> str:
        """Calculate overall RECIST response"""
        # Check for new lesions first
        if self.new_lesions:
            return ResponseCategory.PROGRESSIVE_DISEASE.value
            
        # Get responses from target lesions
        target_responses = [lesion.response_category for lesion in self.target_lesions 
                          if lesion.response_category]
                          
        if not target_responses:
            return ResponseCategory.NOT_EVALUABLE.value
            
        # Calculate overall response
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
            return {
                'standard_name': self.modalities_df.iloc[idx]['Modality'],
                'category': self.modalities_df.iloc[idx]['Category'],
                'similarity': float(sims[idx])
            }
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

# Initialize extraction agent
extraction_agent = Agent(
    "openai:gpt-4o",
    retries=3,
    deps_type=ExtractionDependencies,
    result_type=RadiologyReport,
    system_prompt="""
    You are an expert radiologist specializing in structured data extraction from radiology reports.
    
    Extract all information in a single pass and return a complete RadiologyReport object:
    
    1. Study Information:
       - modality (CT/MRI/PET/etc)
       - primary_location (anatomical area of primary tumor using ICDO3)
       - dates (study and comparison)
       
    2. Measurements and RECIST:
       - For each lesion: current size, prior size if available
       - Mark target vs non-target lesions
       - Record new lesions if any
       
    3. Response Assessment:
       - reported_response: Extract the response assessment as stated by the radiologist
       - recist_calculated_response will be calculated automatically
       
    4. Classifications:
       Classify the report into all applicable categories with descriptions:
       - Normal: No significant abnormality
       - Infection: Any infectious process (pneumonia, abscess, etc.)
       - Metastasis: Evidence of metastatic disease
       - Primary tumor: Primary tumor findings
       - Effusion: Pleural effusion, pericardial effusion, or ascites
       - Trauma: Evidence of injury or trauma
       - Hemorrhage: Any bleeding or hemorrhage
       - Thrombosis: Blood clots, emboli, or thrombosis
       - Others: Any other significant findings
       
    5. Structure output:
       classifications: [
           {"class_name": "<category>", "description": "<detailed finding>"}
       ]
       other_findings: [
           {"item": "<finding type>", "description": "<detailed description>"}
       ]
       
    Only include information explicitly stated in the report.
    Provide detailed, specific descriptions for each classification.
    """
)

@extraction_agent.tool
async def find_modality(ctx: RunContext[ExtractionDependencies], text: str):
    """Tool for matching imaging modalities"""
    result = ctx.deps.modality_mapper.find_closest_modality(text)
    print(f"Modality mapping result: {result}")
    return result

@extraction_agent.tool
async def find_location(ctx: RunContext[ExtractionDependencies], text: str):
    """Tool for matching anatomical locations"""
    result = ctx.deps.location_mapper.find_closest_location(text)
    print(f"Location mapping result: {result}")
    return result

class RadiologyExtractor:
    """Main class for extracting information from radiology reports"""
    
    def __init__(self, modalities_df: pd.DataFrame, topography_df: pd.DataFrame):
        self.modality_mapper = ModalityMapper(modalities_df)
        self.location_mapper = LocationMapper(topography_df)

    async def process_report(self, text: str) -> RadiologyReport:
        """Process a single radiology report"""
        try:
            logger.info("Starting extraction...")
            
            # Create dependencies
            deps = ExtractionDependencies(
                modality_mapper=self.modality_mapper,
                location_mapper=self.location_mapper
            )
            
            # Extract using AI model
            result = await extraction_agent.run(text, deps=deps)
            
            # Match location code for body region
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
            
            return result.data
            
        except Exception as e:
            logger.error("Error processing report", exc_info=True)
            raise

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
        
        # Example report
        example_report = """
        CT CHEST WITH CONTRAST
        
        CLINICAL HISTORY: Follow-up of lung cancer.
        
        FINDINGS:
        1. Right upper lobe mass measures 3.2 x 2.8 cm (previously 4.1 x 3.5 cm),
           representing partial response to therapy.
        2. Multiple bilateral pulmonary nodules consistent with metastases,
           largest measuring 8mm in right lower lobe (previously 12mm).
        3. Small right pleural effusion.
        4. Right lower lobe consolidation suggesting pneumonia.
        5. Small subsegmental pulmonary embolism in the left lower lobe.
        
        IMPRESSION:
        1. Partial response to therapy in primary lung tumor and metastases.
        2. New right lower lobe pneumonia.
        3. Acute pulmonary embolism.
        4. Small right pleural effusion.
        """
        
        # Process report
        result = await extractor.process_report(example_report)
        
        # Print results
        print("\nExtracted Information:")
        print("=" * 50)
        
        # Basic Information
        print("\nStudy Information:")
        print(f"Modality: {result.modality}")
        print(f"Body Region: {result.primary_location}")
        if result.ICDO3_site:
            print(f"ICD-O Site: {result.ICDO3_site} ({result.ICDO3_site_term})")
        
        # Response Assessment
        print("\nResponse Assessment:")
        print(f"Reported Response: {result.reported_response}")
        print(f"RECIST Calculated Response: {result.recist_calculated_response}")
        
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
        
        # Other Findings
        if result.other_findings:
            print("\nOther Findings:")
            for finding in result.other_findings:
                print(f"\n{finding.item}:")
                print(f"  {finding.description}")
        
        # Save complete output
        output_file = Path('example_output.json')
        with open(output_file, 'w') as f:
            json.dump(result.dict(exclude_none=True), f, indent=2)
        print(f"\nComplete results saved to {output_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Target Lesions: {len(result.target_lesions)}")
        print(f"Non-target Lesions: {len(result.non_target_lesions)}")
        print(f"Classifications: {len(result.classifications)}")
        print(f"Other Findings: {len(result.other_findings)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
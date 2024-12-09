# extractor.py

import warnings
import json
import pandas as pd
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Data Models
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

    def __init__(self, **data):
        super().__init__(**data)
        self.standardize_measurements()
    
    def standardize_measurements(self):
        """Convert measurements to mm"""
        if self.current_value:
            if self.current_unit == 'cm':
                self.standardized_value_mm = self.current_value * 10
            elif self.current_unit == 'mm':
                self.standardized_value_mm = self.current_value
            
            if self.prior_value and self.prior_unit:
                prior_mm = self.prior_value * 10 if self.prior_unit == 'cm' else self.prior_value
                self.percent_change = ((self.standardized_value_mm - prior_mm) / prior_mm) * 100
                
                # Determine response category
                if self.percent_change <= -30:
                    self.response_category = 'PR'
                elif self.percent_change >= 20:
                    self.response_category = 'PD'
                else:
                    self.response_category = 'SD'

class RadiologyReport(BaseModel):
    """Model for structured radiology report data"""
    modality: str = Field(description="Imaging modality (CT/MRI/PET/etc)")
    body_region: str = Field(description="Anatomical region studied")
    study_date: Optional[str] = Field(default=None)
    comparison_date: Optional[str] = Field(default=None)
    clinical_history: Optional[str] = Field(default=None)
    
    target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    non_target_lesions: List[RECISTMeasurement] = Field(default_factory=list)
    new_lesions: List[Dict[str, Any]] = Field(default_factory=list)
    
    tumor_response: Optional[str] = Field(default=None)
    overall_assessment: Optional[str] = Field(default=None)
    
    ICDO3_site: Optional[str] = Field(default=None)
    ICDO3_site_term: Optional[str] = Field(default=None)
    ICDO3_site_similarity: Optional[float] = Field(default=None)

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
       - body_region (anatomical area)
       - dates (study and comparison)
       
    2. Measurements:
       - For each lesion: current size, prior size if available
       - Convert all measurements to mm internally
       - Calculate percent changes where applicable
       - Determine RECIST categories (PR/SD/PD) based on changes
       
    3. Findings:
       - target lesions with measurements
       - non-target lesions
       - new lesions if any
       
    4. Assessment:
       - Overall response category
       - Changes from prior studies
       
    Only include information explicitly stated in the report.
    Return all fields in the proper structure without requiring follow-up questions.
    """
)

# Define tools after agent initialization
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
            if result.data.body_region:
                logger.info(f"Matching location for: {result.data.body_region}")
                location_match = await find_location(
                    RunContext(deps=deps, retry=0, tool_name="find_location"),
                    result.data.body_region
                )
                
                if location_match:
                    logger.info(f"Found location match: {location_match}")
                    result.data.ICDO3_site = location_match['code']
                    result.data.ICDO3_site_term = location_match['term']
                    result.data.ICDO3_site_similarity = location_match['similarity']
                else:
                    logger.warning("No location match found")
            
            return result.data
            
        except Exception as e:
            logger.error("Error processing report", exc_info=True)
            raise

async def main():
    """Main function for demonstration"""
    try:
        # Example data
        modalities_df = pd.DataFrame({
            'Modality': ['CT', 'MRI', 'PET/CT', 'X-ray', 'Ultrasound'],
            'Category': ['Cross-sectional', 'Cross-sectional', 'Nuclear', 'Radiograph', 'Ultrasound']
        })
        
        topography_df = pd.DataFrame({
            'ICDO3': ['C34.1', 'C34.2', 'C34.3', 'C34.9'],
            'term': ['Upper lobe of lung', 'Middle lobe of lung', 'Lower lobe of lung', 'Lung']
        })
        
        # Initialize extractor
        extractor = RadiologyExtractor(modalities_df, topography_df)
        
        # Example report
        example_report = """
        CT CHEST WITH CONTRAST
        FINDINGS: Right upper lobe mass measures 3.2 x 2.8 cm.
        """
        
        # Process report
        result = await extractor.process_report(example_report)
        print("\nExtracted Information:")
        print(json.dumps(result.dict(), indent=2))
        
    except Exception as e:
        logger.error("Error in main execution", exc_info=True)
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
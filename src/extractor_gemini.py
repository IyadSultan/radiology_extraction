from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import asyncio
import nest_asyncio
from dataclasses import dataclass
from typing import List
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import torch

load_dotenv()
nest_asyncio.apply()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MeasurementInfo(BaseModel):
    """Simple measurement model"""
    location: str = Field(description="Anatomical location")
    size: str = Field(description="Size with unit")
    is_target: bool = Field(default=False, description="Target lesion status")

class Finding(BaseModel):
    """Simple finding model"""
    category: str = Field(description="Finding category")
    details: str = Field(description="Finding details")

class SimpleReport(BaseModel):
    """Minimal report model for Gemini compatibility"""
    text: str = Field(default="")
    modality: str = Field(default="")
    area: str = Field(default="")
    study_date: str = Field(default="")
    comparison: str = Field(default="")
    history: str = Field(default="")
    findings: List[Finding] = Field(default_factory=list)
    measurements: List[MeasurementInfo] = Field(default_factory=list)
    impression: str = Field(default="")

class SimilarityMapper:
    """Basic similarity mapping"""
    def __init__(self, reference_df: pd.DataFrame):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = reference_df
        self.terms = self.df['term'].fillna('').tolist()
        self._embeddings = self.model.encode(self.terms, convert_to_tensor=True)
    
    def find_match(self, text: str) -> dict:
        """Find closest matching term"""
        if not text:
            return {"term": "", "code": "", "score": 0.0}
        
        try:
            query_embedding = self.model.encode([text], convert_to_tensor=True)
            similarities = cosine_similarity(
                query_embedding.cpu().numpy(),
                self._embeddings.cpu().numpy()
            )[0]
            
            best_idx = similarities.argmax()
            return {
                "term": str(self.df.iloc[best_idx]['term']),
                "code": str(self.df.iloc[best_idx].get('ICDO3', '')),
                "score": float(similarities[best_idx])
            }
        except Exception as e:
            logger.error(f"Matching error: {str(e)}")
            return {"term": "", "code": "", "score": 0.0}

@dataclass
class ExtractorDeps:
    """Simple dependencies"""
    mapper: SimilarityMapper

extraction_agent = Agent(
    "gemini-2.0-flash-exp",
    retries=3,
    deps_type=ExtractorDeps,
    result_type=SimpleReport,
    system_prompt="""
    Extract information from the radiology report into these categories:

    1. Basic Information:
       - modality: Type of imaging (CT, MRI, etc.)
       - area: Anatomical region(s)
       - study_date: Current study date
       - comparison: Prior study details
       - history: Clinical history/indication
    
    2. Measurements:
       For each lesion:
       - location: Specific anatomical site
       - size: Measurement with unit (e.g., "1 cm", "20 mm")
       - is_target: true for significant measurable lesions
    
    3. Findings:
       For each significant finding:
       - category: Type (e.g., Lymphadenopathy, Effusion, Normal)
       - details: Complete description as stated
    
    4. Impression:
       Overall assessment and changes

    Include both normal and abnormal findings.
    Use exact measurements and descriptions from the text.
    """
)

@extraction_agent.tool
async def find_match(ctx: RunContext[ExtractorDeps], text: str) -> dict:
    """Find matching term in reference data"""
    return ctx.deps.mapper.find_match(text)

class Extractor:
    def __init__(self, reference_df: pd.DataFrame):
        self.mapper = SimilarityMapper(reference_df)

    async def process_report(self, text: str) -> SimpleReport:
        try:
            logger.info("Starting extraction...")
            
            deps = ExtractorDeps(mapper=self.mapper)
            result = await extraction_agent.run(text)
            result.data.text = text
            
            return result.data
            
        except Exception as e:
            logger.error("Processing error", exc_info=True)
            raise

async def main():
    try:
        resources_path = Path(__file__).parent.parent / 'resources'
        reference_df = pd.read_csv(resources_path / 'ICDO3Topography.csv')
        
        extractor = Extractor(reference_df)
        
        example_report = """
 Diagnosis: Lymphoma.
Reason: Follow-up.
Comparison: Comparison made to the CT part of PET CT scan dated 2 February 2021.
Findings:
Neck CT with IV contrast:
There are unchanged enlarged hypoattenuating left cervical lymph nodes the largest seen at levels III measuring 1 cm in short axis.
There are unchanged mildly enlarged right supraclavicular lymph nodes, none exceeding 1 cm in short axis.
There is no new cervical lymph node enlargement.
Chest CT with IV contrast:
There are unchanged multiple hypoattenuating conglomerate mediastinal lymph nodes are encasing the aortic arch and its major branches with no significant narrowing, for example the subcarinal lymph node measures 2 cm in short axis with no significant change.
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
There has been no significant interval change regarding the hypoattenuating residual lymph nodes seen above the diaphragm compared to the CT part of PET/CT scan dated 2 February 2021.
Dr. Mohammad Mujalli
Radiologist
        """
        
        result = await extractor.process_report(example_report)
        
        output_file = Path('example_output_gemini_2.json')
        output_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'report_data': result.dict(exclude_none=True)
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
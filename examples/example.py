# examples/example.py

import asyncio
import pandas as pd
import sys
import os
from pathlib import Path
import json
import yaml
import logging
from typing import Optional

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from extractor import RadiologyExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example radiology report
example_report = """
CT CHEST WITH CONTRAST

CLINICAL HISTORY: Lung cancer follow-up.
Study date: 2024-01-15

COMPARISON: CT chest from 2023-10-15

FINDINGS:
Right upper lobe mass measures 3.2 x 2.8 cm (previously 4.1 x 3.5 cm),
representing partial response to therapy.

Multiple bilateral pulmonary nodules, largest measuring 8mm in right
lower lobe, decreased in size from prior study (was 12mm).

No new pulmonary nodules.

Mediastinal and hilar lymph nodes are stable.

Small right pleural effusion, unchanged.

IMPRESSION:
1. Decrease in size of right upper lobe mass and pulmonary nodules,
consistent with partial response to therapy.
2. Stable small right pleural effusion.
"""

def print_separator(text: str = "", char: str = "-", length: int = 50):
    """Print a separator line with optional text"""
    if text:
        print(f"\n{text}")
    print(char * length)

def print_measurement(location: str, current_mm: float, prior_mm: Optional[float] = None, 
                     percent_change: Optional[float] = None, response: Optional[str] = None):
    """Print measurement details in a consistent format"""
    print(f"\nLocation: {location}")
    print(f"Current: {current_mm:.1f}mm")
    if prior_mm is not None:
        print(f"Prior: {prior_mm:.1f}mm")
    if percent_change is not None:
        print(f"Change: {percent_change:.1f}%")
    if response:
        print(f"Response: {response}")

async def run_example():
    """Run the example extraction"""
    logger.info("Starting radiology report extraction example")
    
    # Initialize reference data
    resources_path = Path(__file__).parent.parent / 'resources'
    try:
        # Load reference data with proper typing
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
        
        # Process report
        logger.info("Processing example report")
        result = await extractor.process_report(example_report)
        
        # Print results
        print_separator("Extracted Information", "=")
        
        # Basic Information
        print_separator("Study Information")
        print(f"Modality: {result.modality}")
        print(f"Body Region: {result.body_region}")
        if result.ICDO3_site:
            print(f"ICD-O Site: {result.ICDO3_site} ({result.ICDO3_site_term})")
        print(f"Study Date: {result.study_date}")
        print(f"Comparison Date: {result.comparison_date}")
        
        # Clinical Information
        if result.clinical_history:
            print_separator("Clinical Information")
            print(f"History: {result.clinical_history}")
        
        # Target Lesions
        if result.target_lesions:
            print_separator("Target Lesions")
            for lesion in result.target_lesions:
                print_measurement(
                    location=lesion.location,
                    current_mm=lesion.standardized_value_mm,
                    prior_mm=lesion.prior_value * 10 if lesion.prior_value and lesion.prior_unit == 'cm' 
                            else lesion.prior_value,
                    percent_change=lesion.percent_change,
                    response=lesion.response_category
                )
        
        # Non-target Lesions
        if result.non_target_lesions:
            print_separator("Non-target Lesions")
            for lesion in result.non_target_lesions:
                print_measurement(
                    location=lesion.location,
                    current_mm=lesion.standardized_value_mm
                )
        
        # Response Assessment
        if result.tumor_response or result.overall_assessment:
            print_separator("Response Assessment")
            if result.tumor_response:
                print(f"Tumor Response: {result.tumor_response}")
            if result.overall_assessment:
                print(f"Overall Assessment: {result.overall_assessment}")
        
        # Save complete output
        output_file = Path('example_output.json')
        with open(output_file, 'w') as f:
            json.dump(result.dict(), f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
        
    except FileNotFoundError as e:
        logger.error(f"Required resource files not found: {e}")
    except Exception as e:
        logger.error(f"Error in extraction process: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run_example())
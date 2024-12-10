# process_reports.py

import asyncio
import pandas as pd
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm

# Add the src directory to Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from extractor import RadiologyExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_reports(input_file: str, output_file: str, num_reports: int = 200):
    """
    Process reports from input CSV file and save results
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save results
        num_reports: Number of reports to process
    """
    try:
        # Read input CSV
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Initialize extractor
        resources_path = Path(__file__).parent / 'resources'
        modalities_df = pd.read_csv(resources_path / 'modalities.csv')
        topography_df = pd.read_csv(resources_path / 'ICDO3Topography.csv')
        extractor = RadiologyExtractor(modalities_df, topography_df)
        
        # Process reports until we have num_reports successful results
        results = []
        processed = 0
        idx = 0
        
        with tqdm(total=num_reports, desc="Processing reports") as pbar:
            while processed < num_reports and idx < len(df):
                row = df.iloc[idx]
                try:
                    report_info = {
                        'MRN': row['MRN'],
                        'EXAM_DATE': row['EXAM DATE/TIME'], 
                        'PROCEDURE': row['PROCEDURE'],
                        'REPORT': row['REPORT']
                    }
                    
                    # Process report text
                    result = await extractor.process_report(row['REPORT'])
                    
                    # Combine base info with extraction results
                    combined_result = {
                        **report_info,
                        **result.dict(exclude_none=True)  # Exclude None values for cleaner output
                    }
                    
                    results.append(combined_result)
                    processed += 1
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing report {idx + 1}: {str(e)}")
                    error_result = {
                        **report_info,
                        'error': str(e)
                    }
                    results.append(error_result)
                
                idx += 1
                
                # If we've processed all available reports but haven't reached num_reports
                if idx >= len(df) and processed < num_reports:
                    logger.warning(f"Only {processed} reports available in input file")
                    break
        
        # Save results
        logger.info(f"Saving results to {output_file}")
        results_df = pd.DataFrame(results)
        
        # Reorder columns to put primary fields first
        columns = ['MRN', 'EXAM_DATE', 'PROCEDURE', 'REPORT']
        other_columns = [col for col in results_df.columns if col not in columns]
        results_df = results_df[columns + other_columns]
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        
        # Print summary
        logger.info(f"Successfully processed {processed} reports")
        if 'error' in results_df.columns:
            error_count = results_df['error'].notna().sum()
            logger.info(f"Errors encountered: {error_count}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        raise

def check_resources():
    """Check if required resource files exist"""
    resources_path = Path(__file__).parent / 'resources'
    required_files = ['modalities.csv', 'ICDO3Topography.csv']
    
    missing_files = []
    for file in required_files:
        if not (resources_path / file).exists():
            missing_files.append(file)
    
    return missing_files

async def main():
    """Main function"""
    # Check for required files
    missing_files = check_resources()
    if missing_files:
        logger.error("Missing required resource files:")
        for file in missing_files:
            logger.error(f"- {file}")
        return
    
    input_file = "data/Results.csv"
    output_file = "radiology_results_200.csv"
    
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    await process_reports(input_file, output_file, num_reports=200)

if __name__ == "__main__":
    asyncio.run(main())
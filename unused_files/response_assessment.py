# response_assessment.py
"""
Module for handling response assessment criteria including RECIST 1.1
"""

from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

class ResponseCategory(Enum):
    """Enumeration of possible response categories"""
    COMPLETE_RESPONSE = "CR"
    PARTIAL_RESPONSE = "PR"
    STABLE_DISEASE = "SD"
    PROGRESSIVE_DISEASE = "PD"
    NOT_EVALUABLE = "NE"

@dataclass
class TargetLesion:
    """Class representing a target lesion measurement"""
    location: str
    current_size: float  # in mm
    prior_size: Optional[float] = None  # in mm
    measurement_axis: str = "long"  # long, short, or both
    
    @property
    def percent_change(self) -> Optional[float]:
        """Calculate percent change from prior measurement"""
        if self.prior_size is not None and self.prior_size > 0:
            return ((self.current_size - self.prior_size) / self.prior_size) * 100
        return None

@dataclass
class NonTargetLesion:
    """Class representing a non-target lesion assessment"""
    location: str
    status: str  # Complete Response, Present, Absent, Unequivocal Progression

@dataclass
class NewLesion:
    """Class representing a new lesion"""
    location: str
    description: str
    size: Optional[float] = None  # in mm

class RECISTEvaluator:
    """Class for evaluating response according to RECIST 1.1 criteria"""
    
    def __init__(self):
        self.response_thresholds = {
            'PR': -30,  # 30% decrease
            'PD': 20    # 20% increase
        }
    
    def evaluate_target_response(self, 
                               target_lesions: List[TargetLesion]) -> ResponseCategory:
        """Evaluate response based on target lesions"""
        if not target_lesions:
            return ResponseCategory.NOT_EVALUABLE
            
        # Calculate sum of diameters
        current_sum = sum(lesion.current_size for lesion in target_lesions)
        
        # If all lesions have prior measurements
        if all(lesion.prior_size is not None for lesion in target_lesions):
            prior_sum = sum(lesion.prior_size for lesion in target_lesions if lesion.prior_size)
            
            # Calculate percent change
            if prior_sum > 0:
                percent_change = ((current_sum - prior_sum) / prior_sum) * 100
                
                # Determine response category
                if current_sum == 0:
                    return ResponseCategory.COMPLETE_RESPONSE
                elif percent_change <= self.response_thresholds['PR']:
                    return ResponseCategory.PARTIAL_RESPONSE
                elif percent_change >= self.response_thresholds['PD']:
                    return ResponseCategory.PROGRESSIVE_DISEASE
                else:
                    return ResponseCategory.STABLE_DISEASE
                    
        return ResponseCategory.NOT_EVALUABLE
        
    def evaluate_non_target_response(self,
                                   non_target_lesions: List[NonTargetLesion]) -> ResponseCategory:
        """Evaluate response based on non-target lesions"""
        if not non_target_lesions:
            return ResponseCategory.NOT_EVALUABLE
            
        # Check for unequivocal progression
        if any(lesion.status == "Unequivocal Progression" for lesion in non_target_lesions):
            return ResponseCategory.PROGRESSIVE_DISEASE
            
        # Check for complete response
        if all(lesion.status == "Complete Response" or lesion.status == "Absent" 
               for lesion in non_target_lesions):
            return ResponseCategory.COMPLETE_RESPONSE
            
        # If all lesions are present but stable
        if all(lesion.status == "Present" for lesion in non_target_lesions):
            return ResponseCategory.NON_CR_NON_PD
            
        return ResponseCategory.NOT_EVALUABLE
        
    def evaluate_overall_response(self,
                                target_lesions: List[TargetLesion],
                                non_target_lesions: List[NonTargetLesion],
                                new_lesions: List[NewLesion]) -> ResponseCategory:
        """Determine overall response assessment"""
        # If there are new lesions, it's PD
        if new_lesions:
            return ResponseCategory.PROGRESSIVE_DISEASE
            
        target_response = self.evaluate_target_response(target_lesions)
        non_target_response = self.evaluate_non_target_response(non_target_lesions)
        
        # Logic for overall response
        if target_response == ResponseCategory.COMPLETE_RESPONSE:
            if non_target_response == ResponseCategory.COMPLETE_RESPONSE:
                return ResponseCategory.COMPLETE_RESPONSE
            elif non_target_response == ResponseCategory.NON_CR_NON_PD:
                return ResponseCategory.PARTIAL_RESPONSE
                
        elif target_response == ResponseCategory.PROGRESSIVE_DISEASE:
            return ResponseCategory.PROGRESSIVE_DISEASE
            
        elif target_response == ResponseCategory.STABLE_DISEASE:
            if non_target_response != ResponseCategory.PROGRESSIVE_DISEASE:
                return ResponseCategory.STABLE_DISEASE
                
        return ResponseCategory.NOT_EVALUABLE

def convert_measurements_to_recist(measurements: List[Dict]) -> List[TargetLesion]:
    """Convert raw measurements to RECIST target lesions"""
    target_lesions = []
    
    for measurement in measurements:
        # Convert measurements to mm if needed
        current_size = measurement['current_size']['value']
        if measurement['current_size']['unit'] == 'cm':
            current_size *= 10
            
        prior_size = None
        if 'prior_size' in measurement:
            prior_size = measurement['prior_size']['value']
            if measurement['prior_size']['unit'] == 'cm':
                prior_size *= 10
                
        target_lesions.append(TargetLesion(
            location=measurement['location'],
            current_size=current_size,
            prior_size=prior_size
        ))
        
    return target_lesions

def assess_response(preprocessed_data: Dict) -> Dict:
    """Assess response using preprocessed report data"""
    evaluator = RECISTEvaluator()
    
    # Convert measurements to target lesions
    target_lesions = convert_measurements_to_recist(
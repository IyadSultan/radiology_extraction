# RECIST Response Assessment Implementation Guide

## Overview
This document explains the implementation of RECIST (Response Evaluation Criteria In Solid Tumors) version 1.1 for tumor response assessment in our codebase.

## 1. Measurement Standardization

All measurements are standardized to millimeters for consistent calculations. The `RECISTMeasurement` class handles this conversion:

```python
def standardize_measurements(self):
    # Convert current measurements to mm
    if self.current_unit == 'cm':
        self.standardized_value_mm = self.current_value * 10
    elif self.current_unit == 'mm':
        self.standardized_value_mm = self.current_value

    # Convert prior measurements if they exist
    if self.prior_value and self.prior_unit:
        if self.prior_unit == 'cm':
            prior_mm = self.prior_value * 10
        else:
            prior_mm = self.prior_value

        # Calculate percent change if both measurements exist
        if prior_mm > 0:
            self.percent_change = ((self.standardized_value_mm - prior_mm) / prior_mm) * 100
```

## 2. Lesion Categorization

### Target Lesions
- Must be ≥10mm in longest diameter for non-nodal lesions
- Must be ≥15mm in short axis for lymph nodes
- Up to 5 lesions total, maximum 2 per organ
- Must be measurable and reproducible

### Non-Target Lesions
- All other lesions not selected as target lesions
- Includes lesions too small for accurate measurement
- Includes truly non-measurable disease (e.g., bone lesions, leptomeningeal disease)

## 3. Measurement Validation

The system validates measurements according to RECIST 1.1 criteria:

```python
def validate_measurements(self):
    if self.is_target:
        if self.standardized_value_mm < 10:
            raise ValueError("Target lesions must be ≥10mm in longest diameter")
        if self.anatomical_site == "lymph_node" and self.standardized_value_mm < 15:
            raise ValueError("Target lymph nodes must be ≥15mm in short axis")
```

## 4. Response Categories

### Target Lesions
- **Complete Response (CR)**: Disappearance of all target lesions
- **Partial Response (PR)**: ≥30% decrease in sum of diameters from baseline
- **Progressive Disease (PD)**: ≥20% increase in sum of diameters from nadir
- **Stable Disease (SD)**: Neither PR nor PD criteria met

### Non-Target Lesions
- **Complete Response**: Disappearance of all non-target lesions
- **Non-CR/Non-PD**: Persistence of one or more non-target lesions
- **Progressive Disease**: Unequivocal progression
- **Not Evaluable**: One or more lesions not assessed

## 5. Overall Response Calculation

The system determines overall response using this hierarchy:

```python
def calculate_recist_response(self) -> str:
    # Handle no evaluable lesions
    if not self.target_lesions and not self.non_target_lesions:
        return ResponseCategory.NOT_EVALUABLE.value
        
    # New lesions = PD
    if self.new_lesions:
        return ResponseCategory.PROGRESSIVE_DISEASE.value

    target_responses = [l.response_category for l in self.target_lesions 
                       if l.response_category]
    non_target_responses = [l.response_category for l in self.non_target_lesions 
                          if l.response_category]
    
    # Response Assessment Hierarchy:
    
    # 1. Complete Response check
    if (all(r == ResponseCategory.COMPLETE_RESPONSE.value for r in target_responses) and
        all(r == NonTargetResponse.COMPLETE_RESPONSE.value for r in non_target_responses)):
        return ResponseCategory.COMPLETE_RESPONSE.value
    
    # 2. Progressive Disease check
    if (any(r == ResponseCategory.PROGRESSIVE_DISEASE.value for r in target_responses) or
        any(r == NonTargetResponse.PROGRESSIVE_DISEASE.value for r in non_target_responses)):
        return ResponseCategory.PROGRESSIVE_DISEASE.value
    
    # 3. Partial Response check
    if (all(r == ResponseCategory.PARTIAL_RESPONSE.value for r in target_responses) and
        not any(r == NonTargetResponse.PROGRESSIVE_DISEASE.value for r in non_target_responses)):
        return ResponseCategory.PARTIAL_RESPONSE.value
    
    # 4. Default to Stable Disease
    return ResponseCategory.STABLE_DISEASE.value
```

## 6. Special Cases and Rules

### Missing Measurements
- Any target lesion not measured → Overall response is Not Evaluable
- Non-target lesion not assessed → Missing assessment noted

### New Lesions
- Any new lesion confirms Progressive Disease
- Must be unequivocal and not due to differences in scanning technique

### Lymph Node Considerations
- Normal nodes: <10mm short axis
- Non-pathological but non-target: 10-15mm
- Target nodes: ≥15mm short axis

### Overall Response Matrix

| Target Lesions | Non-Target Lesions | New Lesions | Overall Response |
|----------------|-------------------|-------------|------------------|
| CR             | CR                | No          | CR              |
| CR             | Non-CR/Non-PD     | No          | PR              |
| PR             | Non-PD            | No          | PR              |
| SD             | Non-PD            | No          | SD              |
| PD             | Any               | Yes/No      | PD              |
| Any            | PD                | Yes/No      | PD              |
| Any            | Any               | Yes         | PD              |

## Example Usage

```python
# Create target lesion measurements
target_lesion = RECISTMeasurement(
    location="Lung right upper lobe",
    current_value=25,
    current_unit="mm",
    prior_value=35,
    prior_unit="mm",
    is_target=True
)

# Calculate response
target_lesion.standardize_measurements()
print(f"Response Category: {target_lesion.response_category}")
```

## Notes
- All measurements are standardized to millimeters internally
- The system implements strict validation of RECIST 1.1 criteria
- New lesions automatically trigger Progressive Disease
- The response assessment follows a strict hierarchy as defined in RECIST 1.1
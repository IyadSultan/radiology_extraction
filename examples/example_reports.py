# example_reports.py

MAMMOGRAPHY_EXAMPLE = """
BILATERAL DIAGNOSTIC MAMMOGRAM

CLINICAL HISTORY: 55-year-old female with right breast mass on self-exam.

TECHNIQUE: Standard CC and MLO views of both breasts with spot compression views of the right breast mass.

BREAST COMPOSITION: Heterogeneously dense breast tissue (category C).

FINDINGS:
Right Breast:
- 2.3 cm spiculated mass at 10 o'clock, 8 cm from the nipple
- Associated pleomorphic microcalcifications
- No skin thickening or nipple retraction

Left Breast:
- No suspicious masses, calcifications, or architectural distortion
- Stable scattered fibroglandular densities

IMPRESSION:
Right breast mass with suspicious morphology and associated calcifications.
BIRADS Category: 4C - High suspicion for malignancy
"""

CHEST_CT_FUNGAL_EXAMPLE = """
CT CHEST WITH CONTRAST

CLINICAL HISTORY: Neutropenic fever in patient with AML, concern for fungal infection.

COMPARISON: CT chest from 2 weeks ago.

FINDINGS:
1. Multiple bilateral pulmonary nodules with surrounding ground-glass halos:
   - Right upper lobe: 2.1 cm nodule (previously 1.5 cm)
   - Left lower lobe: 1.8 cm nodule (previously 1.2 cm)
   Both demonstrating characteristic "halo sign"

2. New cavitary lesion in right lower lobe measuring 2.5 cm with air-crescent sign.

3. Scattered ground-glass opacities throughout both lung fields.

4. No pleural effusion.

IMPRESSION:
1. Progressive pulmonary findings highly suspicious for invasive fungal infection,
   demonstrating characteristic halo signs and air-crescent sign.
2. Recommend correlation with galactomannan and beta-D-glucan testing.
"""

BRAIN_TUMOR_EXAMPLE = """
BRAIN MRI WITH AND WITHOUT CONTRAST

CLINICAL HISTORY: Follow-up of known left temporal glioblastoma.

COMPARISON: MRI from 6 weeks ago.

TECHNIQUE: Multiplanar multisequence MRI with and without gadolinium.

FINDINGS:
Left temporal lobe mass:
- Measures 4.2 x 3.8 x 3.5 cm (previously 3.8 x 3.2 x 3.0 cm)
- Heterogeneous enhancement
- Increased surrounding FLAIR signal consistent with vasogenic edema
- Mass effect with 6mm rightward midline shift
- New areas of restricted diffusion along medial margin

Adjacent structures:
- Partial effacement of left temporal horn
- Uncal herniation measuring 3mm
- No hydrocephalus

IMPRESSION:
1. Progressive left temporal glioblastoma with:
   - Interval size increase
   - Increased mass effect and midline shift
   - New areas of restricted diffusion suggesting hypercellular tumor
2. Increased vasogenic edema with early uncal herniation
"""
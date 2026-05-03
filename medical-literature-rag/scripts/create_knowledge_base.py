# Creates sample medical documents representing clinical guidelines 
  
from pathlib import Path 
  
DOCUMENTS = { 
    'malaria_guidelines.txt': """ 
GHANA HEALTH SERVICE — MALARIA CASE MANAGEMENT GUIDELINES (2024) 
  
1. DISEASE OVERVIEW 
Malaria is caused by Plasmodium parasites, primarily P. falciparum (most dangerous) 
in Ghana. 
Transmission is through bites of infected female Anopheles mosquitoes, primarily at 
night. 
Ghana has endemic malaria year-round, with peak transmission during rainy seasons 
(April-June, September-November). 
  
2. CLINICAL PRESENTATION 
Uncomplicated malaria: cyclical fever (38-40°C), chills, rigors, sweating, headache, 
myalgia, nausea. 
Severe malaria (P. falciparum): impaired consciousness, seizures, respiratory 
distress, severe anaemia (Hb < 7g/dL), 
hypoglycaemia, abnormal bleeding, jaundice, haemoglobinuria (blackwater urine). 
  
3. DIAGNOSIS 
Microscopy: thick and thin blood smear — gold standard. Identifies species and 
quantifies parasitaemia. 
Rapid Diagnostic Test (RDT): detects HRP-2 antigen. Sensitivity > 95% for P. 
falciparum. 
PCR: highest sensitivity. Used for low-density parasitaemia and species confirmation 
in research. 
Do NOT treat clinically without a positive test result in non-endemic settings. 
  
4. TREATMENT — UNCOMPLICATED MALARIA 
First-line: Artemether-Lumefantrine (AL) — Coartem® 
Adult dosing (> 35kg): 4 tablets (80mg/480mg) at 0, 8, 24, 36, 48, 60 hours 
CRITICAL: Take with fatty food or whole milk to enhance absorption by 16-fold. 
Contraindications: first trimester pregnancy (use Quinine + Clindamycin instead). 
  
Second-line: Artesunate-Amodiaquine (ASAQ) 
Dosing: once daily for 3 days. Weight-based dosing for children. 
  
5. TREATMENT — SEVERE MALARIA 
IV Artesunate: 2.4 mg/kg IV at 0, 12, 24 hours, then once daily until oral therapy 
tolerated. 
If IV Artesunate unavailable: Quinine IV as alternative (higher toxicity). 
Supportive care: glucose monitoring (risk of hypoglycaemia), IV fluids, 
antipyretics. 
Admit to hospital — do not manage severe malaria in outpatient settings. 
  
6. MALARIA IN PREGNANCY 
Malaria in pregnancy increases risk of anaemia, preterm birth, and low birth weight. 
First trimester: Quinine + Clindamycin (7 days). AL is avoided in T1. 
Second and third trimester: AL is safe and recommended. 
Intermittent Preventive Treatment (IPTp): Sulfadoxine-Pyrimethamine at every 
antenatal visit from 13 weeks. 
  
7. PREVENTION 
Long-lasting insecticidal nets (LLINs): universal coverage target. Use every night. 
Indoor Residual Spraying (IRS): Approved insecticides sprayed on indoor walls. 
Chemoprevention: IPTp for pregnant women, SMC for children 3-59 months in seasonal 
areas. 
"""   , 
  
    'hypertension_guidelines.txt': """ 
HYPERTENSION MANAGEMENT PROTOCOL — PRIMARY AND SECONDARY CARE (2024) 
  
1. DEFINITION AND CLASSIFICATION 
Normal: < 120/80 mmHg 
Elevated: 120-129/<80 mmHg (lifestyle changes only) 
Stage 1 Hypertension: 130-139/80-89 mmHg 
Stage 2 Hypertension: >= 140/90 mmHg 
Hypertensive Crisis: > 180/120 mmHg (emergency) 
  
Diagnosis requires TWO separate readings on TWO different occasions. 
  
2. RISK STRATIFICATION 
Low risk: Stage 1 with no CVD risk factors — lifestyle modification 3 months before 
starting medication. 
Medium risk: Stage 1 with 1-2 risk factors — consider medication alongside 
lifestyle. 
High risk: Stage 2 OR any history of CVD, diabetes, CKD — start medication 
immediately. 
  
3. NON-PHARMACOLOGICAL TREATMENT 
DASH diet: reduce sodium to < 2.3g/day, increase fruits/vegetables. 
Weight loss: every 1kg reduction lowers BP by 1 mmHg. 
Physical activity: 150 min/week moderate intensity. 
Limit alcohol: < 2 standard drinks/day for men, < 1 for women. 
Smoking cessation: not directly lowers BP but reduces total CVD risk. 
  
4. PHARMACOLOGICAL TREATMENT 
Step 1 — first-line options (choose based on patient factors): 
  A. ACE Inhibitor: Lisinopril 10mg daily → titrate to 40mg. Avoid in bilateral 
renal artery stenosis. 
  B. ARB: Losartan 50mg daily → titrate to 100mg. Use if ACE inhibitor cough. 
  C. Calcium Channel Blocker: Amlodipine 5mg daily → titrate to 10mg. Good for 
isolated systolic HTN. 
  D. Thiazide Diuretic: Hydrochlorothiazide 12.5mg daily → titrate to 25mg. Caution 
with gout. 
  
Step 2 — if BP uncontrolled on monotherapy, add a second agent from a different 
class. 
Step 3 — triple therapy: ACEi/ARB + CCB + Thiazide. 
  
5. TREATMENT TARGETS 
General adults: BP < 130/80 mmHg 
Elderly > 65 years: < 140/90 mmHg (more cautious to avoid falls) 
CKD with proteinuria: < 130/80 mmHg 
Diabetes: < 130/80 mmHg 
  
6. HYPERTENSIVE EMERGENCY 
BP > 180/120 with end-organ damage (stroke, MI, acute heart failure, acute kidney 
injury). 
IV Labetalol or IV Nicardipine — reduce BP by no more than 25% in first hour. 
Admit to ICU/HDU. Do NOT reduce BP too rapidly — risk of ischaemia. 
"""   , 
  
    'diabetes_management.txt': """ 
TYPE 2 DIABETES MANAGEMENT PROTOCOL 
  
1. DIAGNOSIS 
Fasting plasma glucose (FPG) >= 7.0 mmol/L (126 mg/dL) 
2-hour OGTT >= 11.1 mmol/L (200 mg/dL) 
HbA1c >= 48 mmol/mol (6.5%) 
Random glucose >= 11.1 mmol/L with symptoms 
Confirm with repeat test unless symptoms are present. 
  
2. INITIAL MANAGEMENT 
All patients: lifestyle modification (diet, exercise, weight loss) from diagnosis. 
Metformin is first-line pharmacotherapy unless contraindicated. 
Metformin dosing: 500mg twice daily with meals. Increase by 500mg weekly to 
2000mg/day. 
Contraindications: eGFR < 30 mL/min/1.73m2, active liver disease, alcohol excess. 
  
3. INTENSIFICATION — ADD SECOND AGENT IF HbA1c NOT AT TARGET AFTER 3 MONTHS
SGLT2 inhibitor (Empagliflozin, Dapagliflozin): add if CVD, heart failure, or CKD. 
Also promotes weight loss. 
GLP-1 agonist (Semaglutide, Dulaglutide): best for weight reduction and CVD risk. 
Weekly injection. 
DPP-4 inhibitor (Sitagliptin): weight-neutral, safe in elderly. 
Sulfonylurea (Glibenclamide): inexpensive but causes hypoglycaemia and weight gain. 
  
4. INSULIN THERAPY 
Start when HbA1c > 10% at diagnosis, or when oral agents fail. 
Starting insulin: Basal insulin (NPH or Glargine) 10 units at bedtime, titrate by 2 
units every 3 days. 
Self-monitoring of blood glucose (SMBG): 4-7x per day when starting insulin. 
  
5. MONITORING SCHEDULE 
HbA1c: every 3 months until stable target reached, then every 6 months. 
Fasting glucose: at each visit. 
Foot examination: annually (check sensation, pulses, skin integrity). 
Eye examination (fundoscopy or retinal photography): annually. 
Kidney function (eGFR + urine ACR): annually. 
BP and lipids: at each visit. 
  
6. TARGET GLYCAEMIC CONTROL 
HbA1c target: < 53 mmol/mol (7%) for most patients. 
Relaxed target HbA1c < 64 mmol/mol (8%): elderly, frail, multiple comorbidities, 
short life expectancy. 
Tight target HbA1c < 48 mmol/mol (6.5%): younger patients, short disease duration, 
no CVD. 
"""   , 
} 
  
if __name__ == '__main__': 
    Path('data/medical').mkdir(parents=True, exist_ok=True) 
    for filename, content in DOCUMENTS.items(): 
        path = Path(f'data/medical/{filename}') 
        path.write_text(content.strip()) 
        print(f'Created: {path} ({len(content)} chars)') 
    print(f'\nKnowledge base ready: {len(DOCUMENTS)} documents in data/medical/')
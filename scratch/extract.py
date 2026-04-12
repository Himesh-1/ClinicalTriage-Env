import sys
import os
import json
sys.path.insert(0, r"d:\ClinicalTriage-Env")

from tasks import TASKS

os.makedirs(r"d:\ClinicalTriage-Env\tasks", exist_ok=True)

easy = TASKS["single_patient_easy"]
easy["interaction_mode"] = "single_step"
with open(r"d:\ClinicalTriage-Env\tasks\single_patient_easy.json", "w") as f:
    json.dump(easy, f, indent=2)

med = TASKS["concurrent_patients_medium"]
med["interaction_mode"] = "multi_patient"
if "patient_profiles" in med: del med["patient_profiles"]
if "ground_truth_ranking" in med: del med["ground_truth_ranking"]

with open(r"d:\ClinicalTriage-Env\tasks\concurrent_patients_medium.json", "w") as f:
    json.dump(med, f, indent=2)

hard = TASKS["incomplete_vitals_hard"]
hard["interaction_mode"] = "incomplete_vitals"
with open(r"d:\ClinicalTriage-Env\tasks\incomplete_vitals_hard.json", "w") as f:
    json.dump(hard, f, indent=2)

print("Extracted successfully!")

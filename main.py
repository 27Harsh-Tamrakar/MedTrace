from preprocess import clean_foil
from ocr import run_ocr
from parser import parse_fields
from tamper import detect_tamper
from cavity_analysis import analyze_cavities
import cv2
import json
import sys

# Image path from command line
image_path = sys.argv[1]

# ---------------- Preprocess Foil ----------------
clean_gray = clean_foil(image_path)

# Convert grayscale back to 3-channel image for CV steps
clean_img = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)

# ---------------- OCR ----------------
texts = run_ocr(clean_img)

with open("ocr_text.txt", "w") as f:
    for t in texts:
        f.write(t + "\n")

print("\n--- OCR TEXT DUMP ---")
for t in texts:
    print(t)
print("----------------------\n")

# ---------------- Parsing ----------------
fields = parse_fields(texts)

# ---------------- Texture Tamper Detection ----------------
tamper_score = detect_tamper(clean_img)

# ---------------- Blister Cavity Geometry ----------------
cavity_score = analyze_cavities(clean_img)

# ---------------- Combined Forensic Scoring ----------------
# Weight texture more, geometry as secondary evidence
final_score = (tamper_score * 0.7) + (cavity_score * 0.3)
final_score = min(100, final_score)

# ---------------- Verdict ----------------
verdict = "SUSPICIOUS" if final_score > 40 else "NORMAL"

result = {
    **fields,
    "tamper_score": round(tamper_score, 2),
    "cavity_deformation_score": round(cavity_score, 2),
    "final_score": round(final_score, 2),
    "confidence": f"{final_score:.2f}%",
    "evidence": "Surface texture disturbance and cavity geometry deformation analysis",
    "verdict": verdict,
}

print(json.dumps(result, indent=4))

# Save forensic report
with open("verdict.json", "w") as f:
    json.dump(result, f, indent=4)
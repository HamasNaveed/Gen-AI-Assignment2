# Generative AI – Assignment 2

| Part | File |
|------|------|
| Q1 – Denoising Autoencoder (BSDS300) | `Q1_Denoising_Autoencoder.ipynb` |
| Q2 – CycleGAN (Person Face Sketches) | `Q2_CycleGAN.ipynb` |
| Q2 – Flask UI | `app/` |
| Report outline | `report/REPORT_TEMPLATE.md` |
| Prompts used | `prompts.txt` |

## Running
**Q1:** open in Google Colab (GPU), *Run all*. BSDS300 is downloaded automatically. Results are written to `q1_*.csv/png`.

**Q2:** open in Colab (GPU), run all and upload your `kaggle.json` when asked. A checkpoint is saved to Google Drive (`MyDrive/cyclegan_faces/last.pth`) after every epoch; re-running the notebook resumes from it automatically. Generator weights `G_face2sketch.pth` and `G_sketch2face.pth` are written alongside.

**UI:**
```bash
pip install -r requirements.txt
# copy the two .pth files into app/weights/
cd app && python app.py        # open http://localhost:5000
```
Upload an image or use the camera; the app auto-detects sketch vs. face (low colour saturation + bright paper background => sketch; a manual override is provided) and applies the matching generator.

## Submission
Zip the notebook(s), the final Word/PDF report, and `prompts.txt` as `ROLLNO NAME.ZIP`.

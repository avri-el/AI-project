from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging
from google import genai
import base64


# ===========================
# GEMINI / GENERATIVE AI CLIENT
# ===========================
# Read API key from environment for safety. Set `GEMINI_API_KEY` or `GOOGLE_API_KEY`.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        # If client init fails, keep client as None and log the issue
        logging.warning("Failed to initialize Gemini client: %s", e)


# ===========================
# FLASK INIT
# ===========================
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


# ===========================
# LOAD CNN MODEL
# ===========================
MODEL_PATH = "model/colon_diseases.h5"
cnn_model = load_model(MODEL_PATH)


# ===========================
# CLASS LABELS
# ===========================
class_labels = [
    "Normal Colon",
    "Colon Ulcerative Colitis",
    "Colon Polyps",
    "Colon Esophagitis"
]

try:
    model_output_size = int(cnn_model.output_shape[-1])
except Exception:
    model_output_size = None

if model_output_size is not None and model_output_size != len(class_labels):

    logging.warning(
        f"Model output size ({model_output_size}) does not match class_labels ({len(class_labels)})."
    )

    if model_output_size > len(class_labels):
        class_labels += [f"Class {i}" for i in range(len(class_labels), model_output_size)]
    else:
        class_labels = class_labels[:model_output_size]

# Ensure unique labels
seen = {}
for i, lab in enumerate(class_labels):
    if lab in seen:
        new_lab = f"{lab} ({seen[lab]+1})"
        seen[lab] += 1
        class_labels[i] = new_lab
    else:
        seen[lab] = 1


# ===========================
# IMAGE PREDICTION FUNCTION
# ===========================
def predict_image(img_path):

    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = cnn_model.predict(x)
    probs = [float(p) for p in preds[0].tolist()]

    if model_output_size is not None and len(probs) != model_output_size:
        if len(probs) < model_output_size:
            probs += [0.0] * (model_output_size - len(probs))
        else:
            probs = probs[:model_output_size]

    predicted_idx = int(np.argmax(probs))
    predicted_class = class_labels[predicted_idx]
    confidence = float(probs[predicted_idx])

    return predicted_idx, predicted_class, confidence, probs


# ===========================
# ROUTES
# ===========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file dikirim"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"})

    ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
    filename = secure_filename(file.filename)

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Tipe file tidak didukung. Gunakan PNG/JPG/JPEG/BMP."})

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    logging.info(f"Saved uploaded file to {file_path}")

    try:
        predicted_idx, predicted_class, confidence, probs = predict_image(file_path)
        os.remove(file_path)

        probs_percent = [round(float(p) * 100, 2) for p in probs]

        return jsonify({
            "predicted_index": predicted_idx,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "confidence_score": confidence,
            "probs": probs_percent,
            "labels": class_labels
        })

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)})


# ===========================
# GEMINI CHAT ASSISTANT
# ===========================
# ===========================
# ROUTE: GEMINI CHAT ASSISTANT
# ===========================
# ===========================
# ROUTE: GEMINI CHAT ASSISTANT
# ===========================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        def safe_get(pred, key, default="-"):
            if pred and isinstance(pred, dict):
                return pred.get(key, default)
            return default

        user_message = ""
        prediction = None

        # IMAGE UPLOAD HANDLING
        if request.files and "file" in request.files:
            file = request.files["file"]
            if file and file.filename:
                filename = secure_filename(file.filename)
                upload_folder = "uploads"
                os.makedirs(upload_folder, exist_ok=True)
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)

                try:
                    idx, cls, conf, probs = predict_image(file_path)
                    probs_percent = [round(float(p) * 100, 2) for p in probs]
                    prediction = {
                        "index": idx,
                        "class": cls,
                        "confidence": round(conf * 100, 2),
                        "probs": probs_percent,
                        "labels": class_labels,
                    }
                except Exception as e:
                    prediction = {"error": str(e)}
                finally:
                    try:
                        os.remove(file_path)
                    except:
                        pass

            user_message = request.form.get("message", "")

        else:
            # JSON mode
            data = request.get_json(silent=True) or {}
            user_message = data.get("message", "")
            prediction = data.get("prediction")

        # PREPARE PREDICTION BLOCK
        if prediction:
            pred_class = safe_get(prediction, "class")
            pred_conf = safe_get(prediction, "confidence")
            pred_probs = safe_get(prediction, "probs", [])
            pred_labels = safe_get(prediction, "labels", [])

            if pred_labels and pred_probs and len(pred_labels) == len(pred_probs):
                prob_block = "\n".join([f"- {l}: {p}%" for l, p in zip(pred_labels, pred_probs)])
            else:
                prob_block = "No detailed probability breakdown available."

        # ===============================
        # CUSTOM PROMPT (ADA DIAGNOSIS + SARAN)
        # ===============================
        if user_message and prediction:
            prompt = f"""
You are an advanced medical AI assistant specialized in colonoscopy interpretation.

USER QUESTION:
{user_message}

IMAGE ANALYSIS:
- Predicted Condition: {pred_class}
- Confidence: {pred_conf}%
- Probability Breakdown:
{prob_block}

Your required output:
1. **Explain the meaning of the prediction in simple, clear language.**
2. **Give a possible early-stage medical interpretation (NOT a final diagnosis).**
3. **Provide recommendations:** 
   - What the user should do next  
   - Whether they should see a doctor  
   - Possible lifestyle or early-care suggestions
4. **If the user asks something unrelated, answer normally but still mention the image result.**
5. **ALWAYS include this disclaimer at the end:**  
   “Analisis ini bukan diagnosis final. Konsultasikan hasil ini kepada dokter atau spesialis gastroenterologi untuk penilaian yang akurat.”

Be clear, helpful, and medically safe.
"""

        elif prediction:
            prompt = f"""
Interpret the colonoscopy AI prediction below:

- Predicted Condition: {pred_class}
- Confidence: {pred_conf}%
- Probability Breakdown:
{prob_block}

Your required output:
1. Meaning of the finding.
2. Possible medical interpretation (early, non-final).
3. Recommended next steps for the patient.
4. Follow-up suggestions.
5. Add the medical disclaimer at the end.
"""

        elif user_message:
            prompt = user_message

        else:
            return jsonify({"error": "No message or prediction provided."})

        # INIT GEMINI
        global client
        if not client:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key:
                try:
                    client = genai.Client(api_key=key)
                except Exception as e:
                    return jsonify({"error": str(e)})

        if not client:
            return jsonify({"reply": "Gemini API key not set.", "prediction": prediction})

        system_instruction = """
You are a safe, professional medical assistant for colonoscopy interpretation.
Always avoid giving definitive diagnoses.
Always include a final disclaimer.
"""

        # CALL GEMINI
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [
                    {"text": f"System: {system_instruction}\n\nUser: {prompt}"}
                ]}]
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[{"role": "user", "parts": [
                    {"text": f"System: {system_instruction}\n\nUser: {prompt}"}
                ]}]
            )

        reply = getattr(response, "text", str(response))

        return jsonify({"reply": reply, "prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})


# ===========================
# RUN SERVER
# ===========================
if __name__ == "__main__":
    app.run(debug=True)

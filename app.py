# app.py
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging
import traceback
import io
import base64
from datetime import datetime
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
from flask import send_file

# Try to support both import styles for different SDK versions/environments
try:
    from google import genai as genai_sdk  # preferred in some installs
except Exception:
    try:
        import google.generativeai as genai_sdk  # older alias
    except Exception:
        genai_sdk = None

# dotenv to load .env in development (optional, best practice)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv not installed — it's optional if you set env vars directly
    pass

# ===========================
# CONFIG / CLIENT INIT
# ===========================
logging.basicConfig(level=logging.INFO)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = None

def init_gemini_client(api_key: str):
    global client, genai_sdk
    if not genai_sdk:
        raise RuntimeError("GenAI SDK not available. Install google-generativeai.")
    try:
        # different SDK surfaces: some use genai_sdk.Client(...)
        # others expect genai_sdk.configure(api_key=...)
        if hasattr(genai_sdk, "Client"):
            c = genai_sdk.Client(api_key=api_key)
        else:
            # older style
            genai_sdk.configure(api_key=api_key)
            c = genai_sdk
        return c
    except Exception as e:
        logging.exception("Failed to initialize Gemini client")
        raise

if GEMINI_API_KEY:
    try:
        client = init_gemini_client(GEMINI_API_KEY)
        logging.info("Gemini client initialized from environment.")
    except Exception as e:
        logging.warning("Gemini client init failed: %s", str(e))

# ===========================
# FLASK + MODEL
# ===========================
app = Flask(__name__)

MODEL_PATH = "model/colon_diseases.h5"
if not os.path.exists(MODEL_PATH):
    logging.error("Model file not found: %s. Please place your model at this path.", MODEL_PATH)
    raise FileNotFoundError(MODEL_PATH)

cnn_model = load_model(MODEL_PATH)
logging.info("CNN model loaded from %s", MODEL_PATH)

# default labels (will be resized to model output if needed)
class_labels = [
    "Normal Colon",
    "Colon Ulcerative Colitis",
    "Colon Polyps",
    "Colon Esophagitis",
]

try:
    model_output_size = int(cnn_model.output_shape[-1])
except Exception:
    model_output_size = None

if model_output_size is not None and model_output_size != len(class_labels):
    logging.warning("Model output size (%s) != class_labels (%s). Adjusting labels.", model_output_size, len(class_labels))
    if model_output_size > len(class_labels):
        class_labels += [f"Class {i}" for i in range(len(class_labels), model_output_size)]
    else:
        class_labels = class_labels[:model_output_size]

# ensure unique
seen = {}
for i, lab in enumerate(class_labels):
    if lab in seen:
        class_labels[i] = f"{lab} ({seen[lab]+1})"
        seen[lab] += 1
    else:
        seen[lab] = 1

# ===========================
# HELPERS
# ===========================
def predict_image(img_path: str):
    """Load image, run model, return idx, label, confidence (0..1), probs list (0..1)."""
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = cnn_model.predict(x)
    probs = list(map(float, preds[0].tolist()))
    if model_output_size is not None and len(probs) != model_output_size:
        if len(probs) < model_output_size:
            probs += [0.0] * (model_output_size - len(probs))
        else:
            probs = probs[:model_output_size]
    idx = int(np.argmax(probs))
    label = class_labels[idx] if idx < len(class_labels) else f"Class {idx}"
    confidence = float(probs[idx])
    return idx, label, confidence, probs

def safe_extract_prediction(pred_dict):
    """Given a prediction object (from frontend or internal), normalize keys."""
    if not pred_dict or not isinstance(pred_dict, dict):
        return None
    # try several common key names
    label = pred_dict.get("predicted_class") or pred_dict.get("prediction") or pred_dict.get("class") or pred_dict.get("predicted") or pred_dict.get("label")
    confidence = pred_dict.get("confidence") or pred_dict.get("confidence_score") or pred_dict.get("conf") or pred_dict.get("probability")
    probs = pred_dict.get("probs") or pred_dict.get("probabilities") or pred_dict.get("scores") or pred_dict.get("prob")
    labels = pred_dict.get("labels") or pred_dict.get("class_labels") or class_labels
    # ensure numeric types where possible
    try:
        if isinstance(confidence, str) and confidence.endswith("%"):
            confidence = float(confidence.strip().replace("%",""))
        elif isinstance(confidence, (int, float)) and confidence > 1:
            # assume percent already
            confidence = float(confidence)
        elif isinstance(confidence, (int, float)) and confidence <= 1:
            confidence = float(confidence) * 100.0
    except Exception:
        pass
    # if probs look like 0..100 convert to 0..100 floats
    if isinstance(probs, list):
        cleaned = []
        for p in probs:
            try:
                pv = float(p)
                if 0 <= pv <= 1:
                    pv = pv * 100.0
                cleaned.append(round(pv, 2))
            except Exception:
                cleaned.append(p)
        probs = cleaned
    return {
        "class": label,
        "confidence": confidence,
        "probs": probs,
        "labels": labels
    }

def extract_text_from_genai_response(resp):
    """Try to extract meaningful text from various SDK response shapes."""
    try:
        # object with .text
        if hasattr(resp, "text"):
            return resp.text
        # vertex-like dict
        if isinstance(resp, dict):
            # google genai sometimes returns {'candidates':[{'content': '...'}]}
            cands = resp.get("candidates") or resp.get("outputs") or resp.get("replies")
            if isinstance(cands, list) and len(cands) > 0:
                first = cands[0]
                if isinstance(first, dict):
                    # common fields
                    for k in ("content", "message", "output", "text"):
                        if k in first:
                            return first[k]
                    # try nested
                    return str(first)
            # fallback to any text field
            for k in ("response", "result", "output"):
                if k in resp:
                    return str(resp[k])
            return str(resp)
        # fallback
        return str(resp)
    except Exception:
        return str(resp)

# ===========================
# ROUTES
# ===========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file dikirim"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400
    ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Tipe file tidak didukung. Gunakan PNG/JPG/JPEG/BMP."}), 400
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    try:
        idx, lab, conf, probs = predict_image(file_path)
        try:
            os.remove(file_path)
        except Exception:
            pass
        probs_percent = [round(float(p) * 100, 2) if p <= 1 else round(float(p), 2) for p in probs]
        # ensure probs are 0..100 floats
        probs_percent = [p if p <= 100 else p for p in probs_percent]
        return jsonify({
            "predicted_index": idx,
            "predicted_class": lab,
            "prediction": lab,
            "confidence": round(conf * 100, 2),
            "confidence_score": conf,
            "probs": probs_percent,
            "labels": class_labels
        })
    except Exception as e:
        logging.exception("Prediction failed")
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts either:
      - JSON body: {"message": "...", "prediction": {...}}  (prediction produced by /predict or client)
      - multipart/form-data with "file" (image) and optional "message": server will run local prediction then call Gemini
    Returns: {"reply": "...", "prediction": {...}} or {"error": "..."}
    """
    try:
        # 1) extract inputs
        message = ""
        prediction = None

        if request.files and "file" in request.files:
            # handle image-upload-question scenario
            file = request.files["file"]
            if file and file.filename:
                filename = secure_filename(file.filename)
                upload_folder = "uploads"
                os.makedirs(upload_folder, exist_ok=True)
                path = os.path.join(upload_folder, filename)
                file.save(path)
                try:
                    idx, lab, conf, probs = predict_image(path)
                    probs_percent = [round(float(p) * 100, 2) if p <= 1 else round(float(p), 2) for p in probs]
                    prediction = {
                        "index": idx,
                        "class": lab,
                        "confidence": round(conf * 100, 2),
                        "probs": probs_percent,
                        "labels": class_labels
                    }
                except Exception as e:
                    logging.exception("Local prediction failed in /ask")
                    prediction = {"error": str(e)}
                finally:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            # optional message form
            message = request.form.get("message", "")
        else:
            # JSON mode
            data = request.get_json(silent=True) or {}
            message = data.get("message", "") or ""
            raw_pred = data.get("prediction")
            if raw_pred:
                prediction = safe_extract_prediction(raw_pred)

        # Build prompt depending on presence of prediction + message
        use_pred = prediction if isinstance(prediction, dict) and not prediction.get("error") else None
        if use_pred:
            pred_class = use_pred.get("class", "-")
            pred_conf = use_pred.get("confidence", "-")
            pred_probs = use_pred.get("probs", [])
            pred_labels = use_pred.get("labels", class_labels)
            if isinstance(pred_labels, list) and isinstance(pred_probs, list) and len(pred_labels) == len(pred_probs):
                prob_block = "\n".join([f"- {l}: {p}%" for l, p in zip(pred_labels, pred_probs)])
            else:
                prob_block = "No detailed probability breakdown available."
        else:
            pred_class = pred_conf = "-"
            prob_block = "No prediction data."

        # If nothing to ask/send, return helpful message
        if not message and not use_pred:
            return jsonify({"error": "No message or prediction provided."}), 400

        # Build prompt (safe, includes medical instructions + request for recommendations)
        if message and use_pred:
            user_prompt = f"""USER QUESTION:
{message}

IMAGE ANALYSIS:
- Predicted Condition: {pred_class}
- Confidence: {pred_conf}%
- Probability Breakdown:
{prob_block}

Please:
1) Explain the analysis in simple language.
2) Provide a possible clinical interpretation (NOT a definitive diagnosis).
3) Recommend next steps (investigations, biopsy, specialist referral) and practical advice.
4) Add a clear medical disclaimer at the end stating this is not a final diagnosis.
"""
        elif use_pred:
            user_prompt = f"""Interpret these image analysis results:

- Predicted Condition: {pred_class}
- Confidence: {pred_conf}%
- Probability Breakdown:
{prob_block}

Explain: meaning, possible causes, clinical relevance, recommended next steps, and end with a medical disclaimer.
"""
        else:
            user_prompt = message  # free question

        # Initialize client if not yet initialized
        global client
        if not client:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key:
                client = init_gemini_client(key)

        if not client:
            return jsonify({
                "reply": "Gemini client not configured on server. Please set GEMINI_API_KEY in environment or .env.",
                "prediction": prediction
            }), 503

        system_instruction = (
            "You are a careful, professional medical assistant specialized in colonoscopy image interpretation. "
            "Always avoid giving definitive diagnoses and always include a disclaimer that the AI output is not a final diagnosis."
        )

        # Try a modern available model; adapt if your account has different models
        preferred_models = ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]
        response_obj = None
        last_exception = None
        for m in preferred_models:
            try:
                response_obj = client.models.generate_content(
                    model=m,
                    contents=[{"role": "user", "parts": [{"text": f"System: {system_instruction}\n\nUser: {user_prompt}"}]}],
                )
                break
            except Exception as e:
                logging.warning("Model %s failed: %s", m, str(e))
                last_exception = e
                continue

        if response_obj is None:
            logging.exception("All Gemini attempts failed: %s", str(last_exception))
            return jsonify({"error": "Gemini API call failed", "details": str(last_exception)}), 502

        reply_text = extract_text_from_genai_response(response_obj)
        return jsonify({"reply": reply_text, "prediction": prediction})

    except Exception as e:
        logging.exception("Unhandled error in /ask")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate PDF report using ReportLab. Accepts JSON with keys: prediction, aiSummary, geminiReply, imageData (data URL), metadata."""
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    prediction = data.get('prediction') or {}
    ai_summary = data.get('aiSummary') or ''
    gemini_reply = data.get('geminiReply') or ''
    metadata = data.get('metadata') or {}
    image_data = data.get('imageData')

    buf = io.BytesIO()
    try:
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4

        # Header
        c.setFont('Helvetica-Bold', 16)
        c.drawString(30, height - 40, 'ColonAI — Analysis Report')
        c.setFont('Helvetica', 9)
        c.drawString(30, height - 56, f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}')

        y = height - 80

        # Metadata table
        if metadata:
            c.setFont('Helvetica-Bold', 12)
            c.drawString(30, y, 'Case Metadata')
            y -= 18
            c.setFont('Helvetica', 9)
            for k, v in metadata.items():
                c.drawString(36, y, f'{k}: {v}')
                y -= 14
            y -= 8

        # Image
        if image_data and image_data.startswith('data:'):
            try:
                header, b64 = image_data.split(',', 1)
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                # resize if too large for page
                max_w = width - 60
                max_h = (height / 2)
                iw, ih = img.size
                scale = min(max_w / iw, max_h / ih, 1.0)
                iw2 = int(iw * scale)
                ih2 = int(ih * scale)
                img = img.resize((iw2, ih2))
                ir = ImageReader(img)
                c.drawImage(ir, 30, y - ih2, width=iw2, height=ih2)
                y = y - ih2 - 12
            except Exception:
                logging.exception('Failed to embed image in PDF')

        # Prediction
        c.setFont('Helvetica-Bold', 12)
        c.drawString(30, y, 'Model Prediction')
        y -= 18
        c.setFont('Helvetica', 10)
        pred_label = prediction.get('predicted') or prediction.get('predicted_class') or prediction.get('class') or '-'
        pred_conf = prediction.get('confidence') or prediction.get('confidence_score') or '-'
        c.drawString(36, y, f'Predicted: {pred_label}')
        y -= 14
        c.drawString(36, y, f'Confidence: {pred_conf}')
        y -= 18

        probs = prediction.get('probs') or []
        labels = prediction.get('labels') or []
        if isinstance(probs, list) and probs:
            c.setFont('Helvetica-Bold', 11)
            c.drawString(30, y, 'Probability Breakdown')
            y -= 16
            c.setFont('Helvetica', 9)
            for i, p in enumerate(probs):
                lab = labels[i] if i < len(labels) else f'Class {i}'
                c.drawString(36, y, f'{lab}: {p}%')
                y -= 12
            y -= 8

        # AI summary
        if ai_summary:
            c.setFont('Helvetica-Bold', 12)
            c.drawString(30, y, 'AI Summary')
            y -= 16
            c.setFont('Helvetica', 10)
            text = c.beginText(36, y)
            for line in str(ai_summary).split('\n'):
                text.textLine(line)
                y -= 12
            c.drawText(text)
            y -= 8

        # Gemini reply
        if gemini_reply:
            c.setFont('Helvetica-Bold', 12)
            c.drawString(30, y, 'Assistant Reply')
            y -= 16
            c.setFont('Helvetica', 10)
            text = c.beginText(36, y)
            for line in str(gemini_reply).split('\n'):
                text.textLine(line)
                y -= 12
            c.drawText(text)
            y -= 8

        # Disclaimer
        c.setFont('Helvetica', 8)
        disclaimer = 'Disclaimer: This report is generated by an AI tool and is not a final clinical diagnosis. Consult a qualified clinician.'
        c.drawString(30, 40, disclaimer)

        c.showPage()
        c.save()
        buf.seek(0)
        fname = f'colonai_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf'
        return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name=fname)

    except Exception as e:
        logging.exception('Failed to generate report (ReportLab): %s', str(e))
        return jsonify({'error': 'Failed to generate PDF (ReportLab)', 'details': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True)

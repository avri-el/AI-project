# app.py
import os
import io
import logging
import traceback
import base64
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# Keras / TF model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# GenAI SDK try imports (support multiple install patterns)
try:
    from google import genai as genai_sdk
except Exception:
    try:
        import google.generativeai as genai_sdk
    except Exception:
        genai_sdk = None

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# =========== Logging ===========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("colonai")

# =========== Gemini client init ===========
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = None


def init_gemini_client(api_key: str):
    global genai_sdk
    if not genai_sdk:
        raise RuntimeError("GenAI SDK not available. Install google-generativeai or google-genai.")
    try:
        if hasattr(genai_sdk, "Client"):
            return genai_sdk.Client(api_key=api_key)
        else:
            genai_sdk.configure(api_key=api_key)
            return genai_sdk
    except Exception:
        logger.exception("Failed to initialize Gemini client")
        raise


if GEMINI_API_KEY:
    try:
        client = init_gemini_client(GEMINI_API_KEY)
        logger.info("Gemini client initialized from environment.")
    except Exception:
        logger.warning("Gemini client initialization failed; /ask will be unavailable until configured.")


# =========== Flask app & model ===========
app = Flask(__name__)

MODEL_PATH = "model/colon_diseases.h5"
if not os.path.exists(MODEL_PATH):
    logger.error("Model not found at %s", MODEL_PATH)
    raise FileNotFoundError(MODEL_PATH)

cnn_model = load_model(MODEL_PATH)
logger.info("CNN model loaded from %s", MODEL_PATH)

# default label set (will trim/extend to model output)
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
    logger.warning("Model output size (%s) differs from provided labels (%s). Adjusting.", model_output_size, len(class_labels))
    if model_output_size > len(class_labels):
        class_labels += [f"Class {i}" for i in range(len(class_labels), model_output_size)]
    else:
        class_labels = class_labels[:model_output_size]

# ensure label uniqueness
seen = {}
for i, lab in enumerate(class_labels):
    if lab in seen:
        class_labels[i] = f"{lab} ({seen[lab] + 1})"
        seen[lab] += 1
    else:
        seen[lab] = 1


# =========== Utilities ===========

def is_colon_image(path: str, red_threshold: float = 0.12) -> bool:
    """
    Heuristic check: colonoscopy mucosa often has reddish/pinkish coloration.
    This function computes the proportion of pixels where R channel is
    significantly higher than G and B. Returns True if proportion > red_threshold.
    This is a simple heuristic — tune thresholds as needed.
    """
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.int32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # Condition: red sufficiently higher than others and somewhat bright
        redness_mask = (r > 100) & (r > g + 20) & (r > b + 20)
        red_ratio = float(np.mean(redness_mask))
        logger.debug("is_colon_image: red_ratio=%.4f (threshold=%.3f) for %s", red_ratio, red_threshold, path)
        return red_ratio >= red_threshold
    except Exception:
        logger.exception("is_colon_image failed for %s", path)
        return False


def predict_image(img_path: str, target_size=(256, 256)):
    """Return idx, label, confidence (0..1), probs list (0..1)."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = cnn_model.predict(x)
    probs = list(map(float, preds[0].tolist()))
    # normalize/protect against sizing differences
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
    """Normalize incoming prediction objects from client."""
    if not pred_dict or not isinstance(pred_dict, dict):
        return None
    label = pred_dict.get("predicted_class") or pred_dict.get("prediction") or pred_dict.get("class") or pred_dict.get("label")
    confidence = pred_dict.get("confidence") or pred_dict.get("confidence_score") or pred_dict.get("conf")
    probs = pred_dict.get("probs") or pred_dict.get("probabilities") or pred_dict.get("scores")
    labels = pred_dict.get("labels") or class_labels
    # normalize numeric confidence (allow 0..1 or 0..100)
    try:
        if isinstance(confidence, (int, float)):
            if confidence <= 1:
                confidence = float(confidence) * 100.0
            else:
                confidence = float(confidence)
    except Exception:
        pass
    # probs to 0..100
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
    return {"class": label, "confidence": confidence, "probs": probs, "labels": labels}


def extract_text_from_genai_response(resp):
    """Try to extract meaningful text from various SDK response shapes."""
    try:
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict):
            # common shapes: {'candidates': [{'content': '...'}]}
            cands = resp.get("candidates") or resp.get("outputs") or resp.get("replies")
            if isinstance(cands, list) and len(cands) > 0:
                first = cands[0]
                if isinstance(first, dict):
                    for k in ("content", "message", "output", "text"):
                        if k in first:
                            return first[k]
                    return str(first)
            for k in ("response", "result", "output"):
                if k in resp:
                    return str(resp[k])
            return str(resp)
        return str(resp)
    except Exception:
        return str(resp)


# =========== Routes ===========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts multipart/form-data with 'file'. Returns prediction JSON or error."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Tidak ada file dikirim"}), 400
        file = request.files["file"]
        if not file or file.filename == "":
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

        # Pre-filter: is this plausibly a colonoscopy frame?
        if not is_colon_image(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
            return jsonify({"error": "Gambar ini tampaknya bukan citra colonoscopy. Mohon unggah gambar endoskopi kolon."}), 400

        # run model prediction
        idx, lab, conf, probs = predict_image(file_path)

        # remove upload
        try:
            os.remove(file_path)
        except Exception:
            pass

        # confidence is 0..1; apply threshold to ensure model is reasonably sure
        CONF_THRESHOLD = 0.30  # tuneable
        if conf < CONF_THRESHOLD:
            return jsonify({"error": "Model tidak cukup yakin; gambar tidak dapat diidentifikasi dengan pasti."}), 400

        probs_percent = [round(float(p) * 100, 2) for p in probs]
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
        logger.exception("Prediction failed")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts:
      - multipart/form-data with 'file' (image) and optional 'message' -> server runs prediction then calls Gemini
      - JSON with {"message": "...", "prediction": {...}} where prediction can be result from /predict
    Behavior:
      - If an image is provided and validated as colon image, allow broad questions about that case (e.g., "jelaskan hasil analisis")
      - If no image provided, restrict questions to colon-related topics only (keyword-based), unless a prediction object is provided
    """
    try:
        message = ""
        prediction = None
        image_validated = False

        # If an image file was uploaded, handle it (and run local prediction)
        if request.files and "file" in request.files:
            file = request.files["file"]
            if not file or file.filename == "":
                return jsonify({"error": "Nama file kosong"}), 400
            filename = secure_filename(file.filename)
            upload_folder = "uploads"
            os.makedirs(upload_folder, exist_ok=True)
            path = os.path.join(upload_folder, filename)
            file.save(path)
            # pre-filter
            if not is_colon_image(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
                return jsonify({"error": "Gambar tidak dikenali sebagai citra colonoscopy."}), 400
            # run prediction
            try:
                idx, lab, conf, probs = predict_image(path)
                # remove file
                try:
                    os.remove(path)
                except Exception:
                    pass
                # confidence check
                if conf < 0.25:
                    return jsonify({"error": "Model tidak cukup yakin pada gambar ini."}), 400
                prediction = {
                    "predicted_index": idx,
                    "predicted_class": lab,
                    "prediction": lab,
                    "confidence": round(conf * 100, 2),
                    "confidence_score": conf,
                    "probs": [round(float(p) * 100, 2) for p in probs],
                    "labels": class_labels
                }
                image_validated = True
            except Exception as e:
                try:
                    os.remove(path)
                except Exception:
                    pass
                logger.exception("Local prediction failed in /ask")
                prediction = {"error": str(e)}
            # user message may be provided in form
            message = (request.form.get("message") or "").strip()
        else:
            # JSON mode: message + optional prediction object
            data = request.get_json(silent=True) or {}
            message = (data.get("message") or "").strip()
            raw_pred = data.get("prediction")
            if raw_pred:
                prediction = safe_extract_prediction(raw_pred)

        # If there's no image validated AND the message is not colon-related AND no prediction -> reject
        allowed_keywords = [
            "colon", "colonoscopy", "polyp", "ulcer", "ulcerative colitis",
            "colitis", "rectum", "digestive", "crohn", "colorectal", "colon cancer", "bowel"
        ]

        if not image_validated:
            # If no message and no prediction -> nothing to act on
            if not message and not prediction:
                return jsonify({"error": "No message or validated image provided."}), 400
            # If there is a prediction object (from /predict), allow general analysis questions like "jelaskan hasil analisis"
            if not prediction:
                lower_msg = message.lower()
                if not any(k in lower_msg for k in allowed_keywords):
                    return jsonify({"reply": "Maaf, saya hanya dapat menjawab pertanyaan terkait colon disease atau hasil colonoscopy.", "prediction": prediction})

        # Build user prompt. If prediction exists, include structured prediction block.
        if prediction:
            # Build probability breakdown if available
            prob_block = "No detailed probability breakdown available."
            try:
                labels = prediction.get("labels") or class_labels
                probs = prediction.get("probs") or []
                if isinstance(labels, list) and isinstance(probs, list) and len(labels) == len(probs):
                    prob_lines = [f"- {lab}: {p}%" for lab, p in zip(labels, probs)]
                    prob_block = "\n".join(prob_lines)
                else:
                    prob_block = str(probs)
            except Exception:
                prob_block = str(prediction.get("probs"))

            user_prompt = f"""CASE DATA:
Predicted Condition: {prediction.get('predicted_class') or prediction.get('class') or '-'}
Confidence: {prediction.get('confidence') or prediction.get('confidence_score') or '-'}%
Probability Breakdown:
{prob_block}

USER QUESTION:
{message if message else '(no extra question; give a full analysis)'}
"""
        else:
            user_prompt = f"USER QUESTION:\n{message}\n"

        # Initialize Gemini client if needed
        global client
        if not client:
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if key:
                client = init_gemini_client(key)

        if not client:
            return jsonify({"reply": "Gemini client tidak dikonfigurasi pada server. Silakan set GEMINI_API_KEY." , "prediction": prediction}), 503

        # System instruction aimed at producing MEDICAL COMPLEX structured output (Medis Lengkap)
        system_instruction = (
            "You are a careful, professional medical assistant specialized in colonoscopy image interpretation. "
            "You MUST follow this output structure exactly, and you are ONLY allowed to discuss colon diseases, colonoscopy findings, "
            "and related colorectal conditions. If asked about any unrelated topics, politely refuse. "
            "You must NOT provide a definitive diagnosis — always use probabilistic language and include a clear disclaimer. "
            "Produce a structured, detailed medical-style response with these sections, and ALWAYS answer in Indonesian:\n\n"
            "1) AI Analysis Summary: concise bullets of the main findings.\n"
            "2) Probabilistic Interpretation: list candidate conditions with probabilities matching the prediction input when available.\n"
            "3) Clinical Relevance & Possible Causes: short explanation.\n"
            "4) Recommended Next Steps: numbered clinical recommendations (investigations, biopsy, referral, immediate actions).\n"
            "5) Practical Patient Advice: brief patient-facing advice if appropriate.\n"
            "6) Limitations & Disclaimer: clearly state this is not a final diagnosis and recommend consultation with a clinician.\n\n"
            "Keep language clear and suitable for clinicians; also include a brief patient-friendly summary at the top (one or two lines)."
        )

        # Build combined prompt content for Gemini
        combined_prompt_text = f"System: {system_instruction}\n\nUser: {user_prompt}"

        # Try available models (best-effort)
        preferred_models = ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5"]
        response_obj = None
        last_exc = None
        for m in preferred_models:
            try:
                # different SDKs may accept different call patterns, keep consistent with earlier usage
                response_obj = client.models.generate_content(
                    model=m,
                    contents=[{"role": "user", "parts": [{"text": combined_prompt_text}]}],
                )
                break
            except Exception as e:
                logger.warning("Model %s failed: %s", m, str(e))
                last_exc = e
                continue

        if response_obj is None:
            logger.exception("All Gemini attempts failed: %s", str(last_exc))
            return jsonify({"error": "Gemini API call failed", "details": str(last_exc)}), 502

        reply_text = extract_text_from_genai_response(response_obj)
        # return reply plus prediction (if any)
        return jsonify({"reply": reply_text, "prediction": prediction})

    except Exception as e:
        logger.exception("Unhandled error in /ask")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/download_report", methods=["POST"])
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
        if image_data and isinstance(image_data, str) and image_data.startswith('data:'):
            try:
                header, b64 = image_data.split(',', 1)
                img_bytes = base64.b64decode(b64)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
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
                logger.exception('Failed to embed image in PDF')

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
        logger.exception('Failed to generate report (ReportLab): %s', str(e))
        return jsonify({'error': 'Failed to generate PDF (ReportLab)', 'details': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True)

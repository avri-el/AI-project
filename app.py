# app.py
import os
import io
import threading
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
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.units import inch    

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

# Optional dotenv (safe if not available)
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
        logger.warning("Gemini client initialization failed; Gemini features will fallback to templated responses.")


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

# =========== Globals for session-less "last prediction" ===========
_last_prediction = None
_last_pred_lock = threading.Lock()


def set_last_prediction(pred: dict):
    global _last_prediction
    with _last_pred_lock:
        _last_prediction = pred


def get_last_prediction():
    with _last_pred_lock:
        return _last_prediction


# =========== Utilities ===========

def is_colon_image(img_path: str, red_threshold: float = 0.06) -> bool:
    """Heuristic check to see if an image looks like an endoscopy/colon image."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        if arr.size == 0:
            return False
        r = arr[:, :, 0].astype(float) / 255.0
        g = arr[:, :, 1].astype(float) / 255.0
        b = arr[:, :, 2].astype(float) / 255.0
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        diff = mx - mn + 1e-8
        hue = np.zeros_like(mx)
        mask = mx == r
        hue[mask] = (g[mask] - b[mask]) / diff[mask]
        mask = mx == g
        hue[mask] = 2.0 + (b[mask] - r[mask]) / diff[mask]
        mask = mx == b
        hue[mask] = 4.0 + (r[mask] - g[mask]) / diff[mask]
        hue = (hue / 6.0) % 1.0
        sat = (mx - mn) / (mx + 1e-8)
        val = mx
        red_mask = ((hue < 0.05) | (hue > 0.9)) & (sat > 0.2) & (val > 0.15)
        red_frac = float(np.sum(red_mask)) / float(hue.size)
        logger.debug("is_colon_image: red_frac=%s", red_frac)
        return red_frac >= red_threshold
    except Exception:
        logger.exception("is_colon_image check failed")
        return False


def predict_image(img_path: str, target_size=(256, 256)):
    """Return idx, label, confidence (0..1), probs list (0..1)."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
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
    """
    Normalize incoming prediction objects from client.
    Produces a dict with keys:
      - predicted_class (str)
      - confidence (float 0..100)
      - probs (list of floats 0..100)
      - labels (list)
    """
    if not pred_dict or not isinstance(pred_dict, dict):
        return None
    # If wrapper with 'prediction' key
    if "prediction" in pred_dict and isinstance(pred_dict["prediction"], dict):
        pred_dict = pred_dict["prediction"]
    label = (
        pred_dict.get("predicted_class")
        or pred_dict.get("predicted")
        or pred_dict.get("prediction")
        or pred_dict.get("class")
        or pred_dict.get("label")
    )
    confidence = (
        pred_dict.get("confidence")
        or pred_dict.get("confidence_score")
        or pred_dict.get("conf")
    )
    probs = (
        pred_dict.get("probs")
        or pred_dict.get("probabilities")
        or pred_dict.get("scores")
        or pred_dict.get("prob")
    )
    labels = pred_dict.get("labels") or pred_dict.get("class_labels") or class_labels

    # Normalize confidence to 0..100 float
    try:
        if isinstance(confidence, str):
            s = confidence.strip()
            if s.endswith("%"):
                confidence = float(s[:-1])
            else:
                confidence = float(s)
        if isinstance(confidence, (int, float)):
            confidence = float(confidence)
            if confidence <= 1.1:
                confidence = confidence * 100.0
    except Exception:
        confidence = None

    # normalize probs into 0..100 floats
    cleaned = []
    if isinstance(probs, list):
        for p in probs:
            try:
                pv = float(p)
                if 0 <= pv <= 1.01:
                    pv = pv * 100.0
                cleaned.append(round(pv, 2))
            except Exception:
                pass
    else:
        probs = []

    if not isinstance(labels, list):
        labels = list(class_labels)

    # Pad/truncate probs to labels length
    if len(cleaned) < len(labels):
        cleaned += [0.0] * (len(labels) - len(cleaned))
    elif len(cleaned) > len(labels):
        cleaned = cleaned[: len(labels)]

    return {
        "predicted_class": label if label is not None else "-",
        "confidence": round(float(confidence), 2) if confidence is not None else None,
        "probs": cleaned,
        "labels": labels,
    }


def extract_text_from_genai_response(resp):
    """Try to extract meaningful text from various SDK response shapes."""
    try:
        if hasattr(resp, "text"):
            return resp.text
        if isinstance(resp, dict):
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


def generate_medical_summary(prediction):
    """
    Generate a structured 'Medis Lengkap' summary for the given prediction.
    Tries Gemini first; on failure returns a templated fallback summary.
    """
    if not prediction or not isinstance(prediction, dict):
        return "Tidak ada data prediksi untuk menghasilkan ringkasan."

    pred_label = prediction.get("predicted_class") or prediction.get("prediction") or "-"
    confidence = prediction.get("confidence") or prediction.get("confidence_score") or "-"
    probs = prediction.get("probs") or []
    labels = prediction.get("labels") or class_labels

    prob_lines = []
    try:
        for i, lab in enumerate(labels):
            p = probs[i] if i < len(probs) else 0
            prob_lines.append(f"- {lab}: {p}%")
    except Exception:
        prob_lines = [str(probs)]

    user_section = f"Predicted Condition: {pred_label}\nConfidence: {confidence}%\nProbability Breakdown:\n" + "\n".join(prob_lines)

    system_instruction = (
        "You are a medical AI assistant specialized ONLY in colonoscopy interpretation and colon diseases. "
        "Produce a structured MEDICAL REPORT in Indonesian with these sections:\n\n"
        "PATIENT-FRIENDLY SUMMARY (1-2 lines)\n"
        "1) AI Analysis Summary\n2) Probabilistic Interpretation\n3) Clinical Relevance & Possible Causes\n4) Recommended Next Steps\n5) Practical Patient Advice\n6) Limitations & Disclaimer\n\n"
        "Use conditional language (mungkin, kemungkinan). Do NOT give definitive diagnoses."
    )

    prompt_text = f"System: {system_instruction}\n\nUser: Berikut data kasus:\n{user_section}\n\nTolong hasilkan laporan sesuai format di atas."

    # Try Gemini
    global client
    if client:
        try:
            preferred_models = ["models/gemini-2.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5"]
            last_exc = None
            for m in preferred_models:
                try:
                    resp = client.models.generate_content(
                        model=m,
                        contents=[{"role": "user", "parts": [{"text": prompt_text}]}],
                    )
                    text = extract_text_from_genai_response(resp)
                    if text and len(text.strip()) > 0:
                        return text
                except Exception as e:
                    logger.warning("Gemini model %s failed for summary: %s", m, str(e))
                    last_exc = e
                    continue
            logger.warning("All Gemini attempts failed for summary generation: %s", str(last_exc))
        except Exception:
            logger.exception("Exception while trying to generate summary with Gemini")

    # fallback templated summary
    try:
        top = sorted(zip(probs, labels), reverse=True)[:len(labels)]
        prob_text = "\n".join([f"- {lab}: {p}%" for p, lab in top])
    except Exception:
        prob_text = str(probs)

    fallback = (
        f"Patient Summary:\nTemuan utama: {pred_label} (Confidence {confidence}%).\n\n"
        "1) AI Analysis Summary:\n"
        f"• Temuan utama: {pred_label} (confidence {confidence}%).\n\n"
        "2) Probabilistic Interpretation:\n" + prob_text + "\n\n"
        "3) Clinical Relevance & Possible Causes:\n• Kemungkinan penyebab termasuk polip atau inflamasi tergantung konteks klinis.\n\n"
        "4) Recommended Next Steps:\n1. Konsultasi gastroenterolog.\n2. Pertimbangkan biopsi atau polypektomi sesuai tampilan endoskopik.\n\n"
        "5) Practical Patient Advice:\n• Hindari tindakan mandiri; konsultasikan ke dokter. Segera ke fasilitas kesehatan bila ada perdarahan berat atau nyeri hebat.\n\n"
        "6) Limitations & Disclaimer:\nIni bukan diagnosis final. Hasil harus dikonfirmasi oleh dokter spesialis gastroenterologi."
    )
    return fallback


# =========== Routes ===========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts multipart/form-data with 'file'. Returns prediction JSON and auto-generated summary (if possible)."""
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

        CONF_THRESHOLD = 0.30  # tuneable
        if conf < CONF_THRESHOLD:
            return jsonify({"error": "Model tidak cukup yakin; gambar tidak dapat diidentifikasi dengan pasti."}), 400

        probs_percent = [round(float(p) * 100, 2) for p in probs]
        prediction = {
            "predicted_index": idx,
            "predicted_class": lab,
            "prediction": lab,
            "confidence": round(conf * 100, 2),
            "confidence_score": conf,
            "probs": probs_percent,
            "labels": class_labels
        }

        # store as last_prediction (so /ask can use it even if frontend forgot to send)
        set_last_prediction(prediction)

        # Auto-generate medical summary (try Gemini, fallback to templated)
        try:
            summary_text = generate_medical_summary(prediction)
        except Exception:
            logger.exception("Summary generation failed")
            summary_text = ""

        return jsonify({
            "prediction": prediction,
            "summary": summary_text,
            # backward compatibility for frontend
            "predicted_index": idx,
            "predicted_class": lab,
            "confidence": round(conf * 100, 2),
            "probs": probs_percent,
            "labels": class_labels
        })
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """
    Chat endpoint:
    - Accepts JSON { message, prediction? , summary? } OR multipart with file + form message.
    - If prediction not provided, will use last server-side prediction if available.
    - If neither prediction nor last prediction exists, will require colon-related keywords.
    - Attempts to call Gemini when available; otherwise uses fallback answer generation.
    """
    try:
        # accept both JSON and form-data
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(silent=True) or {}
            message = (data.get("message") or "").strip()
            raw_pred = data.get("prediction")
            incoming_summary = data.get("summary")
            if raw_pred:
                extracted = safe_extract_prediction(raw_pred)
                if extracted:
                    if incoming_summary:
                        extracted["summary"] = incoming_summary
                    prediction = extracted
                    set_last_prediction(prediction)
                else:
                    prediction = None
            else:
                prediction = None
        else:
            # multipart/form-data path (maybe with uploaded file)
            prediction = None
            message = ""
            if request.files and "file" in request.files:
                file = request.files["file"]
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    upload_folder = "uploads"
                    os.makedirs(upload_folder, exist_ok=True)
                    path = os.path.join(upload_folder, filename)
                    file.save(path)
                    try:
                        if not is_colon_image(path):
                            try:
                                os.remove(path)
                            except Exception:
                                pass
                            return jsonify({"error": "Gambar tidak dikenali sebagai citra colonoscopy."}), 400
                        idx, lab, conf, probs = predict_image(path)
                        try:
                            os.remove(path)
                        except Exception:
                            pass
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
                        set_last_prediction(prediction)
                    except Exception as err:
                        try:
                            os.remove(path)
                        except Exception:
                            pass
                        logger.exception("Local prediction failed in /ask")
                        return jsonify({"error": "Local prediction failed", "details": str(err)}), 500
            message = (request.form.get("message") or "").strip()

        # If no prediction in payload, use last server-side prediction
        if not locals().get("prediction"):
            prediction = get_last_prediction()

        # validation and allowed keywords
        allowed_keywords = [
            "colon", "colonoscopy", "polyp", "ulcer", "ulcerative colitis",
            "colitis", "rectum", "digestive", "crohn", "colorectal", "colon cancer", "bowel"
        ]
        followup_keywords = [
            "jelaskan", "artinya", "penjelasan", "apa maksudnya",
            "lanjutkan", "detailkan", "interpretasi", "analisis", "hasil", "penyakit ini", "kasus ini", "dampak", "resiko", "risiko"
        ]

        if not prediction:
            # no context — require colon-related question
            if not message:
                return jsonify({"error": "No message or prediction available."}), 400
            lower_msg = message.lower()
            if not any(k in lower_msg for k in allowed_keywords + followup_keywords):
                return jsonify({"reply": "Maaf, saya hanya dapat menjawab pertanyaan terkait hasil colonoscopy atau penyakit kolon."}), 400

        # Build user prompt/context
        if prediction:
            labels = prediction.get("labels") or class_labels
            probs = prediction.get("probs") or []
            try:
                prob_lines = "\n".join([f"- {labels[i]}: {probs[i]}%" for i in range(len(labels))])
            except Exception:
                prob_lines = str(probs)
            summary_block = prediction.get("summary", "")
            case_block = f"""Predicted Condition: {prediction.get('predicted_class') or '-'}
Confidence: {prediction.get('confidence') or '-'}%
Probability Breakdown:
{prob_lines}

Existing AI Summary:
{summary_block}
"""
            user_prompt = f"{case_block}\nUser question: {message}\n"
        else:
            user_prompt = f"User question: {message}\n"

        # Setup Gemini prompt for short Q/A (don't force full 6-section unless requested)
        system_instruction = (
            "You are a professional medical assistant specialized in colonoscopy interpretation and colon diseases. "
            "Answer in Indonesian. Use conditional language and do NOT provide definitive diagnosis. "
            "If the user asks for a full formal report, produce the 6-section 'Medis Lengkap' format. "
            "Otherwise answer the user's question directly and concisely for clinicians and provide a short patient-friendly note if relevant."
        )
        combined_prompt_text = f"System: {system_instruction}\n\nContext:\n{user_prompt}"

        # Try to ensure client is initialized
        global client
        if not client and (GEMINI_API_KEY := os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            try:
                client = init_gemini_client(GEMINI_API_KEY)
            except Exception:
                client = None

        # If no Gemini client, return a contextual fallback (not just the summary)
        if not client:
            # generate fallback answer focusing on user's question
            fallback_answer = generate_contextual_fallback(prediction, message)
            return jsonify({"reply": fallback_answer, "prediction": prediction})

        # Call Gemini (guarded)
        preferred_model = "models/gemini-2.5-flash"

        try:
            import time
            time.sleep(1.2)   # anti-429

            response_obj = client.models.generate_content(
                model=preferred_model,
                contents=[{"role": "user", "parts": [{"text": combined_prompt_text}]}],
            )

        except Exception as e:
            logger.warning("Gemini failed: %s", str(e))
            fallback_answer = generate_contextual_fallback(prediction, message)
            fallback_answer += "\n\nCatatan: Gemini API gagal merespons; ini adalah jawaban fallback."
            return jsonify({"reply": fallback_answer, "prediction": prediction})


        if response_obj is None:
            logger.exception("All Gemini attempts failed for /ask: %s", str(last_exc))
            fallback_answer = generate_contextual_fallback(prediction, message)
            fallback_answer += "\n\nCatatan: Gemini API gagal merespons; ini adalah jawaban fallback."
            return jsonify({"reply": fallback_answer, "prediction": prediction}), 200

        reply_text = extract_text_from_genai_response(response_obj)
        if not reply_text or len(reply_text.strip()) == 0:
            reply_text = generate_contextual_fallback(prediction, message)
        return jsonify({"reply": reply_text, "prediction": prediction})

    except Exception as e:
        logger.exception("Unhandled error in /ask")
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def generate_contextual_fallback(prediction, message):
    """
    Produce a fallback answer that addresses the user's question using
    the available prediction context (instead of reprinting the whole summary).
    This ensures that when Gemini fails, the user still gets a helpful reply.
    """
    if not prediction:
        return "Gemini tidak tersedia dan tidak ada konteks prediksi. Mohon jalankan analisis gambar terlebih dahulu."

    label = prediction.get("predicted_class", "-")
    conf = prediction.get("confidence", "-")
    probs = prediction.get("probs", [])[:4]
    # Simple rule-based answers for common question types (dampak, langkah, risiko)
    q = (message or "").lower()
    if any(k in q for k in ("dampak", "pengaruh", "efek", "konsekuensi")):
        return (
            f"Berdasarkan temuan ({label}, confidence {conf}%), dampak klinis yang mungkin:\n"
            "• Gejala: diare, nyeri perut, perdarahan rektal, demam pada beberapa kasus.\n"
            "• Komplikasi: anemia, abses, perforasi (jarang), dan risiko neoplasia jangka panjang jika inflamasi kronis.\n"
            "Rekomendasi: konsultasi gastroenterolog untuk evaluasi lebih lanjut dan pertimbangkan biopsi atau pemeriksaan histopatologi."
        )
    if any(k in q for k in ("langkah", "selanjutnya", "apa yang harus")):
        return (
            "Langkah rekomendasi singkat:\n"
            "1) Konsultasi gastroenterologi segera.\n"
            "2) Jika lesi terlihat mencurigakan: pertimbangkan biopsy atau polypektomi.\n"
            "3) Pemeriksaan lanjutan (CT, kolonoskopi lanjutan) sesuai indikasi.\n"
            "4) Kirim spesimen untuk histopatologi bila diangkat."
        )
    if any(k in q for k in ("risiko", "resiko", "komplikasi")):
        return (
            "Risiko yang mungkin terkait kondisi ini termasuk perdarahan, obstruksi jika lesi besar, dan peningkatan risiko neoplasia pada peradangan kronis. Penilaian klinis dan histopatologi diperlukan."
        )
    # Default fallback: brief explanation + advise
    return (
        f"Temuan utama: {label} (confidence {conf}%).\n"
        "Secara singkat: kondisi ini menunjukkan adanya perubahan mukosa/lesi yang perlu evaluasi klinis dan histologis lebih lanjut.\n"
        "Rekomendasi: konsultasikan ke gastroenterolog untuk rencana tindakan (biopsi/pengawasan)."
    )


@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    prediction = data.get('prediction') or {}
    ai_summary = data.get('aiSummary') or ''
    gemini_reply = data.get('geminiReply') or ''
    metadata = data.get('metadata') or {}
    image_data = data.get('imageData')

    ### --- PREPARE PDF BUFFER ---
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    elements = []
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    normal.fontSize = 10
    normal.leading = 14

    header = Paragraph("<b>ColonAI — Analysis Report</b>", styles["Title"])
    elements.append(header)
    elements.append(Spacer(1, 12))

    generated = Paragraph(
        f"<font size=9>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</font>",
        normal
    )
    elements.append(generated)
    elements.append(Spacer(1, 12))

    ### --- METADATA ---
    if metadata:
        elements.append(Paragraph("<b>Case Metadata</b>", styles["Heading3"]))
        for k, v in metadata.items():
            elements.append(Paragraph(f"{k}: {v}", normal))
        elements.append(Spacer(1, 12))

    ### --- IMAGE HANDLING ---
    if image_data and isinstance(image_data, str) and image_data.startswith("data:"):
        try:
            header, b64 = image_data.split(",", 1)
            img_bytes = base64.b64decode(b64)
            tmp = io.BytesIO(img_bytes)
            img = RLImage(tmp)

            img._restrictSize(6*inch, 4*inch)
            elements.append(img)
            elements.append(Spacer(1, 12))
        except Exception as e:
            logger.exception("Failed to embed image")

    ### --- MODEL PREDICTION ---
    pred_label = prediction.get('predicted') or prediction.get('predicted_class') or '-'
    pred_conf = prediction.get('confidence') or prediction.get('confidence_score') or '-'

    elements.append(Paragraph("<b>Model Prediction</b>", styles["Heading3"]))
    elements.append(Paragraph(f"Predicted: {pred_label}", normal))
    elements.append(Paragraph(f"Confidence: {pred_conf}", normal))
    elements.append(Spacer(1, 12))

    ### --- PROBABILITY BREAKDOWN ---
    probs = prediction.get("probs") or []
    labels = prediction.get("labels") or []

    if probs:
        elements.append(Paragraph("<b>Probability Breakdown</b>", styles["Heading3"]))
        for i, p in enumerate(probs):
            label = labels[i] if i < len(labels) else f"Class {i}"
            elements.append(Paragraph(f"{label}: {p}%", normal))
        elements.append(Spacer(1, 12))

    ### --- AI SUMMARY ---
    if ai_summary:
        elements.append(Paragraph("<b>AI Summary</b>", styles["Heading3"]))
        elements.append(Paragraph(ai_summary.replace("\n", "<br/>"), normal))
        elements.append(Spacer(1, 12))

    ### --- ASSISTANT (GEMINI) REPLY ---
    if gemini_reply:
        elements.append(Paragraph("<b>Assistant Reply</b>", styles["Heading3"]))
        elements.append(Paragraph(gemini_reply.replace("\n", "<br/>"), normal))
        elements.append(Spacer(1, 12))

    ### --- DISCLAIMER (WRAPPED & SAFE) ---
    disc = (
        "Disclaimer: This report is generated by an AI tool and is not a final clinical "
        "diagnosis. Consult a qualified clinician."
    )
    elements.append(Paragraph(f"<font size=8>{disc}</font>", normal))

    ### --- BUILD PDF ---
    doc.build(elements)
    buf.seek(0)

    fname = f"colonai_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=fname)


if __name__ == "__main__":
    app.run(debug=True)

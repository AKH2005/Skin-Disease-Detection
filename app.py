import os
os.environ["KERAS_BACKEND"] = "jax"  # must be set before keras is imported

import io
import uuid
import tempfile
import traceback
import threading

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import numpy as np
from PIL import Image, UnidentifiedImageError

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
CORS(app)

# ---------------- MODEL ----------------
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    with _model_lock:
        if _model is None:
            import keras  # standalone Keras 3 — compatible with .keras format
            _model = keras.models.load_model("model.keras", compile=False)
    return _model

# ---------------- REFERENCE CLASSES ----------------
# IMPORTANT:
# These MUST exactly match the order used during model training
CLASSES = [
    "1. Eczema 1677",
    "10. Warts Molluscum and other Viral Infections - 2103",
    "2. Melanoma 15.75k",
    "3. Atopic Dermatitis - 1.25k",
    "4. Basal Cell Carcinoma (BCC) 3323",
    "5. Melanocytic Nevi (NV) - 7970",
    "6. Benign Keratosis-like Lesions (BKL) 2624",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k"
]

# ---------------- DISPLAY NAMES ----------------
DISPLAY_NAMES = {
    "1. Eczema 1677": "Eczema",
    "10. Warts Molluscum and other Viral Infections - 2103": "Viral Infection",
    "2. Melanoma 15.75k": "Melanoma",
    "3. Atopic Dermatitis - 1.25k": "Atopic Dermatitis",
    "4. Basal Cell Carcinoma (BCC) 3323": "Basal Cell Carcinoma",
    "5. Melanocytic Nevi (NV) - 7970": "Melanocytic Nevi",
    "6. Benign Keratosis-like Lesions (BKL) 2624": "Benign Keratosis",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "Psoriasis",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "Seborrheic Keratosis",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "Fungal Infection"
}

# ---------------- WIDE RANGE DISEASE KNOWLEDGE ----------------
DISEASE_KNOWLEDGE = {
    "1. Eczema 1677": {
        "overview": "Eczema is a common inflammatory skin condition that causes dryness, itching, and irritation.",
        "simple_explanation": "It is a skin problem where the skin becomes dry, itchy, and sensitive.",
        "common_symptoms": ["Dry skin", "Itching", "Red patches", "Inflamed skin", "Rough or scaly skin"],
        "early_symptoms": ["Mild itching", "Dry patches", "Redness"],
        "advanced_symptoms": ["Cracked skin", "Bleeding from scratching", "Skin thickening"],
        "possible_causes": ["Weak skin barrier", "Allergies", "Irritants", "Stress", "Weather changes"],
        "risk_factors": ["Family history of allergies", "Asthma", "Sensitive skin"],
        "contagious": "No",
        "condition_type": "Chronic / recurring",
        "severity_level": "Low to Moderate",
        "safe_home_care": ["Apply fragrance-free moisturizer regularly", "Use lukewarm water while bathing", "Keep nails short"],
        "daily_skin_routine": ["Clean gently with mild soap", "Moisturize after bathing", "Avoid strong skincare products"],
        "what_to_avoid": ["Harsh soaps", "Hot showers", "Scratching the skin", "Known allergens"],
        "hygiene_advice": ["Keep affected area clean", "Do not rub aggressively"],
        "sun_exposure_advice": ["Avoid excessive sun exposure if skin is irritated"],
        "clothing_advice": ["Wear loose cotton clothes", "Avoid rough fabrics like wool"],
        "stress_sleep_advice": ["Manage stress", "Get enough sleep"],
        "foods_to_eat": ["Omega-3 rich foods", "Fruits", "Leafy vegetables", "Hydrating foods"],
        "foods_to_avoid": ["Known food triggers", "Highly processed foods"],
        "hydration_advice": ["Drink enough water daily"],
        "general_vitamins": ["Vitamin E rich foods", "Omega-3 foods"],
        "when_to_see_doctor": ["If rash spreads rapidly", "If itching becomes severe", "If skin becomes infected"],
        "urgent_warning_signs": ["Pus or oozing", "Severe swelling", "Pain or fever"],
        "emergency_red_flags": ["Rapid worsening with fever", "Severe skin infection signs"],
        "work_school_precautions": ["Avoid irritant exposure"],
        "exercise_precautions": ["Shower after sweating", "Avoid overheating"],
        "seasonal_triggers": ["Winter dryness", "Hot humid weather"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Likely eczema-like inflammatory skin condition requiring skin barrier care and irritation avoidance."
    },

    "2. Melanoma 15.75k": {
        "overview": "Melanoma is a serious form of skin cancer that develops in pigment-producing cells.",
        "simple_explanation": "It is a potentially dangerous skin cancer that must be checked by a doctor quickly.",
        "common_symptoms": ["Changing mole", "Irregular borders", "Uneven color", "Itching", "Bleeding lesion"],
        "early_symptoms": ["New unusual mole", "Color change", "Size increase"],
        "advanced_symptoms": ["Bleeding", "Pain", "Ulceration", "Rapid growth"],
        "possible_causes": ["UV exposure", "Genetic factors", "Repeated sunburns"],
        "risk_factors": ["Fair skin", "Family history", "Multiple moles", "Excess sun exposure"],
        "contagious": "No",
        "condition_type": "Potentially life-threatening",
        "severity_level": "High",
        "safe_home_care": ["Do not delay professional evaluation", "Protect lesion from irritation"],
        "daily_skin_routine": ["Avoid picking or scratching", "Observe for changes"],
        "what_to_avoid": ["Ignoring skin changes", "Excessive sun exposure", "Self-treatment attempts"],
        "hygiene_advice": ["Keep area clean and dry"],
        "sun_exposure_advice": ["Strict sun protection is strongly advised"],
        "clothing_advice": ["Use protective clothing outdoors"],
        "stress_sleep_advice": ["Seek emotional support if anxious"],
        "foods_to_eat": ["Balanced diet", "Antioxidant-rich fruits and vegetables"],
        "foods_to_avoid": ["No specific food cure exists"],
        "hydration_advice": ["Maintain normal hydration"],
        "general_vitamins": ["General healthy diet only"],
        "when_to_see_doctor": ["Immediately if mole changes", "If bleeding or itching occurs", "If lesion enlarges"],
        "urgent_warning_signs": ["Bleeding", "Rapid color change", "Painful lesion"],
        "emergency_red_flags": ["Rapid progression", "Non-healing lesion", "Systemic symptoms"],
        "work_school_precautions": ["Avoid intense sun exposure"],
        "exercise_precautions": ["Protect affected skin during activity"],
        "seasonal_triggers": ["Summer UV exposure"],
        "doctor_specialist": "Dermatologist / Oncologist",
        "report_summary": "Potentially serious melanoma-like lesion requiring urgent dermatologist evaluation."
    },

    "3. Atopic Dermatitis - 1.25k": {
        "overview": "Atopic dermatitis is a chronic inflammatory skin condition often associated with allergies.",
        "simple_explanation": "It causes itchy, dry, and irritated skin, especially in people with sensitive skin.",
        "common_symptoms": ["Itching", "Dryness", "Red patches", "Inflammation"],
        "early_symptoms": ["Dryness", "Mild itching"],
        "advanced_symptoms": ["Cracked skin", "Skin thickening", "Secondary infection"],
        "possible_causes": ["Immune sensitivity", "Allergens", "Skin barrier weakness"],
        "risk_factors": ["Allergy history", "Asthma", "Sensitive skin"],
        "contagious": "No",
        "condition_type": "Chronic / recurring",
        "severity_level": "Low to Moderate",
        "safe_home_care": ["Moisturize frequently", "Use mild cleansers", "Avoid triggers"],
        "daily_skin_routine": ["Apply moisturizer twice daily", "Use soft towels"],
        "what_to_avoid": ["Fragranced products", "Harsh detergents", "Scratching"],
        "hygiene_advice": ["Keep skin clean but not over-washed"],
        "sun_exposure_advice": ["Avoid overheating and sunburn"],
        "clothing_advice": ["Use soft cotton clothing"],
        "stress_sleep_advice": ["Stress management helps reduce flare-ups"],
        "foods_to_eat": ["Balanced anti-inflammatory diet", "Hydrating foods"],
        "foods_to_avoid": ["Known food allergens"],
        "hydration_advice": ["Stay well hydrated"],
        "general_vitamins": ["General healthy nutrition"],
        "when_to_see_doctor": ["If severe itching persists", "If infection develops"],
        "urgent_warning_signs": ["Oozing", "Severe redness", "Pain"],
        "emergency_red_flags": ["Fever with skin worsening"],
        "work_school_precautions": ["Avoid chemical irritants"],
        "exercise_precautions": ["Shower after sweating"],
        "seasonal_triggers": ["Winter dryness"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Likely atopic dermatitis pattern with skin sensitivity and recurring irritation."
    },

    "4. Basal Cell Carcinoma (BCC) 3323": {
        "overview": "Basal cell carcinoma is a common form of skin cancer that usually grows slowly.",
        "simple_explanation": "It is a skin cancer that often appears as a shiny bump or sore that doesn't heal.",
        "common_symptoms": ["Pearly bump", "Non-healing sore", "Bleeding spot"],
        "early_symptoms": ["Small shiny lesion", "Slowly changing skin spot"],
        "advanced_symptoms": ["Ulceration", "Repeated bleeding", "Persistent sore"],
        "possible_causes": ["Long-term UV exposure", "Sun damage"],
        "risk_factors": ["Older age", "Fair skin", "Chronic sun exposure"],
        "contagious": "No",
        "condition_type": "Cancerous but usually slow-growing",
        "severity_level": "Moderate to High",
        "safe_home_care": ["Do not attempt self-removal", "Protect lesion from irritation"],
        "daily_skin_routine": ["Keep area clean", "Observe for change"],
        "what_to_avoid": ["Ignoring lesion", "Delaying dermatologist visit"],
        "hygiene_advice": ["Clean gently"],
        "sun_exposure_advice": ["Strong sun protection recommended"],
        "clothing_advice": ["Use protective clothing"],
        "stress_sleep_advice": ["Seek medical follow-up early"],
        "foods_to_eat": ["Balanced healthy diet"],
        "foods_to_avoid": ["No specific diet cure"],
        "hydration_advice": ["Normal hydration"],
        "general_vitamins": ["General nutrition only"],
        "when_to_see_doctor": ["As soon as possible for diagnosis"],
        "urgent_warning_signs": ["Bleeding lesion", "Rapid growth"],
        "emergency_red_flags": ["Aggressive spread", "Persistent ulceration"],
        "work_school_precautions": ["Avoid strong UV exposure"],
        "exercise_precautions": ["Protect lesion from trauma"],
        "seasonal_triggers": ["Summer sun exposure"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Possible basal cell carcinoma-like lesion requiring dermatologist assessment."
    },

    "5. Melanocytic Nevi (NV) - 7970": {
        "overview": "Melanocytic nevi are common moles that are usually benign.",
        "simple_explanation": "These are usually harmless skin moles, but they should be watched for unusual changes.",
        "common_symptoms": ["Flat or raised mole", "Brown or dark spot"],
        "early_symptoms": ["Stable pigmented spot"],
        "advanced_symptoms": ["Usually none unless changing"],
        "possible_causes": ["Pigment cell clustering", "Genetics", "Sun exposure"],
        "risk_factors": ["Multiple moles", "Family history"],
        "contagious": "No",
        "condition_type": "Usually benign",
        "severity_level": "Low",
        "safe_home_care": ["Monitor for changes", "Protect from excessive sun"],
        "daily_skin_routine": ["Observe mole regularly"],
        "what_to_avoid": ["Ignoring rapid change in shape or color"],
        "hygiene_advice": ["Normal skin hygiene"],
        "sun_exposure_advice": ["Use sunscreen"],
        "clothing_advice": ["Protective clothing outdoors"],
        "stress_sleep_advice": ["Routine monitoring is enough"],
        "foods_to_eat": ["Balanced healthy diet"],
        "foods_to_avoid": ["No specific restriction"],
        "hydration_advice": ["Normal hydration"],
        "general_vitamins": ["General healthy nutrition"],
        "when_to_see_doctor": ["If mole changes in size, shape, or color"],
        "urgent_warning_signs": ["Bleeding", "Irregular border", "Rapid evolution"],
        "emergency_red_flags": ["Fast-changing pigmented lesion"],
        "work_school_precautions": ["None specific"],
        "exercise_precautions": ["Avoid repeated friction if irritated"],
        "seasonal_triggers": ["Sun exposure may darken lesion"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Likely benign mole-like lesion, but any change should be medically evaluated."
    },

    "6. Benign Keratosis-like Lesions (BKL) 2624": {
        "overview": "Benign keratosis refers to non-cancerous skin growths that are often harmless.",
        "simple_explanation": "These are usually harmless rough or raised skin growths.",
        "common_symptoms": ["Rough patch", "Raised lesion", "Waxy or scaly appearance"],
        "early_symptoms": ["Small rough spot"],
        "advanced_symptoms": ["Larger raised lesion", "Irritation"],
        "possible_causes": ["Skin aging", "Sun exposure", "Benign skin growth"],
        "risk_factors": ["Older age", "Sun exposure"],
        "contagious": "No",
        "condition_type": "Benign / non-cancerous",
        "severity_level": "Low",
        "safe_home_care": ["Do not scratch or pick", "Keep skin moisturized"],
        "daily_skin_routine": ["Gentle cleansing", "Observe for irritation"],
        "what_to_avoid": ["Self-removal attempts"],
        "hygiene_advice": ["Keep area clean"],
        "sun_exposure_advice": ["Use sun protection"],
        "clothing_advice": ["Avoid friction if irritated"],
        "stress_sleep_advice": ["No major lifestyle restriction"],
        "foods_to_eat": ["Balanced healthy diet"],
        "foods_to_avoid": ["No specific restrictions"],
        "hydration_advice": ["Maintain hydration"],
        "general_vitamins": ["General healthy nutrition"],
        "when_to_see_doctor": ["If lesion changes or becomes painful"],
        "urgent_warning_signs": ["Bleeding", "Rapid shape change"],
        "emergency_red_flags": ["Suspicious rapid progression"],
        "work_school_precautions": ["None specific"],
        "exercise_precautions": ["Avoid rubbing lesion repeatedly"],
        "seasonal_triggers": ["Sun irritation"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Likely benign keratosis-like skin growth with low immediate concern."
    },

    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": {
        "overview": "Psoriasis and related inflammatory skin diseases can cause red, scaly, itchy, or thickened skin patches.",
        "simple_explanation": "It is a long-term inflammatory skin condition that often causes red, itchy, or scaly skin.",
        "common_symptoms": ["Scaly plaques", "Redness", "Dryness", "Itching"],
        "early_symptoms": ["Small red patches", "Mild scaling"],
        "advanced_symptoms": ["Thick plaques", "Cracks", "Joint symptoms in some cases"],
        "possible_causes": ["Autoimmune activity", "Stress", "Skin injury", "Infections"],
        "risk_factors": ["Family history", "Stress", "Smoking"],
        "contagious": "No",
        "condition_type": "Chronic autoimmune / inflammatory condition",
        "severity_level": "Moderate",
        "safe_home_care": ["Moisturize skin", "Avoid skin trauma", "Use gentle cleansers"],
        "daily_skin_routine": ["Daily moisturizing", "Gentle bathing"],
        "what_to_avoid": ["Scratching", "Harsh products", "Smoking"],
        "hygiene_advice": ["Keep skin clean and moisturized"],
        "sun_exposure_advice": ["Mild sunlight may help, but avoid sunburn"],
        "clothing_advice": ["Wear soft breathable fabrics"],
        "stress_sleep_advice": ["Stress control is helpful"],
        "foods_to_eat": ["Anti-inflammatory foods", "Omega-3 rich foods", "Vegetables"],
        "foods_to_avoid": ["Alcohol excess", "Highly processed foods"],
        "hydration_advice": ["Stay hydrated"],
        "general_vitamins": ["Balanced nutrition"],
        "when_to_see_doctor": ["If plaques spread", "If pain or joint stiffness develops"],
        "urgent_warning_signs": ["Widespread flare", "Painful cracks", "Joint pain"],
        "emergency_red_flags": ["Severe widespread inflammation"],
        "work_school_precautions": ["Avoid skin irritation triggers"],
        "exercise_precautions": ["Shower after sweating", "Avoid skin trauma"],
        "seasonal_triggers": ["Winter flare-ups"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Likely psoriasis-like chronic inflammatory skin condition needing skin care and flare control."
    },

    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": {
        "overview": "Seborrheic keratosis is a harmless skin growth often seen in older adults.",
        "simple_explanation": "It is usually a harmless raised skin growth that may look waxy or stuck on the skin.",
        "common_symptoms": ["Waxy lesion", "Raised growth", "Brown or black patch"],
        "early_symptoms": ["Small rough bump"],
        "advanced_symptoms": ["Larger raised lesion", "Occasional irritation"],
        "possible_causes": ["Skin aging", "Genetics"],
        "risk_factors": ["Older age", "Family history"],
        "contagious": "No",
        "condition_type": "Benign / non-cancerous",
        "severity_level": "Low",
        "safe_home_care": ["Do not scratch", "Observe for unusual change"],
        "daily_skin_routine": ["Normal gentle skin care"],
        "what_to_avoid": ["Picking or self-removal"],
        "hygiene_advice": ["Keep skin clean"],
        "sun_exposure_advice": ["Use sun protection"],
        "clothing_advice": ["Avoid friction if irritated"],
        "stress_sleep_advice": ["No major restriction"],
        "foods_to_eat": ["Balanced diet"],
        "foods_to_avoid": ["No specific restrictions"],
        "hydration_advice": ["Maintain hydration"],
        "general_vitamins": ["General healthy nutrition"],
        "when_to_see_doctor": ["If lesion changes, bleeds, or becomes painful"],
        "urgent_warning_signs": ["Bleeding", "Rapid change"],
        "emergency_red_flags": ["Suspicious malignant-like change"],
        "work_school_precautions": ["None specific"],
        "exercise_precautions": ["Avoid repeated rubbing"],
        "seasonal_triggers": ["No major seasonal pattern"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Likely seborrheic keratosis-like benign skin growth with low immediate concern."
    },

    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": {
        "overview": "Fungal skin infections occur when fungi grow on the skin, often causing itching and redness.",
        "simple_explanation": "It is a skin infection caused by fungus and often causes itching, scaling, or ring-like rashes.",
        "common_symptoms": ["Itching", "Red rash", "Scaling", "Ring-like patches"],
        "early_symptoms": ["Small itchy patch", "Mild redness"],
        "advanced_symptoms": ["Spreading rash", "Cracking", "Persistent irritation"],
        "possible_causes": ["Fungal overgrowth", "Sweating", "Poor ventilation", "Contaminated surfaces"],
        "risk_factors": ["Sweating", "Tight clothing", "Shared towels", "Weak immunity"],
        "contagious": "Yes, in some cases",
        "condition_type": "Infectious but usually treatable",
        "severity_level": "Low to Moderate",
        "safe_home_care": ["Keep skin dry", "Wear breathable clothing", "Avoid sharing personal items"],
        "daily_skin_routine": ["Wash and dry affected area properly", "Change sweaty clothes quickly"],
        "what_to_avoid": ["Scratching", "Moist environment", "Sharing towels"],
        "hygiene_advice": ["Maintain good skin hygiene", "Keep folds dry"],
        "sun_exposure_advice": ["No specific restriction"],
        "clothing_advice": ["Wear loose breathable cotton clothes"],
        "stress_sleep_advice": ["Maintain immunity through healthy routine"],
        "foods_to_eat": ["Balanced diet", "Low sugar foods", "Immune-supporting foods"],
        "foods_to_avoid": ["Excess sugary foods"],
        "hydration_advice": ["Stay hydrated"],
        "general_vitamins": ["General healthy nutrition"],
        "when_to_see_doctor": ["If infection spreads", "If not improving", "If pain develops"],
        "urgent_warning_signs": ["Rapid spread", "Severe redness", "Secondary infection"],
        "emergency_red_flags": ["Extensive infection with fever"],
        "work_school_precautions": ["Do not share clothing or towels"],
        "exercise_precautions": ["Shower after exercise", "Keep skin dry"],
        "seasonal_triggers": ["Humid and sweaty weather"],
        "doctor_specialist": "Dermatologist",
        "report_summary": "Likely fungal skin infection pattern requiring dryness, hygiene, and medical review if persistent."
    },

    "10. Warts Molluscum and other Viral Infections - 2103": {
        "overview": "Viral skin infections are caused by viruses and can produce rashes, bumps, or irritation.",
        "simple_explanation": "It is a skin condition caused by a virus and may need medical attention depending on severity.",
        "common_symptoms": ["Rash", "Bumps", "Blisters", "Redness"],
        "early_symptoms": ["Localized irritation", "Small lesions"],
        "advanced_symptoms": ["Spread of lesions", "Pain", "Blistering"],
        "possible_causes": ["Viral exposure", "Skin contact", "Reduced immunity"],
        "risk_factors": ["Weak immunity", "Close contact exposure"],
        "contagious": "Sometimes, depending on the virus",
        "condition_type": "Infectious / may need medical evaluation",
        "severity_level": "Moderate",
        "safe_home_care": ["Avoid touching lesions unnecessarily", "Maintain hygiene", "Do not share personal items"],
        "daily_skin_routine": ["Keep area clean and dry", "Avoid irritating products"],
        "what_to_avoid": ["Scratching", "Sharing towels", "Ignoring spreading lesions"],
        "hygiene_advice": ["Wash hands regularly", "Keep skin clean"],
        "sun_exposure_advice": ["Avoid irritation from excessive sun if skin is inflamed"],
        "clothing_advice": ["Wear breathable clean clothing"],
        "stress_sleep_advice": ["Rest well to support immunity"],
        "foods_to_eat": ["Immune-supporting foods", "Vitamin C rich fruits", "Balanced meals"],
        "foods_to_avoid": ["Highly processed foods"],
        "hydration_advice": ["Drink enough fluids"],
        "general_vitamins": ["Vitamin C rich foods", "Balanced nutrition"],
        "when_to_see_doctor": ["If lesions spread", "If pain increases", "If fever develops"],
        "urgent_warning_signs": ["Painful blistering", "Rapid spread", "Fever"],
        "emergency_red_flags": ["Severe widespread rash with systemic symptoms"],
        "work_school_precautions": ["Avoid close contact if contagious"],
        "exercise_precautions": ["Avoid excessive sweating if lesions are irritated"],
        "seasonal_triggers": ["Some viral rashes worsen with stress or low immunity"],
        "doctor_specialist": "Dermatologist / General Physician",
        "report_summary": "Likely viral skin infection pattern needing hygiene care and medical review if worsening."
    }
}

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image_file):
    # Manual ResNet50 preprocess_input — avoids importing tensorflow.keras entirely
    # Equivalent to tf.keras.applications.resnet50.preprocess_input:
    # converts RGB to BGR, then zero-centers each channel using ImageNet means
    try:
        img = Image.open(image_file).convert("RGB").resize((224, 224))
    except UnidentifiedImageError:
        raise ValueError("Invalid image file. Please upload a valid JPG or PNG image.")
    except Exception:
        raise ValueError("Invalid image file.")

    arr = np.array(img, dtype=np.float32)

    # RGB -> BGR
    arr = arr[..., ::-1]

    # Zero-center by ImageNet mean (BGR order)
    arr[..., 0] -= 103.939  # B
    arr[..., 1] -= 116.779  # G
    arr[..., 2] -= 123.68   # R

    return np.expand_dims(arr, 0)

# ---------------- AI SUMMARY ----------------
def generate_ai_summary(display_name, confidence, guide):
    return (
        f"The uploaded skin image appears most consistent with {display_name} "
        f"with approximately {confidence:.2f}% model confidence. "
        f"This condition is generally classified as {guide.get('severity_level', 'unknown severity')} "
        f"and may involve symptoms such as {', '.join(guide.get('common_symptoms', [])[:3])}. "
        f"Basic care may include {', '.join(guide.get('safe_home_care', [])[:2])}. "
        f"Medical review is recommended if warning signs such as "
        f"{', '.join(guide.get('urgent_warning_signs', [])[:2])} appear. "
        f"This is AI-generated guidance and not a final medical diagnosis."
    )

# ---------------- PDF ----------------
def generate_pdf(name, age, gender, disease, confidence, guide):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    def add_list(title, items):
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        for item in items:
            story.append(Paragraph(f"• {item}", styles["Normal"]))
        story.append(Spacer(1, 8))

    story.append(Paragraph("DermAI Advanced Skin Intelligence Report", styles["Heading1"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Patient Name:</b> {name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Age:</b> {age}", styles["Normal"]))
    story.append(Paragraph(f"<b>Gender:</b> {gender}", styles["Normal"]))
    story.append(Paragraph(f"<b>Predicted Condition:</b> {disease}", styles["Normal"]))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Overview</b>", styles["Heading3"]))
    story.append(Paragraph(guide.get("overview", "Not available"), styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Simple Explanation</b>", styles["Heading3"]))
    story.append(Paragraph(guide.get("simple_explanation", "Not available"), styles["Normal"]))
    story.append(Spacer(1, 8))

    add_list("Common Symptoms", guide.get("common_symptoms", []))
    add_list("Safe Home Care", guide.get("safe_home_care", []))
    add_list("What to Avoid", guide.get("what_to_avoid", []))
    add_list("Foods to Eat", guide.get("foods_to_eat", []))
    add_list("Foods to Avoid", guide.get("foods_to_avoid", []))
    add_list("When to See a Doctor", guide.get("when_to_see_doctor", []))
    add_list("Urgent Warning Signs", guide.get("urgent_warning_signs", []))

    story.append(Paragraph("<b>AI Summary</b>", styles["Heading3"]))
    story.append(Paragraph(generate_ai_summary(disease, confidence, guide), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "⚠️ This report is AI-generated and intended only for educational and early screening support. "
        "It is not a substitute for professional medical diagnosis or treatment.",
        styles["Normal"]
    ))

    doc.build(story)
    return buffer.getvalue()

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return "DermAI Backend Running"

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["image"]
        img = preprocess_image(file)

        preds = get_model().predict(img, verbose=0)

        if isinstance(preds, list):
            preds = preds[0]

        preds = np.array(preds)

        if preds.ndim == 2:
            preds = preds[0]

        if len(preds) != len(CLASSES):
            raise ValueError(
                f"Model output has {len(preds)} classes but CLASSES has {len(CLASSES)}"
            )

        idx = int(np.argmax(preds))
        raw_disease = CLASSES[idx]
        display_disease = DISPLAY_NAMES[raw_disease]
        confidence = float(np.max(preds)) * 100
        guide = DISEASE_KNOWLEDGE[raw_disease]

        warning = None
        if confidence < 60:
            warning = "Prediction confidence is low. Please consult a dermatologist."

        ai_summary = generate_ai_summary(display_disease, confidence, guide)

        return jsonify({
            "disease": display_disease,
            "raw_disease": raw_disease,
            "confidence": confidence,
            "warning": warning,
            "care_guide": guide,
            "ai_summary": ai_summary
        })

    except Exception as e:
        print("PREDICT ERROR:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/save-prescription", methods=["POST"])
def save_prescription():
    try:
        data = request.get_json()

        required_fields = ["name", "age", "gender", "raw_disease", "confidence"]
        for field in required_fields:
            if field not in data or str(data[field]).strip() == "":
                return jsonify({"error": f"{field} is required."}), 400

        raw_disease = data["raw_disease"]
        if raw_disease not in DISEASE_KNOWLEDGE:
            return jsonify({"error": "Invalid disease value."}), 400

        display_disease = DISPLAY_NAMES[raw_disease]
        guide = DISEASE_KNOWLEDGE[raw_disease]

        pdf_bytes = generate_pdf(
            data["name"],
            data["age"],
            data["gender"],
            display_disease,
            float(data["confidence"]),
            guide
        )

        filename = f"{uuid.uuid4()}.pdf"
        path = os.path.join(tempfile.gettempdir(), filename)

        with open(path, "wb") as f:
            f.write(pdf_bytes)

        print("PDF SAVED AT:", path)

        base_url = os.getenv("BASE_URL", "https://akhil23bce-skindetectionbackend.hf.space").rstrip("/")
        pdf_url = f"{base_url}/download/{filename}"

        print("PDF URL:", pdf_url)

        return jsonify({
            "pdf_url": pdf_url,
            "message": "Advanced AI report generated successfully."
        })

    except Exception as e:
        print("SAVE ERROR:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/download/<filename>")
def download(filename):
    try:
        path = os.path.join(tempfile.gettempdir(), filename)
        print("DOWNLOAD REQUEST:", path)

        if not os.path.exists(path):
            print("FILE NOT FOUND:", path)
            return jsonify({"error": "File not found."}), 404

        return send_file(path, mimetype="application/pdf", as_attachment=True)

    except Exception as e:
        print("DOWNLOAD ERROR:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
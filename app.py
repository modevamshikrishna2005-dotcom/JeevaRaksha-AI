import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import re
from PIL import Image
from io import BytesIO
from gtts import gTTS
from deep_translator import GoogleTranslator
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(
    page_title="JeevaRaksha AI",
    page_icon="🚨",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f6f8fb; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 32px 34px; border-radius: 22px; background: linear-gradient(135deg,#111827 0%,#1f2937 55%,#374151 100%); color: white; margin-bottom: 18px; box-shadow: 0 12px 30px rgba(17,24,39,.12); }
.hero h1 { font-size: 44px; margin: 0 0 6px 0; font-weight: 800; }
.hero p { font-size: 18px; margin: 0; opacity: .9; }
.section-label { font-size: 13px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #6b7280; margin-top: 8px; }
.result-title { font-size: 27px; font-weight: 800; margin: 4px 0 12px 0; }
.alert-box { padding: 18px; border-radius: 16px; border: 1px solid #fecaca; background: #fff7f7; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px; }
button[kind="primary"] { min-height: 52px; font-size: 17px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

LANGUAGES = {
    "English": {"code": "en", "tts": "en"},
    "Telugu": {"code": "te", "tts": "te"},
    "Hindi": {"code": "hi", "tts": "hi"}
}

CLASSIFICATION_KEYWORDS = {
    "Accident / Injury": [
        "accident", "bike accident", "car accident", "road accident",
        "vehicle accident", "crash", "collision", "bike crash",
        "car crash", "fell from bike", "fell off bike",
        "యాక్సిడెంట్", "ప్రమాదం", "బైక్ ప్రమాదం", "కారు ప్రమాదం",
        "दुर्घटना", "सड़क दुर्घटना", "बाइक दुर्घटना", "कार दुर्घटना"
    ],
    "Breathing Emergency": [
        "breathing problem", "difficulty breathing", "cannot breathe",
        "can't breathe", "shortness of breath", "breathless",
        "శ్వాస తీసుకోవడం కష్టం", "ఊపిరి తీసుకోవడం కష్టం",
        "सांस लेने में दिक्कत", "सांस नहीं आ रही"
    ],
    "Unconsciousness": [
        "unconscious", "not responding", "passed out", "fainted",
        "స్పృహ కోల్పోయాడు", "స్పృహ లేదు",
        "बेहोश", "होश नहीं है"
    ],
    "Choking": [
        "choking", "cannot swallow", "something stuck in throat",
        "గొంతులో ఇరుక్కుంది", "ఉక్కిరిబిక్కిరి",
        "गले में फंस गया", "दम घुट रहा"
    ],
    "Fire": [
        "fire", "burning", "smoke", "flames",
        "మంట", "అగ్ని", "పొగ",
        "आग", "धुआं", "जल रहा"
    ],
    "Bleeding": [
        "bleeding", "blood", "bleeding badly", "heavy bleeding",
        "రక్తస్రావం", "రక్తం", "ఎక్కువ రక్తం",
        "खून", "खून बह रहा", "बहुत खून"
    ],
    "Minor Injury": [
        "small injury", "minor injury", "small cut", "minor cut",
        "చిన్న గాయం", "చిన్న కోత",
        "छोटी चोट", "हल्की चोट"
    ]
}

RISK_PATTERNS = [
    ("Heavy bleeding", [
        "heavy bleeding", "bleeding badly", "blood everywhere",
        "ఎక్కువ రక్తస్రావం", "చాలా రక్తం",
        "बहुत खून", "बहुत ज्यादा खून"
    ], 5),
    ("Bleeding", [
        "bleeding", "blood", "రక్తస్రావం", "రక్తం",
        "खून", "खून बह रहा"
    ], 3),
    ("Breathing difficulty", [
        "difficulty breathing", "cannot breathe", "can't breathe",
        "shortness of breath", "శ్వాస తీసుకోవడం కష్టం",
        "सांस लेने में दिक्कत"
    ], 5),
    ("Unconsciousness", [
        "unconscious", "not responding", "passed out",
        "స్పృహ లేదు", "स्पृहा नहीं",
        "बेहोश", "होश नहीं"
    ], 5),
    ("Severe pain", [
        "severe pain", "extreme pain", "very painful",
        "తీవ్రమైన నొప్పి", "చాలా నొప్పి",
        "बहुत तेज दर्द", "गंभीर दर्द"
    ], 3),
    ("Dizziness", [
        "dizzy", "dizziness", "feeling dizzy",
        "తల తిరుగుతోంది", "తల తిరగడం",
        "चक्कर", "चक्कर आ रहा"
    ], 2),
    ("Cannot stand", [
        "cannot stand", "can't stand", "unable to stand",
        "నిలబడలేకపోతున్నాను", "నిలబడలేను",
        "खड़ा नहीं हो सकता", "खड़े होने में दिक्कत"
    ], 3),
    ("Accident", [
        "accident", "crash", "collision", "ప్రమాదం", "యాక్సిడెంట్",
        "दुर्घटना", "हादसा"
    ], 3),
    ("Vehicle involvement", [
        "bike", "car", "vehicle", "motorcycle", "బైక్", "కారు",
        "వాహనం", "बाइक", "कार", "वाहन"
    ], 2),
    ("Fire / Smoke", [
        "fire", "flames", "smoke", "మంట", "అగ్ని", "పొగ",
        "आग", "धुआं"
    ], 5),
    ("Choking", [
        "choking", "గొంతులో ఇరుక్కుంది", "ఉక్కిరిబిక్కిరి",
        "दम घुट", "गले में फंस"
    ], 5),
    ("Severe burn", [
        "severe burn", "bad burn", "తీవ్రమైన కాలిన గాయం",
        "गंभीर जलन"
    ], 5),
    ("Seizure", [
        "seizure", "convulsion", "ఫిట్స్", "మూర్ఛ",
        "दौरा", "मिर्गी का दौरा"
    ], 6),
    ("Head injury", [
        "head injury", "hit my head", "head bleeding",
        "తలకు గాయం", "తలకి దెబ్బ",
        "सिर में चोट", "सिर पर चोट"
    ], 5),
    ("Chest pain", [
        "chest pain", "pain in chest",
        "ఛాతి నొప్పి", "గుండె దగ్గర నొప్పి",
        "सीने में दर्द"
    ], 5),
    ("Trapped person", [
        "trapped", "stuck inside", "cannot get out",
        "చిక్కుకున్నాను", "బయటకు రాలేకపోతున్నాను",
        "फंस गया", "बाहर नहीं निकल सकता"
    ], 5)
]

PRIORITY = [
    "Accident / Injury",
    "Breathing Emergency",
    "Unconsciousness",
    "Choking",
    "Fire",
    "Bleeding",
    "Minor Injury"
]

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

def translate_text(text, target):
    if not text:
        return ""
    if target == "en":
        return text
    try:
        return GoogleTranslator(
            source="auto",
            target=target
        ).translate(text)
    except Exception:
        return text

def translate_list(items, target):
    return [translate_text(x, target) for x in items]

def analyze_image(image):
    processor, model = load_blip()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )
    return caption

def get_visual_indicators(caption):
    text = caption.lower()
    indicators = []

    if any(x in text for x in ["car", "vehicle", "truck", "bus", "motorcycle", "bike"]):
        indicators.append("Vehicle visible")

    if any(x in text for x in ["damage", "damaged", "crash", "broken", "collision"]):
        indicators.append("Vehicle damage visible")

    if any(x in text for x in ["person", "man", "woman", "child", "people"]):
        indicators.append("Person visible")

    if any(x in text for x in ["fire", "flame", "smoke"]):
        indicators.append("Fire or smoke visible")

    if any(x in text for x in ["blood", "bleeding"]):
        indicators.append("Possible blood-related visual cue")

    return indicators

def classify_emergency(text):
    lower = text.lower()

    found = []

    for category, keywords in CLASSIFICATION_KEYWORDS.items():
        if any(keyword.lower() in lower for keyword in keywords):
            found.append(category)

    if found:
        for priority in PRIORITY:
            if priority in found:
                return priority

    return "General Emergency"

def calculate_risk(text):
    lower = text.lower()
    factors = []
    score = 0

    heavy_bleeding = any(
        x in lower for x in [
            "heavy bleeding",
            "bleeding badly",
            "ఎక్కువ రక్తస్రావం",
            "చాలా రక్తం",
            "बहुत खून",
            "बहुत ज्यादा खून"
        ]
    )

    for name, patterns, points in RISK_PATTERNS:
        if any(pattern.lower() in lower for pattern in patterns):
            if name == "Bleeding" and heavy_bleeding:
                continue
            factors.append((name, points))
            score += points

    if score >= 8:
        level = "CRITICAL"
    elif score >= 5:
        level = "HIGH"
    elif score >= 3:
        level = "MODERATE"
    else:
        level = "LOW"

    return score, level, factors

def multimodal_analysis(classification, visual_indicators):
    supporting = []
    limitations = []

    if classification == "Accident / Injury":
        if "Vehicle visible" in visual_indicators:
            supporting.append(
                "Vehicle-related visual evidence supports the accident classification."
            )
        if "Vehicle damage visible" in visual_indicators:
            supporting.append(
                "Visible vehicle damage supports the accident context."
            )
        if "Person visible" in visual_indicators:
            supporting.append(
                "A person is visible in the emergency scene."
            )

    elif classification == "Fire":
        if "Fire or smoke visible" in visual_indicators:
            supporting.append(
                "Visible fire or smoke supports the fire classification."
            )
        else:
            limitations.append(
                "The image does not provide clear visual confirmation of fire or smoke."
            )

    elif classification == "Bleeding":
        if "Person visible" in visual_indicators:
            supporting.append(
                "A person is visible, but the image cannot reliably confirm bleeding."
            )
        limitations.append(
            "Visual evidence cannot reliably determine bleeding severity."
        )

    elif classification in [
        "Breathing Emergency",
        "Choking",
        "Unconsciousness"
    ]:
        if "Person visible" in visual_indicators:
            supporting.append(
                "A person is visible in the emergency scene."
            )
        limitations.append(
            "The image cannot reliably confirm the person's medical condition."
        )

    if supporting and not limitations:
        level = "HIGH"
        agreement = 90
    elif supporting and limitations:
        level = "MODERATE"
        agreement = 65
    elif limitations:
        level = "LOW"
        agreement = 40
    else:
        level = "MODERATE"
        agreement = 55

    return level, agreement, supporting, limitations

def localized_response(classification, risk_level, target):
    responses = {
        "en": {
            "Accident / Injury":
                f"Situation: Accident / Injury. Risk level: {risk_level}. Seek emergency assistance immediately. If the situation may be life-threatening, contact your local emergency service.",
            "Breathing Emergency":
                f"Situation: Breathing Emergency. Risk level: {risk_level}. Seek emergency medical assistance immediately.",
            "Unconsciousness":
                f"Situation: Unconsciousness. Risk level: {risk_level}. Seek emergency assistance immediately.",
            "Choking":
                f"Situation: Choking. Risk level: {risk_level}. Seek emergency assistance immediately.",
            "Fire":
                f"Situation: Fire. Risk level: {risk_level}. Move to a safe location and contact emergency services.",
            "Bleeding":
                f"Situation: Bleeding. Risk level: {risk_level}. Seek emergency assistance immediately.",
            "Minor Injury":
                f"Situation: Minor Injury. Risk level: {risk_level}. Consider appropriate medical assistance if needed.",
            "General Emergency":
                f"Situation: General Emergency. Risk level: {risk_level}. Seek appropriate emergency assistance."
        }
    }

    response = responses["en"].get(
        classification,
        responses["en"]["General Emergency"]
    )

    return translate_text(response, target)

def create_summary(
    language,
    classification,
    risk_level,
    risk_score,
    location,
    description,
    image_caption,
    visual_indicators,
    factors,
    evidence_level,
    agreement,
    supporting,
    limitations,
    assessment,
    response
):
    target = LANGUAGES[language]["code"]

    factor_text = "\n".join(
        f"• {translate_text(name, target)} — {points}"
        for name, points in factors
    ) or "• None"

    indicator_text = ", ".join(
        translate_list(visual_indicators, target)
    ) or "None"

    detected_text = ", ".join(
        translate_list([x[0] for x in factors], target)
    ) or "None"

    support_text = " | ".join(
        translate_list(supporting, target)
    ) or "None"

    limitation_text = " | ".join(
        translate_list(limitations, target)
    ) or "None"

    return f"""JEEVARAKSHA AI — EMERGENCY SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Language:
{language}

Situation Classification:
{translate_text(classification, target)}

Risk Level:
{risk_level}

Risk Score:
{risk_score}

Location:
{location or "Not provided"}

Original Description:
{description or "Not provided"}

AI Visual Evidence:
{image_caption or "No image evidence provided"}

Visual Indicators:
{indicator_text}

Detected Indicators:
{detected_text}

Risk Factors:
{factor_text}

Multimodal Evidence Agreement:
{evidence_level} — {agreement}/100

Supporting Evidence:
{support_text}

Evidence Limitations:
{limitation_text}

Multimodal Assessment:
{translate_text(assessment, target)}

Recommended Response:
{response}

Note:
This AI system provides emergency communication support.
It is not a medical diagnosis and does not replace emergency services.
"""

def create_alert(
    language,
    classification,
    risk_level,
    risk_score,
    location,
    description,
    image_caption,
    visual_indicators,
    factors,
    evidence_level,
    agreement,
    supporting,
    limitations,
    assessment,
    response
):
    target = LANGUAGES[language]["code"]

    detected = "\n".join(
        f"• {translate_text(x[0], target)}"
        for x in factors
    ) or "• None"

    visuals = "\n".join(
        f"• {translate_text(x, target)}"
        for x in visual_indicators
    ) or "• None"

    supports = "\n".join(
        f"• {translate_text(x, target)}"
        for x in supporting
    ) or "• None"

    limits = "\n".join(
        f"• {translate_text(x, target)}"
        for x in limitations
    ) or "• None"

    return f"""🚨 JEEVARAKSHA AI — EMERGENCY ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk Level:
{risk_level}

Situation:
{translate_text(classification, target)}

Risk Score:
{risk_score}

📍 Location:
{location or "Not provided"}

⚠️ Detected Indicators:
{detected}

📝 Description:
{translate_text(description, target)}

🖼️ Visual Evidence:
{image_caption or "No image evidence provided"}

👁️ Visual Indicators:
{visuals}

🧠 Multimodal Evidence:
{evidence_level} — {agreement}/100

Supporting Evidence:
{supports}

Possible Limitations:
{limits}

Multimodal Assessment:
{translate_text(assessment, target)}

🛡️ Recommended Response:
{response}

🌐 Language:
{language}

This alert was generated by JeevaRaksha AI
for emergency communication support.

It is not a medical diagnosis and does not
replace emergency services.
"""

def speak_text(text, language):
    try:
        audio = BytesIO()
        tts = gTTS(
            text=text,
            lang=LANGUAGES[language]["tts"]
        )
        tts.write_to_fp(audio)
        audio.seek(0)
        return audio
    except Exception:
        return None

st.markdown("""
<div class="hero">
<h1>🚨 JeevaRaksha AI</h1>
<p>AI-Powered Emergency Communication & Assistance System</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ Safety Notice: JeevaRaksha AI provides emergency communication "
    "support and general first-response guidance. It does not diagnose "
    "medical conditions and does not replace emergency services or medical professionals."
)

with st.sidebar:
    st.header("⚙️ Settings")

    language = st.selectbox(
        "🌐 Emergency Language",
        list(LANGUAGES.keys())
    )

    st.markdown("### ♿ Accessibility")
    st.checkbox("🔊 Voice response", value=True)
    st.checkbox("📝 Show detailed explanation", value=True)

    st.markdown("---")
    st.info(
        "Supported languages:\n\n"
        "🇬🇧 English\n\n"
        "🇮🇳 Telugu\n\n"
        "🇮🇳 Hindi"
    )

st.markdown('<div class="section-label">Step 1 · Tell us what is happening</div>', unsafe_allow_html=True)
st.markdown("## 🚨 Emergency Input")

st.markdown("### ⚡ Quick Emergency")

quick_cols = st.columns(5)

quick_options = [
    ("🚗 Accident", "I have been involved in an accident and need emergency assistance."),
    ("🩸 Bleeding", "I am bleeding badly and need emergency assistance."),
    ("🫁 Breathing", "I am having difficulty breathing and need emergency assistance."),
    ("🔥 Fire", "There is a fire and I need emergency assistance."),
    ("😵 Unconscious", "A person is unconscious and needs emergency assistance.")
]

for i, (label, value) in enumerate(quick_options):
    if quick_cols[i].button(label, use_container_width=True):
        st.session_state["quick_text"] = value

default_text = st.session_state.get("quick_text", "")

emergency_text = st.text_area(
    "📝 Describe the emergency",
    value=default_text,
    height=130,
    placeholder="Example: I had a bike accident and my leg is bleeding."
)

st.markdown("### 🎤 Voice Input")
st.caption("Record a short description. Internet is required for speech recognition.")

voice_file = st.audio_input(
    "Record emergency description"
)

if voice_file is not None:
    st.audio(voice_file)

    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        audio_bytes = voice_file.read()

        with open("temp_voice.wav", "wb") as f:
            f.write(audio_bytes)

        with sr.AudioFile("temp_voice.wav") as source:
            audio_data = recognizer.record(source)

        detected_voice = recognizer.recognize_google(
            audio_data,
            language=LANGUAGES[language]["tts"] + "-IN"
            if language != "English"
            else "en-IN"
        )

        st.success("Voice converted to text successfully.")
        st.text_area(
            "🎤 Voice Text",
            value=detected_voice,
            height=100
        )

        emergency_text = detected_voice

    except Exception as e:
        st.error("Voice recognition could not process the recording.")

st.markdown("### 🖼️ Emergency Image")
st.caption("Optional: add a scene image for visual context. The AI does not diagnose medical conditions.")

uploaded_image = st.file_uploader(
    "Upload an emergency scene image",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Emergency Image",
        use_container_width=True
    )

    st.caption(
        "AI identifies visible scene information. "
        "It is not a medical diagnosis."
    )

st.markdown("### 📍 Emergency Location")
st.caption("Enter the location manually. JeevaRaksha AI does not automatically access precise location.")

location = st.text_input(
    "Location for Emergency Alert",
    placeholder="Example: KITSW, Warangal, Telangana"
)

st.markdown("---")

analyze_button = st.button(
    "🧠 ANALYZE EMERGENCY",
    type="primary",
    use_container_width=True
)

if analyze_button:

    if not emergency_text and image is None:
        st.error(
            "Please provide emergency text, voice input, or an image."
        )
        st.stop()

    with st.spinner("AI is analyzing the emergency..."):

        image_caption = ""

        if image is not None:
            try:
                image_caption = analyze_image(image)
            except Exception:
                image_caption = "Image analysis unavailable."

        classification_text = emergency_text or ""

        classification = classify_emergency(classification_text)

        risk_score, risk_level, factors = calculate_risk(
            classification_text
        )

        if classification == "Accident / Injury" and not any(
            x[0] == "Accident" for x in factors
        ):
            factors.append(("Accident", 3))
            risk_score += 3

        evidence_level, agreement, supporting, limitations = (
            multimodal_analysis(
                classification,
                get_visual_indicators(image_caption)
            )
        )

        visual_indicators = get_visual_indicators(image_caption)

        assessment = (
            "Text and image evidence show strong agreement. "
            "The available text and visual evidence are consistent "
            "with the detected emergency classification."
            if evidence_level == "HIGH"
            else
            "Text and image evidence provide partial support for "
            "the detected emergency classification."
            if evidence_level == "MODERATE"
            else
            "The available evidence has limitations and should not "
            "be treated as confirmation of the emergency condition."
        )

        response = localized_response(
            classification,
            risk_level,
            LANGUAGES[language]["code"]
        )

        st.session_state["result"] = {
            "language": language,
            "classification": classification,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "location": location,
            "description": emergency_text,
            "image_caption": image_caption,
            "visual_indicators": visual_indicators,
            "factors": factors,
            "evidence_level": evidence_level,
            "agreement": agreement,
            "supporting": supporting,
            "limitations": limitations,
            "assessment": assessment,
            "response": response
        }

if "result" in st.session_state:

    r = st.session_state["result"]

    st.markdown("---")
    st.markdown('<div class="section-label">Step 2 · AI assessment</div>', unsafe_allow_html=True)
    st.markdown("## 🧠 AI Emergency Analysis")

    if r["risk_level"] == "CRITICAL":
        st.error("🔴 CRITICAL PRIORITY")
    elif r["risk_level"] == "HIGH":
        st.error("🟠 HIGH PRIORITY")
    elif r["risk_level"] == "MODERATE":
        st.warning("🟡 MODERATE PRIORITY")
    else:
        st.success("🟢 LOW PRIORITY")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Risk Level", r["risk_level"])
    metric_cols[1].metric("Risk Score", r["risk_score"])
    metric_cols[2].metric("Evidence Agreement", f"{r['agreement']}/100")

    st.markdown(
        f'<div class="result-title">Situation: {r["classification"]}</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "AI classification is based on the provided emergency "
        "information. It is not a medical probability."
    )

    if r["description"]:
        st.markdown("### 📝 Text Evidence")
        st.info(r["description"])

    if r["image_caption"]:
        st.markdown("### 👁️ Visual Evidence")
        st.info(r["image_caption"])

    if r["visual_indicators"]:
        st.markdown("### 🔎 Visual Indicators")
        for item in r["visual_indicators"]:
            st.write("• " + item)

    left, right = st.columns(2)

    with left:
        st.markdown("### ⚠️ Detected Indicators")
        if r["factors"]:
            for name, points in r["factors"]:
                st.write("• " + name)
        else:
            st.write("• None")

    with right:
        st.markdown("### 📊 Risk Factors")
        if r["factors"]:
            for name, points in r["factors"]:
                st.write(f"• {name} — {points} points")
        else:
            st.write("• None")

    st.markdown("### 🧠 Multimodal Evidence Agreement")

    if r["evidence_level"] == "HIGH":
        st.success(
            f"🟢 STRONG EVIDENCE AGREEMENT — {r['agreement']}/100"
        )
    elif r["evidence_level"] == "MODERATE":
        st.warning(
            f"🟡 MODERATE EVIDENCE AGREEMENT — {r['agreement']}/100"
        )
    else:
        st.error(
            f"🔴 LIMITED EVIDENCE AGREEMENT — {r['agreement']}/100"
        )

    st.caption(
        "This score measures consistency between available text "
        "and visual evidence. It is not a medical probability."
    )

    if r["supporting"]:
        st.markdown("#### ✅ Supporting Evidence")
        for item in r["supporting"]:
            st.write("• " + item)

    if r["limitations"]:
        st.markdown("#### ⚠️ Evidence Limitations")
        for item in r["limitations"]:
            st.write("• " + item)

    st.markdown("### 🧠 Multimodal Assessment")
    st.info(r["assessment"])

    st.markdown("### 🛡️ Recommended Response")
    st.warning(r["response"])

    summary = create_summary(
        r["language"],
        r["classification"],
        r["risk_level"],
        r["risk_score"],
        r["location"],
        r["description"],
        r["image_caption"],
        r["visual_indicators"],
        r["factors"],
        r["evidence_level"],
        r["agreement"],
        r["supporting"],
        r["limitations"],
        r["assessment"],
        r["response"]
    )

    st.markdown("### 📋 Emergency Summary")
    st.code(summary, language="text")

    alert = create_alert(
        r["language"],
        r["classification"],
        r["risk_level"],
        r["risk_score"],
        r["location"],
        r["description"],
        r["image_caption"],
        r["visual_indicators"],
        r["factors"],
        r["evidence_level"],
        r["agreement"],
        r["supporting"],
        r["limitations"],
        r["assessment"],
        r["response"]
    )

    st.markdown('<div class="section-label">Step 3 · Share</div>', unsafe_allow_html=True)
    st.markdown("### 🚨 Emergency Alert Generator")

    st.info(
        "This generates a shareable emergency message. "
        "It does not automatically send the alert."
    )

    st.code(alert, language="text")
    st.caption("Copy this alert or download it and share it with a trusted person or emergency service.")

    alert_for_js = (
        alert
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )

    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{alert_for_js}`)"
        style="
        width:100%;
        padding:12px;
        border:none;
        border-radius:8px;
        background:#111827;
        color:white;
        font-size:16px;
        cursor:pointer;">
        📋 Copy Emergency Alert
        </button>
        """,
        height=55
    )

    st.download_button(
        "⬇️ Download Emergency Alert",
        data=alert,
        file_name="jeevaraksha_emergency_alert.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.markdown("### 🔊 AI Voice Response")
    st.caption("Optional spoken response generated in the selected language.")

    voice_response = speak_text(
        r["response"],
        r["language"]
    )

    if voice_response:
        st.audio(
            voice_response,
            format="audio/mp3"
        )
    else:
        st.warning(
            "Voice generation requires an internet connection."
        )

st.markdown("---")

st.markdown('<div class="section-label">Product overview</div>', unsafe_allow_html=True)
st.markdown("## ℹ️ About JeevaRaksha AI")

st.write(
    "JeevaRaksha AI supports emergency communication through "
    "text, voice and image inputs. It combines emergency "
    "classification, risk indicators, visual scene understanding, "
    "multimodal evidence analysis and multilingual communication "
    "to create a structured emergency summary and shareable alert."
)

st.info(
    "The system is intended as a communication-support tool. "
    "It does not diagnose medical conditions, guarantee AI "
    "predictions, automatically access precise location, or "
    "directly contact emergency services."
)

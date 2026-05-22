import os
from typing import Dict, Optional, Tuple

import numpy as np
import streamlit as st
import vertexai
from google.auth.exceptions import DefaultCredentialsError
from vertexai.generative_models import ChatSession, GenerativeModel


DEFAULT_MODEL = "gemini-1.5-pro"
WEIGHT_BUCKET_SIZE = 20
INTAKE_TRIGGER_HINT = (
    "To estimate mortality risk, type: `estimate mortality`."
)


def _get_defaults() -> Dict[str, str]:
    return {
        "project_id": os.getenv("VERTEX_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip(),
        "location": os.getenv("VERTEX_LOCATION", "us-central1").strip(),
        "model_name": os.getenv("VERTEX_MODEL", DEFAULT_MODEL).strip(),
    }


def _init_chat(project_id: str, location: str, model_name: str) -> ChatSession:
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name)
    return model.start_chat(response_validation=False)


def _render_sidebar(defaults: Dict[str, str]) -> Dict[str, str]:
    st.sidebar.header("Vertex AI Settings")
    st.sidebar.write("Set values below, or prefill them with environment variables.")

    project_id = st.sidebar.text_input(
        "Project ID",
        value=defaults["project_id"],
        help="GCP project ID with Vertex AI enabled.",
    ).strip()
    location = st.sidebar.text_input(
        "Location",
        value=defaults["location"],
        help="Vertex AI region, e.g. us-central1.",
    ).strip()
    model_name = st.sidebar.text_input(
        "Model",
        value=defaults["model_name"],
        help="Example: gemini-1.5-pro",
    ).strip()

    st.sidebar.code(
        "VERTEX_PROJECT_ID=<your-gcp-project-id>\n"
        "VERTEX_LOCATION=us-central1\n"
        f"VERTEX_MODEL={DEFAULT_MODEL}\n"
        "GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
        language="bash",
    )
    st.sidebar.caption(
        "Auth uses Application Default Credentials (ADC). "
        "You can set GOOGLE_APPLICATION_CREDENTIALS or run gcloud auth application-default login."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Mortality intake mode")
    st.sidebar.write(
        "When started, the chatbot asks for num_lab_procedures, number_diagnoses, num_procedures, num_medications, weight (kg), and readmission status, then returns an estimated mortality risk."
    )
    st.sidebar.caption(INTAKE_TRIGGER_HINT)
    return {"project_id": project_id, "location": location, "model_name": model_name}


def _start_intake_mode() -> str:
    st.session_state.intake_mode = True
    st.session_state.intake_step = "num_lab_procedures"
    st.session_state.intake_data = {}
    return (
        "Starting mortality intake.\n\n"
        "What is the patient's **num_lab_procedures**?"
    )


def _extract_numeric_value(text: str) -> Optional[float]:
    cleaned = text.replace(",", " ").strip()
    for token in cleaned.split():
        token = token.strip()
        try:
            return float(token)
        except ValueError:
            continue
    return None


def _extract_readmitted_value(text: str) -> Optional[int]:
    lowered = text.strip().lower()
    true_tokens = {"yes", "y", "true", "1", "<30", ">30", "readmitted"}
    false_tokens = {"no", "n", "false", "0", "not readmitted", "none"}
    if lowered in true_tokens:
        return 1
    if lowered in false_tokens:
        return 0
    return None


def _estimate_mortality_risk(
    num_lab_procedures: float,
    number_diagnoses: float,
    num_procedures: float,
    num_medications: float,
    weight_kg: float,
    readmitted_flag: int,
    readmitted_30d_flag: int,
) -> float:
    """Simple logistic estimate from key EDA-ranked factors."""
    weight_bucket = np.floor(weight_kg / WEIGHT_BUCKET_SIZE) * WEIGHT_BUCKET_SIZE
    logit = (
        -5.0
        + 0.022 * num_lab_procedures
        + 0.090 * number_diagnoses
        + 0.065 * num_procedures
        + 0.040 * num_medications
        + 0.008 * weight_bucket
        + 0.750 * readmitted_flag
        + 0.350 * readmitted_30d_flag
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(probability, 0.0, 1.0))


def _process_intake_answer(answer: str) -> Tuple[bool, str]:
    step = st.session_state.intake_step

    if step == "readmitted":
        readmitted_flag = _extract_readmitted_value(answer)
        if readmitted_flag is None:
            return False, "Please reply with yes or no for readmitted status."
        st.session_state.intake_data["readmitted_flag"] = readmitted_flag
        st.session_state.intake_step = "readmitted_30d"
        return False, 'Was it in the last 30 days? Answer "yes" or "no".'

    if step == "readmitted_30d":
        readmitted_30d_flag = _extract_readmitted_value(answer)
        if readmitted_30d_flag is None:
            return False, 'Please reply with "yes" or "no" for last 30 days.'
        st.session_state.intake_data["readmitted_30d_flag"] = readmitted_30d_flag

        num_lab_procedures = st.session_state.intake_data["num_lab_procedures"]
        number_diagnoses = st.session_state.intake_data["number_diagnoses"]
        num_procedures = st.session_state.intake_data["num_procedures"]
        num_meds = st.session_state.intake_data["num_medications"]
        weight_kg = st.session_state.intake_data["weight_kg"]
        readmitted_flag = st.session_state.intake_data["readmitted_flag"]
        risk = _estimate_mortality_risk(
            num_lab_procedures=num_lab_procedures,
            number_diagnoses=number_diagnoses,
            num_procedures=num_procedures,
            num_medications=num_meds,
            weight_kg=weight_kg,
            readmitted_flag=readmitted_flag,
            readmitted_30d_flag=readmitted_30d_flag,
        )
        st.session_state.intake_mode = False
        st.session_state.intake_step = None
        risk_pct = risk * 100.0

        return True, (
            f"Estimated mortality risk: **{risk_pct:.1f}%**\n\n"
            f"Inputs used: num_lab_procedures={num_lab_procedures:.0f}, "
            f"number_diagnoses={number_diagnoses:.0f}, "
            f"num_procedures={num_procedures:.0f}, "
            f"num_medications={num_meds:.0f}, "
            f"weight_kg={weight_kg:.1f}, "
            f"readmitted={bool(readmitted_flag)}, "
            f"readmitted_last_30_days={bool(readmitted_30d_flag)}\n\n"
            "_This is an educational estimate, not a medical diagnosis. Use clinical judgment and validated hospital models for care decisions._"
        )

    value = _extract_numeric_value(answer)
    if value is None:
        return False, "I couldn't find a number. Please enter a numeric value."

    if step == "num_lab_procedures":
        if value < 0 or value > 200:
            return False, "Please enter a valid num_lab_procedures value (0-200)."
        st.session_state.intake_data["num_lab_procedures"] = value
        st.session_state.intake_step = "number_diagnoses"
        return False, "What is the patient's **number_diagnoses**?"

    if step == "number_diagnoses":
        if value < 0 or value > 30:
            return False, "Please enter a valid number_diagnoses value (0-30)."
        st.session_state.intake_data["number_diagnoses"] = value
        st.session_state.intake_step = "num_procedures"
        return False, "What is the patient's **num_procedures**?"

    if step == "num_procedures":
        if value < 0 or value > 30:
            return False, "Please enter a valid num_procedures value (0-30)."
        st.session_state.intake_data["num_procedures"] = value
        st.session_state.intake_step = "num_medications"
        return False, "What is the patient's **num_medications**?"

    if step == "num_medications":
        if value < 0 or value > 200:
            return False, "Please enter a valid medication count (0-200)."
        st.session_state.intake_data["num_medications"] = value
        st.session_state.intake_step = "weight_kg"
        return False, "What is the patient's **weight** in kg?"

    if step == "weight_kg":
        if value <= 0 or value > 500:
            return False, "Please enter a valid weight in kg (1-500)."
        st.session_state.intake_data["weight_kg"] = value
        st.session_state.intake_step = "readmitted"
        return (
            False,
            "Was the patient **readmitted**? Reply with yes/no.",
        )

    st.session_state.intake_mode = False
    st.session_state.intake_step = None
    return False, "Intake reset. Type `estimate mortality` to start again."


def _is_intake_trigger(prompt: str) -> bool:
    lowered = prompt.lower()
    return "mortality" in lowered and (
        "estimate" in lowered or "predict" in lowered or "rate" in lowered
    )


def main() -> None:
    st.set_page_config(page_title="Vertex AI Chat Dashboard", layout="wide")
    st.title("Vertex AI Chat Dashboard")
    st.write("Chat with your Vertex AI model from this dashboard.")
    defaults = _get_defaults()
    settings = _render_sidebar(defaults)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
        st.session_state.chat_signature = None

    if "intake_mode" not in st.session_state:
        st.session_state.intake_mode = False
        st.session_state.intake_step = None
        st.session_state.intake_data = {}

    if not settings["project_id"]:
        st.warning("Enter a GCP Project ID in the sidebar to connect.")
        st.stop()

    if not settings["location"] or not settings["model_name"]:
        st.warning("Location and model are required.")
        st.stop()

    signature = (
        settings["project_id"],
        settings["location"],
        settings["model_name"],
    )
    if st.session_state.chat_session is None or st.session_state.chat_signature != signature:
        try:
            st.session_state.chat_session = _init_chat(
                project_id=settings["project_id"],
                location=settings["location"],
                model_name=settings["model_name"],
            )
            st.session_state.chat_signature = signature
            st.session_state.messages = []
        except DefaultCredentialsError:
            st.error(
                "Google Cloud credentials were not found. Set ADC with "
                "GOOGLE_APPLICATION_CREDENTIALS or run "
                "`gcloud auth application-default login`."
            )
            st.stop()
        except Exception as exc:
            st.error(f"Failed to initialize Vertex AI chat: {exc}")
            st.stop()

    st.caption(
        f"Connected to project `{settings['project_id']}` in `{settings['location']}` "
        f"using model `{settings['model_name']}`."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your ICU model or patient workflows...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.intake_mode:
        _, response_text = _process_intake_answer(prompt)
    elif _is_intake_trigger(prompt):
        response_text = _start_intake_mode()
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_session.send_message(
                        f"{prompt}\n\nIf user wants mortality estimation workflow, remind them: {INTAKE_TRIGGER_HINT}"
                    )
                    response_text = response.text
                except Exception as exc:
                    response_text = f"Vertex AI request failed: {exc}"
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        return

    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()

"""A local, neural language translator powered by Meta's NLLB-200 model."""

from __future__ import annotations

import logging
from threading import Lock

import httpx
import streamlit as st

MODEL_ID = "facebook/nllb-200-distilled-600M"
MAX_INPUT_TOKENS = 512

LANGUAGES: dict[str, str] = {
    "Arabic": "arb_Arab",
    "Bengali": "ben_Beng",
    "Chinese (Simplified)": "zho_Hans",
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Italian": "ita_Latn",
    "Japanese": "jpn_Jpan",
    "Kannada": "kan_Knda",
    "Korean": "kor_Hang",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Portuguese": "por_Latn",
    "Punjabi": "pan_Guru",
    "Russian": "rus_Cyrl",
    "Spanish": "spa_Latn",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
}

logger = logging.getLogger(__name__)


def validate_request(text: str, source_code: str, target_code: str) -> str:
    """Validate the user input and return its whitespace-trimmed form."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Enter some text before translating.")

    supported_codes = set(LANGUAGES.values())
    if source_code not in supported_codes:
        raise ValueError("The selected source language is not supported.")
    if target_code not in supported_codes:
        raise ValueError("The selected target language is not supported.")
    if source_code == target_code:
        raise ValueError("Choose two different languages to translate.")

    return cleaned_text


class NLLBTranslator:
    """Loads and runs the NLLB model locally on CPU or an available CUDA GPU."""

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Translation packages are missing. Install them with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            src_lang=LANGUAGES["English"],
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        self._model.to(self._device)
        self._model.eval()
        self._lock = Lock()

    def translate(self, text: str, source_code: str, target_code: str) -> str:
        """Translate text with the already-loaded model."""
        cleaned_text = validate_request(text, source_code, target_code)

        with self._lock:
            self._tokenizer.src_lang = source_code
            encoded = self._tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=False,
            )
            token_count = encoded["input_ids"].shape[-1]
            if token_count > MAX_INPUT_TOKENS:
                raise ValueError(
                    f"This text is {token_count} tokens long. The model supports "
                    f"up to {MAX_INPUT_TOKENS} input tokens per translation; "
                    "please shorten it and try again."
                )

            target_token_id = self._tokenizer.convert_tokens_to_ids(target_code)
            if target_token_id == self._tokenizer.unk_token_id:
                raise ValueError("The selected target language is not supported.")

            encoded = encoded.to(self._device)
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    forced_bos_token_id=target_token_id,
                    max_new_tokens=MAX_INPUT_TOKENS,
                )

            translated_text = self._tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0].strip()

        if not translated_text:
            raise RuntimeError("The translation model returned an empty result.")
        return translated_text


@st.cache_resource(show_spinner=False)
def get_translator() -> NLLBTranslator:
    """Create one shared model instance instead of reloading it on each rerun."""
    return NLLBTranslator()


def main() -> None:
    st.set_page_config(
        page_title="LingoLocal | AI Translator",
        page_icon="🌍",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        .block-container { max-width: 900px; padding-top: 2.5rem; }
        .hero { padding: 1.4rem 1.6rem; border-radius: 16px;
                background: linear-gradient(120deg, #102a43, #176b87);
                color: white; margin-bottom: 1.2rem; }
        .hero h1 { margin: 0; color: white; font-size: 2rem; }
        .hero p { margin: .45rem 0 0; color: #e3f4f7; }
        .caption-card { border: 1px solid rgba(128,128,128,.25);
                        border-radius: 12px; padding: .9rem 1rem; }
        </style>
        <div class="hero">
          <h1>LingoLocal</h1>
          <p>Private, free neural translation — processed on your computer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Powered by Meta's NLLB-200 distilled model. The model downloads once "
        "on first use; no translation API key or billing account is needed."
    )

    language_names = list(LANGUAGES)
    with st.form("translation_form"):
        source_column, target_column = st.columns(2)
        with source_column:
            source_name = st.selectbox(
                "Translate from",
                language_names,
                index=language_names.index("English"),
            )
        with target_column:
            target_name = st.selectbox(
                "Translate to",
                language_names,
                index=language_names.index("Hindi"),
            )

        source_text = st.text_area(
            "Text to translate",
            placeholder="Type or paste your text here...",
            height=180,
            max_chars=12000,
        )
        submitted = st.form_submit_button(
            "Translate",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            cleaned_text = validate_request(
                source_text,
                LANGUAGES[source_name],
                LANGUAGES[target_name],
            )
        except ValueError as exc:
            st.warning(str(exc))
            return

        try:
            with st.spinner(
                "Loading the translation model and translating..."
            ):
                translator = get_translator()
                translated_text = translator.translate(
                    cleaned_text,
                    LANGUAGES[source_name],
                    LANGUAGES[target_name],
                )
        except ValueError as exc:
            st.warning(str(exc))
            return
        except (
            ImportError,
            OSError,
            RuntimeError,
            TimeoutError,
            httpx.HTTPError,
        ) as exc:
            logger.exception("Translation failed")
            st.error(
                "Translation could not be completed. Check your internet "
                "connection for the initial model download, available memory, "
                "and the technical details below."
            )
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
            return

        st.markdown("### Translation")
        st.caption(f"{source_name} → {target_name}")
        st.code(translated_text, language=None, wrap_lines=True)
        st.caption(
            "Use the copy icon in the translation box to copy the result."
        )

    with st.expander("About privacy, limits, and model licensing"):
        st.markdown(
            """
            - Translation runs locally after the model has been downloaded.
              Your text is not sent to a translation API.
            - The model is several gigabytes on first download and supports
              inputs up to 512 model tokens per translation.
            - The NLLB-200 model is provided under **CC-BY-NC**. It is intended
              for non-commercial use; review the model terms before any other
              use. Machine translations can contain errors.
            """
        )


if __name__ == "__main__":
    main()

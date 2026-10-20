# LingoLocal — AI Language Translator

A professional Streamlit translation app for Windows 11 and Python 3.14. It
uses Meta's publicly downloadable pretrained **NLLB-200 distilled 600M** model
locally through Hugging Face Transformers. English, Hindi, Telugu, and 18 other
languages are included.

## Why this translation engine?

Google Cloud Translation and Azure AI Translator are managed, metered cloud
services; their pricing and account requirements do not provide a dependable
no-billing, no-card setup for this assignment. LibreTranslate is open source,
but its free public demo is not a guaranteed, unlimited hosted API and may
require an API key. This project therefore uses the pretrained model locally:

- **No paid API, API key, billing activation, or credit/debit card.**
- **Actual neural translation**, not sample or hardcoded output.
- After the initial model download, translation runs on your computer and
  submitted text is not sent to a translation service.
- The model supports Telugu (`tel_Telu`), Hindi (`hin_Deva`), English
  (`eng_Latn`), and many additional languages.

The trade-off is a multi-gigabyte first download and CPU-based translations can
take longer than a managed cloud API. A GPU is optional.

> **Model license:** The `facebook/nllb-200-distilled-600M` checkpoint is
> licensed **CC-BY-NC**. Its terms restrict commercial use. Review the model
> card and license before using it outside a non-commercial educational
> assignment. It is not intended for certified, medical, or legal translation.

## Project structure

```text
AI_Language_Translator/
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
└── tests/
    └── test_app.py
```

## Requirements

- Windows 11
- Python 3.14
- Internet access for installing packages and downloading the model the first
  time
- Several gigabytes of free disk space for the model and Python packages

## Install in PowerShell

Open PowerShell in this project folder:

```powershell
cd D:\Projects\AI\AI_Language_Translator
```

Create and activate a virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this terminal, use this command for the
current PowerShell process, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the CPU-only PyTorch build and the remaining dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The model is public; no Hugging Face account or token is required.

## Run

From the activated virtual environment and project folder:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`) and opens the
app in your browser. Select the source and target languages, enter text, and
click **Translate**. The first translation downloads the model checkpoint and
may take several minutes; later starts use the local model cache. Use the copy
icon in the result box to copy translated text.

## Supported languages

Arabic, Bengali, Chinese (Simplified), English, French, German, Gujarati,
Hindi, Italian, Japanese, Kannada, Korean, Malayalam, Marathi, Portuguese,
Punjabi, Russian, Spanish, Tamil, Telugu, and Urdu.

Inputs are limited to 512 model tokens. If a longer input is entered, the app
asks you to shorten it rather than silently dropping text.

## Test cases

Run the input-validation tests with:

```powershell
python -m unittest discover -s tests -v
```

The five representative test cases and expected results are:

| # | Input | Expected result |
|---|---|---|
| 1 | Blank or whitespace-only text | A clear “Enter some text” validation warning; no model download. |
| 2 | Unsupported target language code | A target-language validation error. |
| 3 | `Hello!` from English to Hindi | A Hindi translation is displayed (example from the model: `हैलो!`). |
| 4 | `आप कैसे हैं?` from Hindi to English | An English translation is displayed (example: `How are you?`). |
| 5 | `Hello, world!` from English to Telugu | A Telugu translation is displayed (example: `హలో, ప్రపంచం!`). |

The automated suite checks empty input, unsupported source and target codes,
identical source/target languages, and valid input handling. Translation
examples above were produced by the real local model; wording can vary with
model or library updates. The test suite does not fake model responses.

## Troubleshooting

- **Model download fails:** Check internet access and retry. The checkpoint is
  downloaded from Hugging Face on first use.
- **Out of memory:** Close other applications, use shorter input, and ensure
  there is enough free memory. The model runs on CPU by default unless CUDA is
  available.
- **PowerShell cannot find `py`:** Install Python 3.14 from
  [python.org](https://www.python.org/downloads/windows/) and enable the
  Python launcher.
- **Activation is blocked:** Use the process-scoped
  `Set-ExecutionPolicy` command above, or call
  `.\.venv\Scripts\python.exe -m pip ...` and
  `.\.venv\Scripts\python.exe -m streamlit run app.py` without activation.

## Model and API references

- [NLLB-200 model card and license](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [NLLB Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/nllb)
- [Google Cloud Translation pricing](https://cloud.google.com/translate/pricing)
- [LibreTranslate documentation](https://docs.libretranslate.com/)

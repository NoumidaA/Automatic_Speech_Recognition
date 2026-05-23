from __future__ import annotations
import os
import torch
import torchaudio
import gradio as gr
import spaces
from transformers import AutoModel, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from huggingface_hub import login

DESCRIPTION = "STT"
device = "cuda" if torch.cuda.is_available() else "cpu"

token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN is not set. Add it in Space Settings -> Variables and secrets.")

login(token=token)

# --- Model Loading ---
print("Loading ASR model (IndicConformer)...")
asr_model_id = "ai4bharat/indic-conformer-600m-multilingual"
asr_model = AutoModel.from_pretrained(
    asr_model_id,
    trust_remote_code=True,
    token=token
).to(device)
asr_model.eval()
print("ASR Model loaded.")

print("\nLoading Language ID model (MMS-LID-1024)...")
lid_model_id = "facebook/mms-lid-1024"
lid_processor = Wav2Vec2FeatureExtractor.from_pretrained(lid_model_id)
lid_model = AutoModelForAudioClassification.from_pretrained(lid_model_id).to(device)
lid_model.eval()
print("Language ID Model loaded.")


LID_TO_ASR_LANG_MAP = {
    "asm_Beng": "as", "ben_Beng": "bn", "brx_Deva": "br", "doi_Deva": "doi",
    "guj_Gujr": "gu", "hin_Deva": "hi", "kan_Knda": "kn", "kas_Arab": "ks",
    "kas_Deva": "ks", "gom_Deva": "kok", "mai_Deva": "mai", "mal_Mlym": "ml",
    "mni_Beng": "mni", "mar_Deva": "mr", "nep_Deva": "ne", "ory_Orya": "or",
    "pan_Guru": "pa", "san_Deva": "sa", "sat_Olck": "sat", "snd_Arab": "sd",
    "tam_Taml": "ta", "tel_Telu": "te", "urd_Arab": "ur",
    "asm": "as", "ben": "bn", "brx": "br", "doi": "doi", "guj": "gu", "hin": "hi",
    "kan": "kn", "kas": "ks", "gom": "kok", "mai": "mai", "mal": "ml", "mni": "mni",
    "mar": "mr", "npi": "ne", "ory": "or", "pan": "pa", "san": "sa", "sat": "sat",
    "snd": "sd", "tam": "ta", "tel": "te", "urd": "ur", "eng": "en"
}

ASR_CODE_TO_NAME = {
    "as": "Assamese", "bn": "Bengali", "br": "Bodo", "doi": "Dogri", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "ks": "Kashmiri", "kok": "Konkani", "mai": "Maithili",
    "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu", "en": "English"
}


def load_and_prepare_audio(audio_path: str):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.numel() == 0:
        raise ValueError("Loaded audio is empty.")

    # Ensure shape is [channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 2:
        raise ValueError(f"Unexpected audio tensor shape from torchaudio.load: {tuple(waveform.shape)}")

    # Convert to mono as required by IndicConformer examples/model usage
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz if needed
    target_sr = 16000
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Final shape check: must be [1, num_samples]
    if waveform.dim() != 2:
        raise ValueError(f"Audio tensor must be 2D after preprocessing, got {tuple(waveform.shape)}")

    if waveform.size(0) != 1:
        raise ValueError(f"Audio must be mono after preprocessing, got shape {tuple(waveform.shape)}")

    return waveform.contiguous(), target_sr


@spaces.GPU
def transcribe_audio_with_lid(audio_path):
    if not audio_path:
        return "Please provide an audio file.", "", ""

    try:
        waveform_16k, sr_16k = load_and_prepare_audio(audio_path)
    except Exception as e:
        return f"Error loading audio: {e}", "", ""

    try:
        # LID expects raw waveform array; pass 1D mono audio
        lid_audio = waveform_16k.squeeze(0).cpu().numpy()

        inputs = lid_processor(
            lid_audio,
            sampling_rate=sr_16k,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = lid_model(**inputs)

        logits = outputs.logits
        predicted_lid_id = logits.argmax(dim=-1).item()
        detected_lid_code = lid_model.config.id2label[predicted_lid_id]

        asr_lang_code = LID_TO_ASR_LANG_MAP.get(detected_lid_code)

        if not asr_lang_code:
            detected_lang_str = f"Detected '{detected_lid_code}', which is not supported by the ASR model."
            return detected_lang_str, "N/A", "N/A"

        detected_lang_str = f"Detected Language: {ASR_CODE_TO_NAME.get(asr_lang_code, 'Unknown')}"

        asr_input = waveform_16k.to(device)

        # Safety check before ASR
        if asr_input.dim() != 2 or asr_input.size(0) != 1:
            raise ValueError(f"ASR input must be [1, num_samples], got {tuple(asr_input.shape)}")

        with torch.no_grad():
            transcription_ctc = asr_model(asr_input, asr_lang_code, "ctc")
            transcription_rnnt = asr_model(asr_input, asr_lang_code, "rnnt")

    except Exception as e:
        return f"Error during processing: {str(e)}", "", ""

    return detected_lang_str, transcription_ctc.strip(), transcription_rnnt.strip()


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"## {DESCRIPTION}")
    gr.Markdown("Upload or record audio in any of the supported Indian languages. The app will detect the language and transcribe it.")

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(label="Upload or Record Audio", type="filepath")
            transcribe_btn = gr.Button("Transcribe", variant="primary")

        with gr.Column(scale=2):
            detected_lang_output = gr.Textbox(label="Language Detection Result")
            gr.Markdown("### CTC Transcription")
            ctc_output = gr.Textbox(lines=3, label="CTC Output")
            gr.Markdown("### RNNT Transcription")
            rnnt_output = gr.Textbox(lines=3, label="RNNT Output")

    transcribe_btn.click(
        fn=transcribe_audio_with_lid,
        inputs=[audio],
        outputs=[detected_lang_output, ctc_output, rnnt_output],
        api_name="transcribe"
    )

if __name__ == "__main__":
    demo.queue().launch()
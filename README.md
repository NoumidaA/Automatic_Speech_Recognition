# Multilingual Indic Speech Recognition

A multilingual speech-to-text system supporting 22 Indian languages using:

- Meta MMS-LID-1024 for automatic language identification
- AI4Bharat IndicConformer for multilingual speech transcription
- Gradio for the interactive web interface
- Hugging Face Spaces for deployment

This project enables users to upload or record speech audio and automatically receive:
- detected language output,
- CTC transcription,
- and RNNT transcription.

---

# Live Demo

## Hugging Face Space

https://huggingface.co/spaces/Noumida/ASR_New

---

# Project Overview

The system combines two deep learning pipelines:

1. Language Identification using MMS-LID-1024
2. Speech-to-Text Transcription using IndicConformer

The uploaded audio is first standardized into mono 16 kHz format before language detection is performed. The detected language code is then mapped to the corresponding IndicConformer language identifier for multilingual ASR inference.

The application supports both uploaded audio files and microphone recordings.

---

# Features

- Automatic language detection
- Support for 22 Indic languages
- Dual transcription outputs (CTC + RNNT)
- Upload or microphone recording support
- GPU-enabled inference
- Hugging Face deployment
- Real-time multilingual transcription workflow
- Audio preprocessing and validation pipeline

---

# Architecture

```text
          User Audio Input
      (Upload / Microphone)
                    │
                    ▼
         Audio Preprocessing
   (Mono Conversion + Resampling)
                    │
                    ▼
     MMS-LID-1024 Language Detection
                    │
                    ▼
     Language Code Mapping Layer
                    │
                    ▼
   IndicConformer Multilingual ASR
          ┌────────────────┐
          │     CTC        │
          │    RNNT        │
          └────────────────┘
                    │
                    ▼
         Final Transcription

# SPDX-License-Identifier: Apache-2.0
"""
Gradio UI for Higgs Audio v2

Run:
  python examples/serve_engine/app_gradio.py
Then open from another device on your LAN:
  http://<your-machine-ip>:7860
"""
from __future__ import annotations

import base64
import os
import sys
from io import BytesIO
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from transformers import pipeline

# Ensure project root (parent of this file's directory) is on sys.path so
# `boson_multimodal` can be imported when running this script directly
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent


# --------- Global initialization (single load) ---------
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

_device = "cuda" if torch.cuda.is_available() else "cpu"
serve_engine: Optional[HiggsAudioServeEngine] = None
asr_pipe = None


def _lazy_init() -> HiggsAudioServeEngine:
    global serve_engine
    if serve_engine is None:
        serve_engine = HiggsAudioServeEngine(
            model_name_or_path=MODEL_PATH,
            audio_tokenizer_name_or_path=AUDIO_TOKENIZER_PATH,
            device=_device,
        )
    return serve_engine


# --------- Helpers ---------

def _build_system_prompt(scene_desc: str) -> str:
    return (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        f"{scene_desc}\n"
        "<|scene_desc_end|>"
    )


def _run_generate(chat_ml: ChatMLSample, *, temperature: float, top_p: float, top_k: int, max_new_tokens: int) -> Tuple[int, np.ndarray]:
    engine = _lazy_init()
    resp = engine.generate(
        chat_ml_sample=chat_ml,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_strings=["<|end_of_text|>", "<|eot_id|>", "<|audio_eos|>"],
    )
    if resp.audio is None:
        raise gr.Error("No audio was generated. Try increasing max_new_tokens or adjusting parameters.")
    return resp.sampling_rate, resp.audio.astype(np.float32)


# --------- Tab handlers ---------

def tts_smart(transcript: str, scene_desc: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int):
    if not transcript.strip():
        raise gr.Error("Please enter a transcript.")
    system_prompt = _build_system_prompt(scene_desc.strip() or "Audio is recorded from a quiet room.")
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=transcript.strip()),
    ]
    chat_ml = ChatMLSample(messages=messages)
    return _run_generate(chat_ml, temperature=temperature, top_p=top_p, top_k=int(top_k), max_new_tokens=int(max_new_tokens))


def tts_clone(reference_audio, reference_text: str, target_transcript: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int):
    if reference_audio is None:
        raise gr.Error("Please upload a reference audio file.")
    if not reference_text.strip():
        raise gr.Error("Please provide the reference audio's transcript.")
    if not target_transcript.strip():
        raise gr.Error("Please provide a target transcript to generate.")

    # Gradio file component gives a dict or tempfile path depending on version
    if isinstance(reference_audio, dict) and "path" in reference_audio:
        file_path = reference_audio["path"]
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(reference_audio, str):
        with open(reference_audio, "rb") as f:
            raw_bytes = f.read()
    elif hasattr(reference_audio, "read"):
        raw_bytes = reference_audio.read()
    else:
        raise gr.Error("Unsupported file input.")

    b64_audio = base64.b64encode(raw_bytes).decode("utf-8")

    messages = [
        Message(role="user", content=reference_text.strip()),
        Message(role="assistant", content=AudioContent(raw_audio=b64_audio, audio_url="placeholder")),
        Message(role="user", content=target_transcript.strip()),
    ]
    chat_ml = ChatMLSample(messages=messages)
    return _run_generate(chat_ml, temperature=temperature, top_p=top_p, top_k=int(top_k), max_new_tokens=int(max_new_tokens))


essay_multi_speaker_default = (
    "[SPEAKER0] Hey! Have you checked the results from the latest experiment?\n"
    "[SPEAKER1] Not yet. Anything interesting?\n"
    "[SPEAKER0] Quite a bit. The model handled the edge cases better than expected."
)


def tts_multispeaker(transcript: str, scene_desc: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int):
    if not transcript.strip():
        raise gr.Error("Please enter a multi-speaker transcript.")

    system_prompt = (
        "You are an AI assistant designed to convert text into speech.\n"
        "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
        "If no speaker tag is present, select a suitable voice on your own.\n\n"
        "<|scene_desc_start|>\n"
        f"{scene_desc.strip() or 'SPEAKER0: feminine\nSPEAKER1: masculine'}\n"
        "<|scene_desc_end|>"
    )

    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=transcript.strip()),
    ]
    chat_ml = ChatMLSample(messages=messages)
    return _run_generate(chat_ml, temperature=temperature, top_p=top_p, top_k=int(top_k), max_new_tokens=int(max_new_tokens))


# --------- ASR (Transcription) ---------
def _lazy_init_asr():
    global asr_pipe
    if asr_pipe is None:
        device_index = 0 if torch.cuda.is_available() else -1
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=device_index,
        )
    return asr_pipe


def transcribe_audio(audio_file, language: str):
    if audio_file is None:
        raise gr.Error("Please upload an audio file to transcribe.")
    pipe = _lazy_init_asr()

    # Resolve path or file-like to bytes
    if isinstance(audio_file, dict) and "path" in audio_file:
        path = audio_file["path"]
    elif isinstance(audio_file, str):
        path = audio_file
    elif hasattr(audio_file, "read"):
        # Save to buffer and decode with soundfile
        raw_bytes = audio_file.read()
        buf = BytesIO(raw_bytes)
        path = None
    else:
        raise gr.Error("Unsupported file input.")

    # Load with soundfile (or librosa fallback) to avoid ffmpeg dependency
    audio = None
    sr = None
    try:
        import soundfile as sf
        if path is not None:
            audio, sr = sf.read(path, always_2d=False)
        else:
            audio, sr = sf.read(buf, always_2d=False)
    except Exception:
        # Fallback to librosa
        import librosa
        if path is not None:
            audio, sr = librosa.load(path, sr=None, mono=False)
        else:
            # librosa cannot load from BytesIO without soundfile backend; re-raise
            raise gr.Error("Failed to read audio. Please install 'soundfile' or upload a standard WAV/MP3/FLAC file.")

    # Ensure mono float32
    if audio is None or sr is None:
        raise gr.Error("Could not decode audio file.")
    if audio.ndim > 1:
        # Mix down to mono
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32)

    inputs = {"array": audio, "sampling_rate": int(sr)}
    generate_kwargs = {"language": language} if language else {}
    # If audio is longer than 30s, enable chunking and timestamps to satisfy Whisper requirements
    duration_s = float(len(audio)) / float(sr)
    call_kwargs = {}
    if duration_s > 30.0:
        call_kwargs.update({
            "chunk_length_s": 30,
            "stride_length_s": 5,
            "return_timestamps": True,
        })
    result = pipe(inputs, generate_kwargs=generate_kwargs, **call_kwargs)
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    return text.strip()


# --------- UI (Blocks) ---------
with gr.Blocks(theme=gr.themes.Soft(), title="Higgs Audio v2") as demo:
    gr.Markdown("""
    # Higgs Audio v2 UI
    - Smart Voice TTS, Voice Cloning, and Multi-speaker generation
    - Device: {}  
    """.format(_device))

    with gr.Tab("Smart Voice"):
        with gr.Row():
            transcript = gr.Textbox(label="Transcript", lines=5, placeholder="Type your text to synthesize...")
        with gr.Row():
            scene_desc = gr.Textbox(
                label="Scene description",
                lines=3,
                value="Audio is recorded from a quiet room.",
            )
        with gr.Accordion("Advanced", open=False):
            temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
            top_k = gr.Slider(0, 100, value=50, step=1, label="top_k")
            max_new_tokens = gr.Slider(64, 2048, value=1024, step=64, label="max_new_tokens")
        btn_smart = gr.Button("Generate", variant="primary")
        audio_out_smart = gr.Audio(label="Output", type="numpy")
        btn_smart.click(
            tts_smart,
            inputs=[transcript, scene_desc, temperature, top_p, top_k, max_new_tokens],
            outputs=[audio_out_smart],
        )

    with gr.Tab("Voice Clone"):
        with gr.Row():
            ref_audio = gr.File(label="Reference audio (wav/mp3)")
        with gr.Row():
            ref_text = gr.Textbox(label="Reference text (what the reference audio says)", lines=2)
        with gr.Row():
            target_text = gr.Textbox(label="Target transcript", lines=5)
        with gr.Accordion("Advanced", open=False):
            temperature_c = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="temperature")
            top_p_c = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
            top_k_c = gr.Slider(0, 100, value=50, step=1, label="top_k")
            max_new_tokens_c = gr.Slider(64, 2048, value=1024, step=64, label="max_new_tokens")
        btn_clone = gr.Button("Generate", variant="primary")
        audio_out_clone = gr.Audio(label="Output", type="numpy")
        btn_clone.click(
            tts_clone,
            inputs=[ref_audio, ref_text, target_text, temperature_c, top_p_c, top_k_c, max_new_tokens_c],
            outputs=[audio_out_clone],
        )

    with gr.Tab("Multi-speaker"):
        with gr.Row():
            transcript_ms = gr.Textbox(label="Transcript (use [SPEAKER*] tags optionally)", lines=6, value=essay_multi_speaker_default)
        with gr.Row():
            scene_desc_ms = gr.Textbox(label="Scene description (speaker styles)", lines=3, value="SPEAKER0: feminine\nSPEAKER1: masculine")
        with gr.Accordion("Advanced", open=False):
            temperature_m = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
            top_p_m = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
            top_k_m = gr.Slider(0, 100, value=50, step=1, label="top_k")
            max_new_tokens_m = gr.Slider(64, 4096, value=1536, step=64, label="max_new_tokens")
        btn_ms = gr.Button("Generate", variant="primary")
        audio_out_ms = gr.Audio(label="Output", type="numpy")
        btn_ms.click(
            tts_multispeaker,
            inputs=[transcript_ms, scene_desc_ms, temperature_m, top_p_m, top_k_m, max_new_tokens_m],
            outputs=[audio_out_ms],
        )

    with gr.Tab("Transcription"):
        gr.Markdown("Upload an audio file and transcribe using Whisper (transformers pipeline).")
        asr_file = gr.File(label="Audio file (wav/mp3/flac)")
        asr_lang = gr.Textbox(label="Language hint (e.g., en, fr, zh)", placeholder="Optional")
        btn_asr = gr.Button("Transcribe", variant="primary")
        asr_text = gr.Textbox(label="Transcription", lines=8)
        btn_asr.click(transcribe_audio, inputs=[asr_file, asr_lang], outputs=[asr_text])


if __name__ == "__main__":
    # Bind to 0.0.0.0 to allow LAN access
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

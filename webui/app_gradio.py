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
import time
import asyncio
import threading
import queue as pyqueue
import subprocess
from typing import Optional, Tuple
import json
import urllib.request
import urllib.error
import re

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
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern


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


def _clear_text() -> str:
    """Utility for Gradio .then() to clear a Textbox value."""
    return ""


def _create_cancel_event():
    """Create a new cancellation token for a streaming session."""
    return threading.Event()


def _cancel_event(evt: Optional[threading.Event]):
    """Set the cancellation token if present (wired to Stop button)."""
    if isinstance(evt, threading.Event):
        evt.set()
    # No UI outputs to update here; returning None keeps Gradio happy
    return None


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


# --------- Mock Chat Streaming (UI-first) ---------
def chat_stream_mock(
    user_text: str,
    llm_provider: str,
    voice_mode: str,
    chunk_ms: int,
    ollama_model: str = "",
):
    """Simulate a streaming TTS session.

    Yields progressively longer audio (sine wave) and transcript text.
    Keeps the UI interactive and demonstrates streaming UX before real wiring.
    """
    if not user_text.strip():
        raise gr.Error("Please enter a prompt to chat.")

    # Sine-wave parameters
    sr = 24000
    total_s = 6.0  # total mock audio duration
    freq = 440.0 if voice_mode == "Default" else (220.0 if voice_mode == "Warm" else 660.0)
    samples_total = int(total_s * sr)
    t = np.arange(samples_total) / sr
    full_wave = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Chunking
    step = max(40, int(chunk_ms))  # ms
    step_samples = int(sr * (step / 1000.0))

    # Progressive buffer and transcript
    acc = np.zeros(0, dtype=np.float32)
    transcript_chunks = []
    words = (user_text.strip() + " ").split()
    words_per_chunk = max(1, len(words) // max(1, samples_total // max(1, step_samples)))

    # Initial yield: tiny epsilon to avoid zero-length and zero-max normalization warnings
    yield (sr, np.array([1e-6], dtype=np.float32)), ""

    pos = 0
    wpos = 0
    while pos < samples_total:
        nxt = min(samples_total, pos + step_samples)
        acc = np.concatenate([acc, full_wave[pos:nxt]])

        # Simulate transcript trickle
        if wpos < len(words):
            take = min(words_per_chunk, len(words) - wpos)
            transcript_chunks.extend(words[wpos:wpos + take])
            wpos += take
        transcript = " ".join(transcript_chunks)

        # Yield only the new chunk to avoid restarting playback
        chunk = full_wave[pos:nxt]
        yield (sr, chunk), transcript
        time.sleep(step / 1000.0)
        pos = nxt

    # Final yield: only update transcript to avoid replacing audio buffer
    yield gr.update(), " ".join(words)


def _build_chatml_for_text(user_text: str) -> ChatMLSample:
    # Strong directive to read the given text verbatim to avoid paraphrasing
    system_prompt = (
        "You are a TTS engine. Read the following text VERBATIM, without adding or changing words.\n\n"
        + _build_system_prompt("Audio is recorded from a quiet room.")
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=f"Read exactly this: {user_text.strip()}"),
    ]
    return ChatMLSample(messages=messages)


def chat_stream_tts_realtime(
    user_text: str,
    llm_provider: str,
    voice_mode: str,
    chunk_ms: int,
    ollama_model: str = "",
    cancel_evt: Optional[threading.Event] = None,
):
    """Real-time streaming using HiggsAudio generate_delta_stream.

    For now, bypasses LLM and synthesizes the user's text directly for low-latency demo.
    """
    if not user_text.strip():
        raise gr.Error("Please enter a prompt to chat.")

    engine = _lazy_init()
    print(f"[DirectTTS] Synthesizing user text (len={len(user_text.strip())}): {user_text.strip()[:160]!r}")
    chat_ml = _build_chatml_for_text(user_text)

    # Threaded async producer -> queue of deltas
    q: pyqueue.Queue = pyqueue.Queue(maxsize=128)
    stop_evt = cancel_evt or threading.Event()

    async def producer():
        try:
            async for delta in engine.generate_delta_stream(
                chat_ml_sample=chat_ml,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>", "<|audio_eos|>"],
            ):
                if stop_evt.is_set():
                    break
                q.put(delta)
        finally:
            q.put(None)

    def run_loop():
        asyncio.run(producer())

    th = threading.Thread(target=run_loop, daemon=True)
    th.start()

    # Progressive PCM buffer
    sr = engine.audio_tokenizer.sampling_rate
    acc_pcm = np.zeros(0, dtype=np.float32)
    # Buffer for batching sentence-aware segments before emitting
    emit_buf_sa = np.array([], dtype=np.float32)
    last_decoded_len = 0

    # Buffer of audio tokens [codebooks, frames]
    token_buf = None  # torch.Tensor or None

    # Initial yield: tiny epsilon to avoid zero-length and normalization warnings
    yield (sr, np.array([1e-6], dtype=np.float32)), ""

    transcript_acc = ""
    while True:
        # Check for cancellation promptly
        if stop_evt.is_set():
            break
        try:
            item = q.get(timeout=0.05)
        except pyqueue.Empty:
            continue
        if item is None:
            break

        # Collect audio tokens
        if item.audio_tokens is not None:
            at = item.audio_tokens
            # Normalize to 2D [codebooks, frames]
            if at.ndim == 1:
                at = at.unsqueeze(0)
            if at.ndim != 2:
                at = None
            if at is not None:
                if token_buf is None or token_buf.ndim != 2 or token_buf.shape[0] != at.shape[0]:
                    token_buf = at
                else:
                    token_buf = torch.cat([token_buf, at], dim=1)

            # Decode when we have a sufficient buffer (~chunk_ms)
            frames_per_ms = engine.audio_tokenizer.tps / 1000.0
            min_frames = int(max(1, chunk_ms * frames_per_ms))
            if token_buf is not None and token_buf.ndim == 2 and token_buf.shape[1] >= min_frames:
                try:
                    # Prepare vq code: handle delay pattern; keep BOS removal, EOS may not exist yet
                    vq_code = revert_delay_pattern(token_buf).clip(0, engine.audio_codebook_size - 1)
                    if vq_code.shape[1] > 1:
                        vq_code = vq_code[:, 1:]
                    # Decode whole buffer, then append only new tail
                    pcm = engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    if pcm.shape[0] > last_decoded_len:
                        new_tail = pcm[last_decoded_len:]
                        if new_tail.size > 0:
                            acc_pcm = np.concatenate([acc_pcm, new_tail.astype(np.float32)])
                            last_decoded_len = pcm.shape[0]
                            if stop_evt.is_set():
                                break
                            # Yield only the new tail chunk to avoid restarting playback
                            yield (sr, new_tail.astype(np.float32)), (transcript_acc or "")
                except Exception:
                    # If decode fails on partial tokens, continue accumulating
                    pass

        # Stream text tokens if present
        if getattr(item, "text", None):
            transcript_acc += item.text
            # Text-only update: do not touch audio
            if stop_evt.is_set():
                break
            yield gr.update(), transcript_acc

    # Final flush: one more decode attempt
    if token_buf is not None:
        try:
            vq_code = revert_delay_pattern(token_buf).clip(0, engine.audio_codebook_size - 1)
            if vq_code.shape[1] > 1:
                vq_code = vq_code[:, 1:]
            pcm = engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
            if pcm.shape[0] > last_decoded_len:
                new_tail = pcm[last_decoded_len:]
                if new_tail.size > 0:
                    acc_pcm = np.concatenate([acc_pcm, new_tail.astype(np.float32)])
                    last_decoded_len = pcm.shape[0]
        except Exception:
            pass

    if not stop_evt.is_set():
        # Final text update only; leave audio as-is to avoid restart
        yield gr.update(), (transcript_acc or user_text)


def chat_stream_entry(
    user_text: str,
    llm_provider: str,
    voice_mode: str,
    chunk_ms: int,
    ollama_model: str = "",
    seg_min_chars: int = 80,
    seg_max_chars: int = 200,
    seg_overlap_ms: int = 30,
    full_response_tts: bool = False,
    sentence_aware_stream: bool = False,
    cancel_evt: Optional[threading.Event] = None,
    chat_history: Optional[list] = None,
    clone_enabled: bool = False,
    clone_ref_audio=None,
    clone_ref_text: str = "",
):
    if llm_provider == "Mock":
        yield from chat_stream_mock(user_text, llm_provider, voice_mode, chunk_ms, ollama_model)
    elif llm_provider == "Direct TTS":
        yield from chat_stream_tts_realtime(user_text, llm_provider, voice_mode, chunk_ms, ollama_model, cancel_evt)
    elif llm_provider == "Ollama":
        yield from chat_stream_ollama_incremental_tts(
            user_text,
            ollama_model,
            int(seg_min_chars),
            int(seg_max_chars),
            int(seg_overlap_ms),
            int(chunk_ms),
            bool(full_response_tts),
            bool(sentence_aware_stream),
            cancel_evt,
            chat_history,
            clone_enabled,
            clone_ref_audio,
            clone_ref_text,
        )
    else:
        # Placeholder before LLM integration
        yield from chat_stream_mock(user_text, llm_provider, voice_mode, chunk_ms, ollama_model)


# --------- Ollama helpers ---------
def list_ollama_models() -> list[str]:
    """Return installed Ollama model names by calling `ollama list`.
    Falls back to an empty list on failure.
    """
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        # Skip header if present (starts with NAME)
        names = []
        for ln in lines:
            if ln.lower().startswith("name"):
                continue
            # First column is model name (split by whitespace)
            parts = ln.split()
            if parts:
                names.append(parts[0])
        # Deduplicate, preserve order
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out
    except Exception:
        return []


def _ollama_stream_chat(model: str, prompt: str, cancel_evt: Optional[threading.Event] = None, messages: Optional[list] = None):
    """Stream assistant tokens from Ollama's /api/chat endpoint.
    Yields text chunks (strings). Requires local Ollama at http://localhost:11434.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": messages if isinstance(messages, list) and len(messages) > 0 else [{"role": "user", "content": prompt}],
        "stream": True,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                try:
                    if cancel_evt is not None and cancel_evt.is_set():
                        break
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("done"):
                        break
                    msg = obj.get("message", {})
                    chunk = msg.get("content", "")
                    if chunk:
                        yield chunk
                except Exception:
                    continue
    except urllib.error.URLError as e:
        raise gr.Error(f"Failed to reach Ollama at localhost:11434: {e}")


def chat_stream_ollama_tts(user_text: str, ollama_model: str):
    """Stream transcript from Ollama, then synthesize final audio with Higgs.

    For now, audio is generated once at the end (non-streaming) to keep the flow simple and working.
    """
    if not user_text.strip():
        raise gr.Error("Please enter a prompt to chat.")
    if not ollama_model or ollama_model.startswith("<no "):
        raise gr.Error("Please select a valid Ollama model.")

    # Initial tiny epsilon to prime the Audio component
    yield (24000, np.array([1e-6], dtype=np.float32)), ""

    # Stream assistant text as it arrives
    transcript_acc = ""
    for chunk in _ollama_stream_chat(ollama_model, user_text.strip()):
        transcript_acc += chunk
        yield (24000, np.array([1e-6], dtype=np.float32)), transcript_acc

    # Synthesize full assistant message to audio
    if transcript_acc.strip():
        system_prompt = _build_system_prompt("Audio is recorded from a quiet room.")
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=transcript_acc.strip()),
        ]
        chat_ml = ChatMLSample(messages=messages)
        sr, audio = _run_generate(chat_ml, temperature=0.5, top_p=0.95, top_k=50, max_new_tokens=1536)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        yield (sr, audio), transcript_acc
    else:
        yield (24000, np.array([1e-6], dtype=np.float32)), "(No response from Ollama)"


def _crossfade_concat(acc: np.ndarray, new: np.ndarray, sr: int, overlap_ms: int) -> np.ndarray:
    """Linearly crossfade 'new' into the end of 'acc' over overlap_ms."""
    if acc.size == 0 or overlap_ms <= 0:
        return np.concatenate([acc, new]) if acc.size > 0 else new
    overlap = int(sr * (overlap_ms / 1000.0))
    if overlap <= 0 or acc.size < overlap or new.size < overlap:
        return np.concatenate([acc, new])
    fade_out = np.linspace(1.0, 0.0, num=overlap, dtype=np.float32)
    fade_in = 1.0 - fade_out
    mixed_tail = acc[-overlap:] * fade_out + new[:overlap] * fade_in
    return np.concatenate([acc[:-overlap], mixed_tail, new[overlap:]]).astype(np.float32)


def _pop_segment(buf: str, min_chars: int, max_chars: int) -> tuple[str, str]:
    """Return (segment, remainder) if a segment boundary is found, else ("", buf)."""
    if len(buf) < min_chars:
        return "", buf
    # Prefer punctuation boundary
    m = None
    for punct in [".", "!", "?", "。", "！", "？", ";", ":", ","]:
        idx = buf.rfind(punct, 0, max(min(len(buf), max_chars) + 1, 0))
        if idx != -1 and idx + 1 >= min_chars:
            m = idx + 1
            break
    if m is None and len(buf) >= max_chars:
        m = max_chars
    if m is None:
        return "", buf
    seg = buf[:m].strip()
    rem = buf[m:]
    return seg, rem


_SENTENCE_RE = re.compile(r"(.+?[\.\!\?\u2026]+)(\s+|$)")


def _pop_sentence(buf: str, min_len: int = 12) -> tuple[str, str]:
    """
    Extract the next complete sentence from buf.
    Returns (sentence, remainder) or ("", buf) if no complete sentence yet.
    A sentence ends with ., !, ?, or … (unicode ellipsis). Requires a minimal length to avoid tiny fragments.
    """
    if len(buf.strip()) < min_len:
        return "", buf
    m = _SENTENCE_RE.match(buf)
    if not m:
        return "", buf
    sent = m.group(1).strip()
    rem = buf[m.end():]
    if len(sent) < min_len:
        return "", buf
    return sent, rem


def _extract_audio_b64(reference_audio) -> str:
    """Extract base64 from a Gradio File-like input."""
    if reference_audio is None:
        return ""
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
        return ""
    return base64.b64encode(raw_bytes).decode("utf-8")


def chat_stream_ollama_incremental_tts(
    user_text: str,
    ollama_model: str,
    seg_min_chars: int,
    seg_max_chars: int,
    seg_overlap_ms: int,
    chunk_ms: int,
    full_response_tts: bool = False,
    sentence_aware_stream: bool = False,
    cancel_evt: Optional[threading.Event] = None,
    chat_history: Optional[list] = None,
    clone_enabled: bool = False,
    clone_ref_audio=None,
    clone_ref_text: str = "",
):
    """Stream text from Ollama and synthesize audio per segment incrementally."""
    if not user_text.strip():
        raise gr.Error("Please enter a prompt to chat.")
    if not ollama_model or ollama_model.startswith("<no "):
        raise gr.Error("Please select a valid Ollama model.")

    engine = _lazy_init()
    sr = engine.audio_tokenizer.sampling_rate

    # Prime Audio
    yield (sr, np.array([1e-6], dtype=np.float32)), ""

    transcript_acc = ""
    buf = ""
    acc_pcm = np.zeros(0, dtype=np.float32)
    # Buffer for batching sentence-aware audio before yielding to client
    emit_buf_sa = np.array([], dtype=np.float32)

    def _pcm_to_b64(arr: np.ndarray) -> str:
        x = np.asarray(arr, dtype=np.float32).tobytes()
        return base64.b64encode(x).decode("ascii")

    b64_ref = _extract_audio_b64(clone_ref_audio) if clone_enabled else ""

    def synth_and_stream(segment_text: str):
        nonlocal acc_pcm
        nonlocal emit_buf_sa
        st = segment_text.strip()
        print(f"[OllamaTTS] Synthesizing segment (len={len(st)}): {st[:160]!r}")

        # In sentence-aware mode, use non-streaming generation for Smart Voice parity
        if sentence_aware_stream:
            if clone_enabled and b64_ref and clone_ref_text.strip():
                messages = [
                    Message(role="user", content=clone_ref_text.strip()),
                    Message(role="assistant", content=AudioContent(raw_audio=b64_ref, audio_url="placeholder")),
                    Message(role="user", content=st),
                ]
            else:
                system_prompt = _build_system_prompt("Audio is recorded from a quiet room.")
                messages = [
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=st),
                ]
            chat_ml = ChatMLSample(messages=messages)
            sr_seg, audio_seg = _run_generate(
                chat_ml,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                max_new_tokens=1024,
            )
            # Append without overlap; batch until >= 400ms before yielding to client
            if audio_seg is not None and audio_seg.size > 0:
                audio_seg = np.nan_to_num(audio_seg.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                acc_pcm = _crossfade_concat(acc_pcm, audio_seg, sr_seg, 0)
                emit_buf_sa = np.concatenate([emit_buf_sa, audio_seg])
                min_ms_sa = 400
                if emit_buf_sa.size >= int(sr_seg * (min_ms_sa / 1000.0)):
                    if cancel_evt is None or not cancel_evt.is_set():
                        yield (sr_seg, emit_buf_sa), transcript_acc
                    emit_buf_sa = np.array([], dtype=np.float32)
            return

        # Fallback: original delta-stream path for non-sentence-aware mode
        chat_ml = _build_chatml_for_text(st)

        q: pyqueue.Queue = pyqueue.Queue(maxsize=128)
        # Buffer for gating small tails before emitting to client
        emit_buf = np.array([], dtype=np.float32)
        async def producer():
            try:
                async for delta in engine.generate_delta_stream(
                    chat_ml_sample=chat_ml,
                    max_new_tokens=1024,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>", "<|audio_eos|>"],
                ):
                    if cancel_evt is not None and cancel_evt.is_set():
                        break
                    q.put(delta)
            finally:
                q.put(None)

        th = threading.Thread(target=lambda: asyncio.run(producer()), daemon=True)
        th.start()

        token_buf: Optional[torch.Tensor] = None
        last_decoded_len = 0
        while True:
            item = q.get()
            if item is None:
                break
            if cancel_evt is not None and cancel_evt.is_set():
                break
            if isinstance(item, AudioContent) and item.audio_tokens is not None:
                if token_buf is None:
                    token_buf = item.audio_tokens
                else:
                    token_buf = torch.cat([token_buf, item.audio_tokens], dim=1)

                if token_buf is not None and token_buf.shape[1] - last_decoded_len >= 8:
                    try:
                        vq_code = revert_delay_pattern(token_buf).clip(0, engine.audio_codebook_size - 1)
                        if vq_code.shape[1] > 1:
                            vq_code = vq_code[:, 1:]
                        pcm = engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                        new_tail = pcm[last_decoded_len:]
                        if new_tail.size > 0:
                            new_tail = np.nan_to_num(new_tail.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                            acc_pcm = _crossfade_concat(acc_pcm, new_tail, sr, seg_overlap_ms)
                            # Gate: accumulate until we have >= min_chunk_ms
                            emit_buf = np.concatenate([emit_buf, new_tail])
                            min_ms = max(200, int(chunk_ms))
                            if emit_buf.size >= int(sr * (min_ms / 1000.0)):
                                if cancel_evt is None or not cancel_evt.is_set():
                                    yield (sr, emit_buf), transcript_acc
                                emit_buf = np.array([], dtype=np.float32)
                        last_decoded_len = pcm.shape[0]
                    except Exception:
                        pass

        if token_buf is not None and token_buf.ndim == 2 and not (cancel_evt is not None and cancel_evt.is_set()):
            try:
                vq_code = revert_delay_pattern(token_buf).clip(0, engine.audio_codebook_size - 1)
                if vq_code.shape[1] > 1:
                    vq_code = vq_code[:, 1:]
                pcm = engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                if pcm.shape[0] > last_decoded_len:
                    new_tail = pcm[last_decoded_len:]
                    if new_tail.size > 0:
                        new_tail = np.nan_to_num(new_tail.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                        overlap_ms = 0 if sentence_aware_stream else seg_overlap_ms
                        acc_pcm = _crossfade_concat(acc_pcm, new_tail, sr, overlap_ms)
                        # Gate: accumulate until we have >= min_chunk_ms
                        emit_buf = np.concatenate([emit_buf, new_tail])
                        min_ms = max(200, int(chunk_ms))
                        if emit_buf.size >= int(sr * (min_ms / 1000.0)):
                            if cancel_evt is None or not cancel_evt.is_set():
                                yield (sr, emit_buf), transcript_acc
                            emit_buf = np.array([], dtype=np.float32)
            except Exception:
                pass

    # Stream text and synthesize segments
    history_msgs = chat_history if isinstance(chat_history, list) else []
    msgs = [*history_msgs, {"role": "user", "content": user_text.strip()}]
    if full_response_tts:
        # Buffer full LLM text first, updating transcript as it arrives, then synthesize once
        full_text = ""
        for chunk in _ollama_stream_chat(ollama_model, user_text.strip(), cancel_evt, messages=msgs):
            if cancel_evt is not None and cancel_evt.is_set():
                break
            full_text += chunk
            # Text-only update during buffering
            yield gr.update(), full_text
        if cancel_evt is None or not cancel_evt.is_set():
            final_text = full_text.strip()
            if final_text:
                print(f"[OllamaTTS:Full] Synthesizing full response (len={len(final_text)}). Using _run_generate (non-stream) for parity with Smart Voice.")
                # Build ChatML using clone context if enabled
                if clone_enabled and b64_ref and clone_ref_text.strip():
                    messages = [
                        Message(role="user", content=clone_ref_text.strip()),
                        Message(role="assistant", content=AudioContent(raw_audio=b64_ref, audio_url="placeholder")),
                        Message(role="user", content=final_text),
                    ]
                else:
                    system_prompt = _build_system_prompt("Audio is recorded from a quiet room.")
                    messages = [
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=final_text),
                    ]
                chat_ml_full = ChatMLSample(messages=messages)
                # Use same defaults as Smart Voice advanced controls
                sr_full, audio_full = _run_generate(
                    chat_ml_full,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=50,
                    max_new_tokens=1024,
                )
                # Emit the final audio once as a single chunk (sanitized)
                audio_full = np.nan_to_num(audio_full.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                yield (sr_full, audio_full), final_text
        return

    for chunk in _ollama_stream_chat(ollama_model, user_text.strip(), cancel_evt, messages=msgs):
        if cancel_evt is not None and cancel_evt.is_set():
            break
        buf += chunk
        transcript_acc += chunk
        # Always update transcript promptly without touching audio
        if cancel_evt is not None and cancel_evt.is_set():
            break
        yield gr.update(), transcript_acc
        while True:
            if sentence_aware_stream:
                seg, rem = _pop_sentence(buf)
            else:
                seg, rem = _pop_segment(buf, seg_min_chars, seg_max_chars)
            if not seg:
                break
            buf = rem
            # Ensure transcript shows before starting audio for this unit (text-only)
            yield gr.update(), transcript_acc
            # Synthesize this unit and stream its audio (single run per unit)
            if cancel_evt is not None and cancel_evt.is_set():
                break
            yield from synth_and_stream(seg)

    # After stream end, flush remaining buffer
    if cancel_evt is None or not cancel_evt.is_set():
        tail = buf.strip()
        if tail and (cancel_evt is None or not cancel_evt.is_set()):
            # If sentence-aware, try to flush remaining complete sentences first
            if sentence_aware_stream:
                while True:
                    seg, rem = _pop_sentence(tail)
                    if not seg:
                        break
                    yield from synth_and_stream(seg)
                    tail = rem.strip()
            # Any small leftover (no terminator) can be synthesized once
            if tail:
                yield from synth_and_stream(tail)
        if cancel_evt is None or not cancel_evt.is_set():
            # Flush any remaining gated audio for non-sentence-aware path
            if 'emit_buf' in locals() and emit_buf.size > 0:
                yield (sr, emit_buf), transcript_acc
            # Flush any remaining batched sentence-aware audio
            if 'emit_buf_sa' in locals() and emit_buf_sa.size > 0:
                yield (sr, emit_buf_sa), transcript_acc
            # Final text update only to avoid buffer replacement
            yield gr.update(), transcript_acc


def _append_history(history: Optional[list], user_text: str, assistant_text: str) -> list:
    """Utility to append a user/assistant turn to chat history state."""
    hist = history if isinstance(history, list) else []
    u = (user_text or "").strip()
    a = (assistant_text or "").strip()
    if u:
        hist = [*hist, {"role": "user", "content": u}]
    if a:
        hist = [*hist, {"role": "assistant", "content": a}]
    return hist


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
        audio_out_smart = gr.Audio(label="Output", type="numpy", autoplay=True)
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
        audio_out_clone = gr.Audio(label="Output", type="numpy", autoplay=True)
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
        audio_out_ms = gr.Audio(label="Output", type="numpy", autoplay=True)
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

    with gr.Tab("Chat"):
        gr.Markdown(
            """
            ### Chat with TTS (Mock Streaming)
            This tab simulates near-real-time audio streaming. After approval, we'll wire it to an LLM and the real
            `generate_delta_stream()` for low-latency voice responses.
            """
        )
        with gr.Row():
            chat_input = gr.Textbox(label="Your message", placeholder="Ask me anything...", lines=3)
            btn_send_chat = gr.Button("Send", variant="primary")
        with gr.Row():
            llm_provider = gr.Dropdown(
                ["Mock", "Direct TTS", "Ollama", "OpenAI", "Local vLLM"],
                value="Ollama",
                label="LLM Provider",
            )
            voice_mode = gr.Dropdown(["Default", "Warm", "Bright"], value="Default", label="Voice preset")
            chunk_ms = gr.Slider(40, 400, value=120, step=20, label="Chunk duration (ms)")
            ollama_model = gr.Dropdown(
                choices=(
                    ["qwen2.5-coder:1.5b-maxctx"]
                    + (list_ollama_models() or ["<no local models detected>"])
                ),
                value="qwen2.5-coder:1.5b-maxctx",
                label="Ollama model (if provider is Ollama)",
            )
        with gr.Row():
            full_response_toggle = gr.Checkbox(value=False, label="Full response TTS (no segmentation)")
        with gr.Row():
            sentence_aware_toggle = gr.Checkbox(value=True, label="Sentence-aware streaming (per sentence)")
        with gr.Accordion("Segmentation (for Ollama incremental TTS)", open=False):
            with gr.Row():
                seg_min_chars = gr.Slider(40, 300, value=80, step=10, label="Min chars per segment")
                seg_max_chars = gr.Slider(120, 500, value=200, step=10, label="Max chars per segment")
                seg_overlap_ms = gr.Slider(0, 120, value=30, step=5, label="Audio crossfade overlap (ms)")
        with gr.Row():
            btn_start_chat = gr.Button("Start Streaming", variant="primary")
            btn_stop_chat = gr.Button("Stop", variant="stop")
        with gr.Row():
            audio_out_chat = gr.Audio(label="Streaming Audio", type="numpy", autoplay=True, streaming=True)
        with gr.Row():
            text_out_chat = gr.Textbox(label="Assistant (streaming transcript)", lines=6)
        with gr.Accordion("Voice Cloning (optional)", open=False):
            with gr.Row():
                clone_enabled = gr.Checkbox(value=False, label="Use voice cloning for assistant replies")
            with gr.Row():
                clone_ref_audio = gr.File(label="Reference voice audio (wav/mp3)")
            with gr.Row():
                clone_ref_text = gr.Textbox(label="Reference transcription (what the reference audio says)", lines=2)

        # Cancellation token state and chat history state for the current session
        cancel_state = gr.State(value=None)
        chat_history_state = gr.State(value=[])

        # Start: create a new cancel event, then stream
        btn_start_chat.click(
            _create_cancel_event,
            inputs=None,
            outputs=[cancel_state],
        ).then(
            chat_stream_entry,
            inputs=[chat_input, llm_provider, voice_mode, chunk_ms, ollama_model, seg_min_chars, seg_max_chars, seg_overlap_ms, full_response_toggle, sentence_aware_toggle, cancel_state, chat_history_state, clone_enabled, clone_ref_audio, clone_ref_text],
            outputs=[audio_out_chat, text_out_chat],
        ).then(
            _append_history,
            inputs=[chat_history_state, chat_input, text_out_chat],
            outputs=[chat_history_state],
        ).then(_clear_text, inputs=None, outputs=[chat_input])

        # Stop: signal cancellation
        btn_stop_chat.click(_cancel_event, inputs=[cancel_state], outputs=None)

        # Send button mirrors Start Streaming chain (quick single-turn send)
        btn_send_chat.click(
            _create_cancel_event,
            inputs=None,
            outputs=[cancel_state],
        ).then(
            chat_stream_entry,
            inputs=[chat_input, llm_provider, voice_mode, chunk_ms, ollama_model, seg_min_chars, seg_max_chars, seg_overlap_ms, full_response_toggle, sentence_aware_toggle, cancel_state, chat_history_state, clone_enabled, clone_ref_audio, clone_ref_text],
            outputs=[audio_out_chat, text_out_chat],
        ).then(
            _append_history,
            inputs=[chat_history_state, chat_input, text_out_chat],
            outputs=[chat_history_state],
        ).then(_clear_text, inputs=None, outputs=[chat_input])


if __name__ == "__main__":
    # Bind to 0.0.0.0 to allow LAN access
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

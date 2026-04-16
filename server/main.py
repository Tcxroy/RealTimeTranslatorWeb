"""
SyncVoice Server — MacBook Pro M4 · 36GB Unified Memory
FastAPI + WebSocket + Cohere Transcribe + Silero VAD + Ollama

Pipeline:
  Client (audio PCM chunks)
    → WebSocket
      → VAD (silero-vad, CPU/ANE)
        → Cohere Transcribe  (CohereLabs/cohere-transcribe-03-2026, 2B params)
          → Ollama  (qwen3.5:9b / translategemma:4b / translategemma:12b)
            → Streaming JSON  → Client

Cohere Transcribe notes:
  - 2B parameter open-source ASR model, Apache 2.0 license
  - Best-in-class WER across 14 languages (incl. Dutch/nl)
  - Requires explicit language code — no auto-detect
  - Runs well on Apple Silicon (MPS) or CPU
  - Model loads from HuggingFace on first run (~4GB download)
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import mlx_whisper

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("syncvoice")

# ─────────────────────────────────────────────
#  Config  (edit here or via env)
# ─────────────────────────────────────────────
# MLX-Whisper 模型配置 (Apple Silicon 优化)
# 注意: MLX-Whisper 使用专门为 MLX 优化的模型，不是标准的 openai/whisper 模型
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"  # large-v3-turbo on M4 36GB
WHISPER_LANGUAGE = None  # None = auto-detect

# ── Energy-based VAD (no extra deps) ───────────────────────────────────────
# RMS energy gate — faster and more reliable than Silero for real-time use
ENERGY_VAD_THRESHOLD = 0.01   # RMS amplitude (0..1).  Raise if too sensitive in noisy room.
ENERGY_VAD_HANGOVER  = 3      # keep "voice=True" for this many frames after energy drops

OLLAMA_BASE_URL   = "http://localhost:11434"
DEFAULT_LLM       = "qwen3.5:9b"       # translategemma:4b | translategemma:12b
SAMPLE_RATE       = 16000              # Hz  (Cohere Transcribe expects 16kHz mono float32)

# ── VAD / segmentation ─────────────────────────────────────────────────────
# Each "frame" = one ScriptProcessor callback = 4096 samples @ 16kHz ≈ 256ms
FRAME_MS          = 256                # approx ms per audio chunk from browser

VAD_THRESHOLD     = 0.5                # Silero VAD confidence gate
# Long-pause flush: sustained silence → commit segment (default 5 frames ≈ 1.3s)
SILENCE_FRAMES    = 5                  # ↓ lowered from 15 (was ~3.8s, now ~1.3s)
# Hard-cap flush: even with continuous speech, flush every N seconds to avoid
# accumulating too much text that the LLM can't fit in num_predict tokens.
MAX_BUFFER_SECS   = 5.0                # force-flush if buffer exceeds this
# Short-pause flush: a brief dip in VAD that doesn't meet SILENCE_FRAMES
# but suggests a natural sentence boundary (e.g. comma pause ~300-600ms)
SHORT_PAUSE_FRAMES = 2                 # 2 consecutive low-VAD frames ≈ 512ms
SHORT_PAUSE_THRESHOLD = 0.25           # VAD conf below this = "soft silence"
# Don't flush tiny buffers with almost no speech (avoids translating "hmm")
MIN_SPEECH_FRAMES = 2                  # minimum voice frames before flush counts
# ──────────────────────────────────────────────────────────────────────────

CONTEXT_WORDS     = 5                  # last N words carried as context overlap
MAX_ANCHORS       = 6                  # keep most recent K anchors in system prompt
CHUNK_MS          = 256               # expected client chunk size (ms) — matches ScriptProcessor

# Supported LLM models
LLM_MODEL_OPTIONS = ["qwen3.5:9b", "translategemma:4b", "translategemma:12b"]

# MLX-Whisper 支持的语言（99 种语言）
WHISPER_SUPPORTED_LANGS = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dv', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'zh'
]

# Language code to full name mapping for LLM prompts
LANGUAGE_NAMES = {
    'nl': 'Dutch',
    'en': 'English',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ru': 'Russian',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'el': 'Greek',
    'pl': 'Polish',
    'vi': 'Vietnamese',
}

# ─────────────────────────────────────────────
#  Lifespan (replaces deprecated on_event)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Server ready — using MLX-Whisper %s for ASR", WHISPER_MODEL)
    log.info("MLX-Whisper models are loaded on-demand from HuggingFace (~3GB for large-v3-turbo)")
    yield

# ─────────────────────────────────────────────
#  App bootstrap
# ─────────────────────────────────────────────
app = FastAPI(title="SyncVoice", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




# ─────────────────────────────────────────────
#  MLX-Whisper ASR
# ─────────────────────────────────────────────
def transcribe(pcm_float32: np.ndarray, source_lang: Optional[str] = None) -> str:
    """Transcribe a 16kHz mono float32 PCM array using MLX-Whisper."""
    import io
    import wave
    import tempfile

    t0 = time.perf_counter()

    # Convert float32 PCM to int16
    pcm_int16 = np.clip(pcm_float32 * 32767, -32768, 32768).astype(np.int16)

    # Create WAV file in memory
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_int16.tobytes())
    wav_io.seek(0)
    wav_data = wav_io.read()

    # MLX-Whisper 需要文件路径，创建临时文件
    language = source_lang if source_lang else WHISPER_LANGUAGE

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_data)
            tmp_path = tmp.name

        # Transcribe with MLX-Whisper
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=WHISPER_MODEL,
            language=language,
            verbose=False,
        )

        text = result.get("text", "").strip()

        # Clean up temp file
        os.unlink(tmp_path)

    except Exception as exc:
        log.error("MLX-Whisper error: %s", exc)
        return ""

    elapsed_ms = (time.perf_counter() - t0) * 1000
    detected_lang = result.get("language", language or "auto")
    log.info("ASR  %.0fms  lang=%s  model=%s  text='%s'",
             elapsed_ms, detected_lang, WHISPER_MODEL, text[:80])
    return text





# ─────────────────────────────────────────────
#  Ollama streaming translation
# ─────────────────────────────────────────────
async def translate_stream(
    original: str,
    src_lang: str,
    tgt_lang: str,
    context_tail: str,
    anchors: list[str],
    model: str,
    ws: WebSocket,
    num_predict: int = 256,
):
    """
    Stream Ollama tokens back to WebSocket client as:
      {"type":"token",  "text":"<word>"}
      {"type":"done",   "original":"...", "translated":"...", "latency_ms":N}
    """
    anchor_block = ""
    if anchors:
        anchor_block = "Key context anchors (always respect these):\n"
        anchor_block += "\n".join(f"  • {a}" for a in anchors[-MAX_ANCHORS:])
        anchor_block += "\n\n"

    # Convert language codes to full names for LLM clarity
    src_lang_name = LANGUAGE_NAMES.get(src_lang, src_lang) if src_lang != "auto-detected" else "Auto-detected language"
    tgt_lang_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)

    system_prompt = (
        f"You are a professional real-time simultaneous interpreter. "
        f"Translate from {src_lang_name} to {tgt_lang_name}.\n"
        f"{anchor_block}"
        f"Rules:\n"
        f"  1. Output ONLY the translation — no explanations, no notes.\n"
        f"  2. Preserve proper nouns, technical terms, and speaker tone.\n"
        f"  3. Be concise; this is live interpretation.\n"
        f"  4. If unsure of a term, keep the original in parentheses.\n"
    )

    user_msg = original
    if context_tail:
        user_msg = (
            f"[Previous context: …{context_tail}]\n\n"
            f"Translate this: {original}"
        )

    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        "options": {
            "temperature": 0.1,    # lower = more deterministic, faster convergence
            "top_p": 0.9,
            "num_predict": num_predict,   # adaptive: longer audio → more tokens allowed
            "num_ctx": 1024,       # short context window = faster prefill
        },
    }

    t0 = time.perf_counter()
    translated_tokens: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        translated_tokens.append(token)
                        await ws.send_json({"type": "token", "text": token})

                    if chunk.get("done", False):
                        break

    except httpx.HTTPError as exc:
        log.error("Ollama HTTP error: %s", exc)
        await ws.send_json({"type": "error", "message": str(exc)})
        return

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    translated = "".join(translated_tokens).strip()
    log.info("LLM  %dms  model=%s  '%s'", elapsed_ms, model, translated[:80])

    await ws.send_json({
        "type":       "done",
        "original":   original,
        "translated": translated,
        "latency_ms": elapsed_ms,
        "model":      model,
    })


# ─────────────────────────────────────────────
#  WebSocket endpoint
# ─────────────────────────────────────────────
@app.websocket("/ws/interpret")
async def interpret(ws: WebSocket):
    await ws.accept()
    log.info("Client connected: %s", ws.client)

    # ── Per-connection audio state ──────────────────────────────
    audio_buffer:    list[np.ndarray] = []   # raw PCM frames accumulator
    silent_count:    int              = 0    # consecutive low-VAD frames
    soft_pause_count:int              = 0    # consecutive "soft silence" frames
    speech_frames:   int              = 0    # voice frames in current buffer
    buffer_start_ts: float            = 0.0  # wall-clock time buffer started filling
    context_tail:    str              = ""   # last N words of previous translation
    anchors:         list[str]        = []   # semantic anchor list
    energy_hangover: int              = 0    # energy VAD hangover counter

    # ── Per-connection configurable parameters ──────────────────
    src_lang          = None           # None = auto-detect
    tgt_lang          = "English"
    llm_model         = DEFAULT_LLM
    # Segmentation params (all adjustable via "config" message)
    silence_frames    = SILENCE_FRAMES
    max_buffer_secs   = MAX_BUFFER_SECS
    short_pause_frames= SHORT_PAUSE_FRAMES

    def _buf_secs() -> float:
        return sum(len(f) for f in audio_buffer) / SAMPLE_RATE

    async def _flush(reason: str):
        """Flush current audio buffer to ASR→LLM pipeline."""
        nonlocal audio_buffer, silent_count, soft_pause_count, speech_frames, buffer_start_ts, context_tail, energy_hangover
        if not audio_buffer or speech_frames < MIN_SPEECH_FRAMES:
            audio_buffer.clear()
            silent_count = soft_pause_count = speech_frames = 0
            buffer_start_ts = 0.0
            return
        segment = np.concatenate(audio_buffer)
        dur = _buf_secs()
        log.info("Flush  reason=%-12s  dur=%.2fs  speech_frames=%d", reason, dur, speech_frames)
        audio_buffer = []
        silent_count = soft_pause_count = speech_frames = 0
        buffer_start_ts = 0.0
        # Adaptive num_predict: allow more tokens for longer segments
        tokens = min(512, max(128, int(dur * 30)))
        asyncio.create_task(
            _process_segment(segment, src_lang, tgt_lang, llm_model, context_tail, anchors, ws, tokens)
        )
        context_tail = ""

    # Send ready signal
    await ws.send_json({
        "type":    "ready",
        "asr":     "mlx-whisper",
        "asr_model": WHISPER_MODEL,
        "llm":     llm_model,
        "message": "SyncVoice server ready",
        "supported_langs": WHISPER_SUPPORTED_LANGS,
        "seg_params": {
            "silence_frames":     silence_frames,
            "max_buffer_secs":    max_buffer_secs,
            "short_pause_frames": short_pause_frames,
        }
    })

    try:
        async for message in ws.iter_text():
            # ── Control messages (JSON strings) ──────────────────
            try:
                ctrl = json.loads(message)
                msg_type = ctrl.get("type", "")

                if msg_type == "config":
                    src_lang  = ctrl.get("src_lang", src_lang)
                    tgt_lang  = ctrl.get("tgt_lang", tgt_lang)
                    llm_model = ctrl.get("model",    llm_model)
                    # Segmentation tuning from frontend sliders
                    if "silence_frames"     in ctrl: silence_frames     = int(ctrl["silence_frames"])
                    if "max_buffer_secs"    in ctrl: max_buffer_secs    = float(ctrl["max_buffer_secs"])
                    if "short_pause_frames" in ctrl: short_pause_frames = int(ctrl["short_pause_frames"])
                    # Validate src_lang against MLX-Whisper supported languages
                    if src_lang and src_lang not in WHISPER_SUPPORTED_LANGS:
                        log.warning("src_lang '%s' not in MLX-Whisper supported langs, will auto-detect", src_lang)
                    log.info("Config  %s→%s  llm=%s  sil=%d  max=%.1fs  short=%d",
                             src_lang, tgt_lang, llm_model, silence_frames, max_buffer_secs, short_pause_frames)
                    await ws.send_json({
                        "type": "config_ack",
                        "src_lang": src_lang,
                        "tgt_lang": tgt_lang,
                        "model": llm_model,
                        "asr": "mlx-whisper",
                        "asr_model": WHISPER_MODEL,
                        "supported_langs": WHISPER_SUPPORTED_LANGS,
                        "seg_params": {
                            "silence_frames":     silence_frames,
                            "max_buffer_secs":    max_buffer_secs,
                            "short_pause_frames": short_pause_frames,
                        }
                    })

                elif msg_type == "anchor":
                    anchors.append(ctrl.get("text", ""))
                    log.info("Anchor added: %s", ctrl.get("text"))

                elif msg_type == "clear_anchors":
                    anchors.clear()

                elif msg_type == "flush":
                    await _flush("manual")

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})

                continue  # JSON processed, skip audio path

            except (json.JSONDecodeError, TypeError):
                pass  # Not JSON — fall through to audio path

            # ── Audio chunk (base64-encoded raw PCM float32 little-endian) ──

            try:
                raw = base64.b64decode(message)
                pcm = np.frombuffer(raw, dtype=np.float32).copy()
            except Exception as exc:
                log.warning("Bad audio chunk: %s", exc)
                continue

            if len(pcm) == 0:
                continue

            # ── Energy-based VAD ──────────────────────────────────
            rms = float(np.sqrt(np.mean(pcm ** 2)))
            raw_voice = rms >= ENERGY_VAD_THRESHOLD

            # Hangover: keep voice=True for a few frames after energy drops
            # (avoids cutting off trailing consonants / word endings)
            if raw_voice:
                energy_hangover = ENERGY_VAD_HANGOVER
            elif energy_hangover > 0:
                energy_hangover -= 1

            is_voice = raw_voice or (energy_hangover > 0)
            # "soft silence": energy dropped but hangover still active
            is_soft  = (not raw_voice) and (energy_hangover > 0)
            vad_conf = min(1.0, rms / max(ENERGY_VAD_THRESHOLD, 1e-9))

            buf_s = _buf_secs()
            await ws.send_json({
                "type":       "vad",
                "voice":      is_voice,
                "conf":       round(vad_conf, 3),
                "buf_secs":   round(buf_s, 2),
                "buf_pct":    round(min(buf_s / max_buffer_secs, 1.0) * 100, 1),
            })

            # ── Buffer accumulation ───────────────────────────────
            if is_voice:
                if not audio_buffer:
                    buffer_start_ts = time.monotonic()
                audio_buffer.append(pcm)
                speech_frames   += 1
                silent_count     = 0
                soft_pause_count = 0

            else:
                # Append a little silence for natural acoustic boundaries
                if audio_buffer:
                    audio_buffer.append(pcm)

                if is_soft:
                    soft_pause_count += 1
                    silent_count      = 0
                else:
                    silent_count      += 1
                    soft_pause_count   = 0

            # ── Multi-tier flush decision ─────────────────────────
            #
            # Tier 1 — Hard cap: buffer has grown beyond max_buffer_secs
            #   → flush immediately regardless of VAD state
            #   Purpose: prevents LLM from receiving too much text at once
            if audio_buffer and _buf_secs() >= max_buffer_secs:
                await _flush("max_dur")
                continue

            # Tier 2 — Long pause: sustained silence (e.g. end of sentence)
            #   Traditional VAD flush, but threshold is now much lower (5 vs 15)
            if silent_count >= silence_frames and audio_buffer:
                await _flush("long_pause")
                continue

            # Tier 3 — Short pause: brief VAD dip suggesting comma/phrase boundary
            #   Only fires when buffer already has meaningful content (≥2s)
            #   Prevents over-segmenting at the very start of utterances
            if (soft_pause_count >= short_pause_frames
                    and speech_frames >= MIN_SPEECH_FRAMES
                    and _buf_secs() >= 2.0
                    and audio_buffer):
                await _flush("short_pause")
                continue

    except WebSocketDisconnect:
        log.info("Client disconnected: %s", ws.client)


async def _process_segment(
    segment:      np.ndarray,
    src_lang:     Optional[str],
    tgt_lang:     str,
    llm_model:    str,
    context_tail: str,
    anchors:      list[str],
    ws:           WebSocket,
    num_predict:  int = 256,
):
    """ASR → LLM pipeline for one audio segment (runs as background task)."""
    # Notify client: transcription started
    await ws.send_json({"type": "processing", "stage": "asr"})

    original = await asyncio.get_event_loop().run_in_executor(
        None, transcribe, segment, src_lang
    )

    if not original.strip():
        segment_duration = len(segment) / SAMPLE_RATE
        log.warning("Empty transcription: src_lang=%s, segment_duration=%.2fs — possible language detection failure",
                    src_lang, segment_duration)
        await ws.send_json({"type": "empty"})
        return

    await ws.send_json({"type": "original", "text": original})
    await ws.send_json({"type": "processing", "stage": "llm", "model": llm_model})

    # Build 5-word context tail from last sentence
    tail_words = context_tail.split()[-CONTEXT_WORDS:] if context_tail else []
    tail_str   = " ".join(tail_words)

    await translate_stream(
        original     = original,
        src_lang     = src_lang or "auto-detected",
        tgt_lang     = tgt_lang,
        context_tail = tail_str,
        anchors      = anchors,
        model        = llm_model,
        ws           = ws,
        num_predict  = num_predict,
    )


# ─────────────────────────────────────────────
#  Health / Info endpoints
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "asr":              "mlx-whisper",
        "asr_model":        WHISPER_MODEL,
        "asr_langs":       WHISPER_SUPPORTED_LANGS,
        "ollama":           OLLAMA_BASE_URL,
        "llm":              DEFAULT_LLM,
        "llm_opts":         LLM_MODEL_OPTIONS,
    }


@app.get("/models")
async def list_models():
    """Proxy Ollama model list."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            data = resp.json()
            # Inject local ASR info
            data["asr"]         = "mlx-whisper"
            data["asr_model"]   = WHISPER_MODEL
            data["asr_langs"]  = WHISPER_SUPPORTED_LANGS
            return data
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5500, reload=True)

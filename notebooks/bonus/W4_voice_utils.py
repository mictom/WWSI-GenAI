"""
Voice Utilities for OpenAI STT (Whisper) and TTS APIs.

This module provides helper functions for:
- Speech-to-Text (STT) using OpenAI Whisper
- Text-to-Speech (TTS) using OpenAI TTS API
"""

import io
from pathlib import Path
from openai import OpenAI

from dotenv import load_dotenv
# Load .env from project root (two levels up from bonus/)
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')


def get_openai_client() -> OpenAI:
    """Get OpenAI client instance."""
    return OpenAI()


def transcribe_audio(
    audio_bytes: bytes | io.BytesIO,
    language: str = None,
    prompt: str = None
) -> str:
    """
    Transcribe audio to text using OpenAI Whisper API.

    Args:
        audio_bytes: Audio data as bytes or BytesIO object
        language: Optional language code (e.g., 'en', 'pl') for better accuracy
        prompt: Optional prompt to guide transcription style

    Returns:
        Transcribed text string
    """
    client = get_openai_client()

    # Convert bytes to BytesIO if needed
    if isinstance(audio_bytes, bytes):
        audio_file = io.BytesIO(audio_bytes)
    else:
        audio_file = audio_bytes

    # Whisper API requires a filename
    audio_file.name = "recording.wav"

    # Build kwargs
    kwargs = {
        "model": "whisper-1",
        "file": audio_file,
        "response_format": "text"
    }

    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt

    transcript = client.audio.transcriptions.create(**kwargs)

    return transcript.strip()


def text_to_speech(
    text: str,
    voice: str = "nova",
    model: str = "tts-1",
    response_format: str = "mp3",
    speed: float = 1.0
) -> bytes:
    """
    Convert text to speech using OpenAI TTS API.

    Args:
        text: Text to convert to speech (max 4096 characters)
        voice: Voice to use. Options: alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer
        model: TTS model. Options: tts-1 (fast), tts-1-hd (high quality)
        response_format: Audio format. Options: mp3, opus, aac, flac, wav, pcm
        speed: Speech speed from 0.25 to 4.0 (default 1.0)

    Returns:
        Audio data as bytes
    """
    client = get_openai_client()

    # Truncate text if too long
    if len(text) > 4096:
        text = text[:4093] + "..."

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
        speed=speed
    )

    return response.content


def text_to_speech_streaming(
    text: str,
    voice: str = "nova",
    model: str = "tts-1",
    response_format: str = "mp3",
    speed: float = 1.0
):
    """
    Convert text to speech with streaming response.

    Yields audio chunks as they're generated for lower latency playback.

    Args:
        text: Text to convert to speech
        voice: Voice to use
        model: TTS model
        response_format: Audio format
        speed: Speech speed

    Yields:
        Audio data chunks as bytes
    """
    client = get_openai_client()

    if len(text) > 4096:
        text = text[:4093] + "..."

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
        speed=speed
    ) as response:
        for chunk in response.iter_bytes(chunk_size=4096):
            yield chunk


# Available voices for reference
AVAILABLE_VOICES = [
    "alloy",    # Neutral
    "ash",      # Neutral
    "coral",    # Warm female
    "echo",     # Male
    "fable",    # British
    "onyx",     # Deep male
    "nova",     # Friendly female (recommended)
    "sage",     # Neutral
    "shimmer"   # Expressive female
]


if __name__ == "__main__":
    # Quick test
    print("Testing TTS...")
    audio = text_to_speech("Hello! This is a test of the text to speech system.")

    # Save to file for verification
    with open("test_tts_output.mp3", "wb") as f:
        f.write(audio)
    print(f"Saved test audio to test_tts_output.mp3 ({len(audio)} bytes)")

    print("\nAvailable voices:", ", ".join(AVAILABLE_VOICES))

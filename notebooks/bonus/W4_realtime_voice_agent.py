"""
OpenAI Realtime API Voice Agent - Terminal Application.

This script demonstrates real-time voice conversation with OpenAI's Realtime API.
It provides direct speech-to-speech interaction without intermediate text processing.

Requirements:
    pip install websockets sounddevice numpy

Usage:
    python W3_realtime_voice_agent.py

Press Ctrl+C to exit.
"""

import os
import sys
import json
import base64
import asyncio
import argparse
import numpy as np
from pathlib import Path

try:
    import websockets
    import sounddevice as sd
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install websockets sounddevice numpy")
    sys.exit(1)

from dotenv import load_dotenv
# Load .env from project root (two levels up from bonus/)
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_MODEL = "gpt-4o-realtime-preview"
REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

# Audio settings (OpenAI Realtime API requirements)
SAMPLE_RATE = 24000  # 24kHz required by OpenAI
CHANNELS = 1         # Mono
DTYPE = np.int16     # PCM16
CHUNK_SIZE = 2400    # 100ms chunks at 24kHz

# Voice options: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar
VOICE = "alloy"

# Language setting: "english" or "polish"
LANGUAGE = "english"

# System prompts for different languages
SYSTEM_INSTRUCTIONS_EN = """You are a helpful, friendly voice assistant.
You MUST respond in English only.
Keep your responses concise and conversational since this is a voice interaction.
If you don't understand something, ask for clarification.
Be natural and engaging in your responses."""

SYSTEM_INSTRUCTIONS_PL = """Jesteś pomocnym, przyjaznym asystentem głosowym.
MUSISZ odpowiadać TYLKO po polsku.
Odpowiadaj zwięźle i konwersacyjnie, ponieważ to jest interakcja głosowa.
Jeśli czegoś nie rozumiesz, poproś o wyjaśnienie.
Bądź naturalny i angażujący w swoich odpowiedziach."""

# Select system instructions based on language
SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_PL if LANGUAGE == "polish" else SYSTEM_INSTRUCTIONS_EN


class RealtimeVoiceAgent:
    """Real-time voice agent using OpenAI's Realtime API."""

    def __init__(self):
        self.websocket = None
        self.is_running = False
        self.audio_queue = asyncio.Queue()
        self.playback_queue = asyncio.Queue()
        self.is_speaking = False

    async def connect(self):
        """Establish WebSocket connection to OpenAI Realtime API."""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        print(f"🔌 Connecting to OpenAI Realtime API...")
        self.websocket = await websockets.connect(
            REALTIME_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=20
        )
        print("✅ Connected!")

        # Configure the session
        await self.configure_session()

    async def configure_session(self):
        """Send session configuration to OpenAI."""
        config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": SYSTEM_INSTRUCTIONS,
                "voice": VOICE,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",  # Voice Activity Detection
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            }
        }

        await self.websocket.send(json.dumps(config))
        print(f"🎙️ Session configured with voice: {VOICE}")

    async def send_audio(self):
        """Capture audio from microphone and send to API."""
        print("🎤 Microphone ready. Start speaking...")

        def audio_callback(indata, frames, time, status):
            """Callback for audio input stream."""
            if status:
                print(f"⚠️ Audio input status: {status}")
            # Put audio data in queue for async processing
            audio_bytes = indata.tobytes()
            try:
                self.audio_queue.put_nowait(audio_bytes)
            except asyncio.QueueFull:
                pass  # Drop frame if queue is full

        # Start audio input stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )

        with stream:
            while self.is_running:
                try:
                    # Get audio from queue
                    audio_bytes = await asyncio.wait_for(
                        self.audio_queue.get(),
                        timeout=0.1
                    )

                    # Encode as base64 and send
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_base64
                    }

                    await self.websocket.send(json.dumps(message))

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"⚠️ Send error: {e}")

    async def receive_messages(self):
        """Receive and process messages from the API."""
        audio_buffer = []

        while self.is_running:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=0.1
                )
                event = json.loads(message)
                event_type = event.get("type", "")

                # Handle different event types
                if event_type == "session.created":
                    print("📡 Session created successfully")

                elif event_type == "session.updated":
                    print("⚙️ Session updated")

                elif event_type == "input_audio_buffer.speech_started":
                    print("👂 Listening...")
                    self.is_speaking = False

                elif event_type == "input_audio_buffer.speech_stopped":
                    print("💭 Processing...")

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # User's speech transcription
                    transcript = event.get("transcript", "")
                    if transcript:
                        print(f"🗣️ You: {transcript}")

                elif event_type == "response.audio_transcript.delta":
                    # Partial AI response text
                    delta = event.get("delta", "")
                    print(delta, end="", flush=True)

                elif event_type == "response.audio_transcript.done":
                    # AI response complete
                    print()  # New line

                elif event_type == "response.audio.delta":
                    # Audio chunk from AI
                    self.is_speaking = True
                    audio_base64 = event.get("delta", "")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_buffer.append(audio_bytes)

                elif event_type == "response.audio.done":
                    # Play accumulated audio
                    if audio_buffer:
                        await self.play_audio(b''.join(audio_buffer))
                        audio_buffer = []
                    self.is_speaking = False

                elif event_type == "response.done":
                    print("🤖 Response complete")

                elif event_type == "error":
                    error = event.get("error", {})
                    print(f"❌ Error: {error.get('message', 'Unknown error')}")

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("🔌 Connection closed")
                break
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Receive error: {e}")

    async def play_audio(self, audio_bytes: bytes):
        """Play audio through speakers."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=DTYPE)

            # Play audio (blocking but in async context)
            sd.play(audio_array, samplerate=SAMPLE_RATE)
            sd.wait()

        except Exception as e:
            print(f"⚠️ Playback error: {e}")

    async def run(self):
        """Main run loop."""
        if not OPENAI_API_KEY:
            print("❌ Error: OPENAI_API_KEY not found in environment")
            print("   Set it in your .env file or export it")
            return

        try:
            await self.connect()
            self.is_running = True

            print("\n" + "=" * 50)
            print("🎙️  VOICE AGENT READY")
            print("=" * 50)
            print("Speak naturally - the AI will respond with voice.")
            print("Press Ctrl+C to exit.\n")

            # Run send and receive concurrently
            await asyncio.gather(
                self.send_audio(),
                self.receive_messages()
            )

        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        except Exception as e:
            print(f"❌ Fatal error: {e}")
        finally:
            self.is_running = False
            if self.websocket:
                await self.websocket.close()
            print("✅ Disconnected")


def check_audio_devices():
    """Check and display available audio devices."""
    print("\n📢 Audio Devices:")
    print("-" * 40)

    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')

        print(f"Input:  {default_input['name']}")
        print(f"Output: {default_output['name']}")
        print("-" * 40)
        return True
    except Exception as e:
        print(f"❌ Audio device error: {e}")
        return False


def main():
    """Entry point."""
    global SYSTEM_INSTRUCTIONS

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="OpenAI Realtime Voice Agent")
    parser.add_argument("--polish", "-pl", action="store_true", help="Use Polish language")
    parser.add_argument("--english", "-en", action="store_true", help="Use English language (default)")
    args = parser.parse_args()

    # Set language based on arguments
    if args.polish:
        SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_PL
        lang_display = "Polski 🇵🇱"
    else:
        SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_EN
        lang_display = "English 🇬🇧"

    print("\n" + "=" * 50)
    print("   OpenAI Realtime Voice Agent")
    print("=" * 50)
    print(f"🌐 Language: {lang_display}")

    # Check audio devices
    if not check_audio_devices():
        print("Please check your audio configuration.")
        sys.exit(1)

    # Create and run agent
    agent = RealtimeVoiceAgent()
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()

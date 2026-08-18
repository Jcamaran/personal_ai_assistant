"""Voice assistant entry point.

Press the configured hotkey to record a question, which is transcribed,
sent to the brain server, and answered out loud.
"""
import glob
import os
import sys
import time
from collections import deque

import keyboard
from rich.console import Console
from dotenv import load_dotenv

# Allow importing the shared package from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api_client import BrainServerClient
from audio.recorder import record_audio, record_until_silence
from audio.stt import SpeechRecognizer
from audio.tts import TextToSpeech

load_dotenv()


class VoiceAssistant:
    """Coordinates recording, transcription, querying, and speech output."""

    def __init__(self, config_path=".env"):
        load_dotenv(config_path)

        self.brain_url = os.getenv("BRAIN_SERVER_URL", "http://localhost:8000")
        self.hotkey = os.getenv("ACTIVATION_HOTKEY", "ctrl+shift+space")
        self.recording_duration = int(os.getenv("MAX_RECORDING_DURATION", 5))
        # 'fixed' records for a set duration; 'auto' records until silence
        self.recording_mode = os.getenv("RECORDING_MODE", "fixed").lower()
        default_activation = "manual" if sys.platform == "darwin" else "hotkey"
        self.activation_method = os.getenv("ACTIVATION_METHOD", default_activation).lower()

        # Heavy components are created lazily on first use
        self.client = None
        self.recognizer = None
        self.tts = None

        self.console = Console()
        self.conversation_history = deque(maxlen=5)
        self.is_running = False

    def _init_components(self):
        """Create client, recognizer, and TTS engine on first use."""
        if self.client is None:
            self.client = BrainServerClient(base_url=self.brain_url)

        if self.recognizer is None:
            model_size = os.getenv("WHISPER_MODEL", "base")
            device = os.getenv("WHISPER_DEVICE", "auto")
            self.recognizer = SpeechRecognizer(model_size=model_size, device=device)

        if self.tts is None:
            self.tts = TextToSpeech(
                rate=int(os.getenv("TTS_RATE", 175)),
                volume=float(os.getenv("TTS_VOLUME", 0.9)),
                voice_index=int(os.getenv("TTS_VOICE_INDEX", 1)),
            )

    def _record(self) -> str:
        """Record a query, returning the path to the audio file."""
        output_file = f"temp_query_{int(time.time() * 1000)}.wav"

        if self.recording_mode == "auto":
            return record_until_silence(
                output_file=output_file,
                max_duration=self.recording_duration,
            )
        return record_audio(
            duration=self.recording_duration,
            output_file=output_file,
        )

    def process_voice_query(self):
        """Run one full record -> transcribe -> query -> speak cycle."""
        self._init_components()
        audio_file = None

        try:
            self.console.print("\n[bold green]Listening...[/bold green]")
            audio_file = self._record()

            self.console.print("[blue]Transcribing...[/blue]")
            query_text = self.recognizer.transcribe(audio_file)

            if not query_text:
                self.console.print("[yellow]No speech detected[/yellow]")
                self.tts.speak("I didn't hear anything.")
                return

            self.console.print(f"[cyan]You asked: {query_text}[/cyan]")

            if self.conversation_history:
                self.console.print(
                    f"[dim]Using {len(self.conversation_history)} previous "
                    f"interaction(s) for context[/dim]"
                )

            self.console.print("[blue]Thinking...[/blue]")
            response = self.client.query(
                query_text,
                top_k=5,
                conversation_history=list(self.conversation_history),
            )

            if response:
                self.conversation_history.append(
                    {"query": query_text, "answer": response.answer}
                )

                self.console.print("\n[bold green]Answer:[/bold green]")
                self.console.print(f"[white]{response.answer}[/white]\n")
                self.console.print(f"[dim]Sources: {len(response.sources)} documents[/dim]")
                if response.agent_trace:
                    trace = response.agent_trace
                    rewritten = trace.rewritten_query or ""
                    if rewritten and rewritten.strip() != query_text.strip():
                        self.console.print(f"[dim]Rewritten search: {rewritten}[/dim]")
                    self.console.print(
                        f"[dim]Agent: kept {trace.chunks_kept}/{trace.chunks_retrieved} chunks "
                        f"in {trace.iterations} round(s)"
                        f"{' (fallback)' if trace.used_fallback else ''}[/dim]"
                    )
                    if trace.dropped_file_names:
                        dropped = ", ".join(trace.dropped_file_names)
                        self.console.print(f"[dim]Dropped notes: {dropped}[/dim]")
                self.console.print(f"[dim]Processing time: {response.processing_time:.2f}s[/dim]\n")
                self.tts.speak(response.answer)
            else:
                self.console.print("[red]Failed to get response from brain server[/red]")
                self.tts.speak("Sorry, I couldn't process your request. Please check the brain server.")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            try:
                self.tts.speak("Sorry, an error occurred.")
            except Exception:
                pass

        finally:
            self._cleanup_temp_files(audio_file)

    def _cleanup_temp_files(self, audio_file):
        """Remove the current recording and any stale temp files."""
        for path in {audio_file, *glob.glob("temp_query_*.wav")}:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def check_health(self) -> bool:
        """Return True if the brain server is reachable and healthy."""
        self._init_components()
        return self.client.check_health()

    def run(self):
        """Start the main loop and wait for hotkey activation."""
        self.console.print("[bold blue]Obsidian Voice Assistant started[/bold blue]")
        if self.activation_method == "manual":
            self.console.print("[yellow]Press Enter to activate voice input[/yellow]")
        else:
            self.console.print(f"[yellow]Press {self.hotkey} to activate voice input[/yellow]")
        self.console.print("[yellow]Press Ctrl+C to exit[/yellow]\n")

        self.console.print("[dim]Checking brain server connection...[/dim]")
        if not self.check_health():
            self.console.print("[red]Brain server is not healthy or not running[/red]")
            self.console.print("[yellow]Start the brain server first:[/yellow]")
            self.console.print("[dim]  cd brain_server[/dim]")
            self.console.print("[dim]  docker-compose up -d[/dim]\n")
            return

        self.console.print("[green]Brain server is healthy[/green]\n")
        self.console.print("[dim]Waiting for activation...[/dim]")

        if self.activation_method == "manual":
            self._run_manual_loop()
            return

        try:
            keyboard.add_hotkey(self.hotkey, self.process_voice_query)
            self.is_running = True
        except Exception as e:
            self.console.print(f"[red]Failed to register hotkey '{self.hotkey}': {e}[/red]")
            self.console.print("[yellow]Try running as administrator or changing the hotkey in .env[/yellow]")
            return

        try:
            keyboard.wait()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            self.shutdown()

    def _run_manual_loop(self):
        """Wait for Enter before each voice interaction."""
        self.is_running = True
        try:
            while True:
                input("\nPress Enter to speak (Ctrl+C to exit): ")
                self.process_voice_query()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Shutting down...[/yellow]")
        finally:
            self.shutdown()

    def shutdown(self):
        self.is_running = False
        if self.client is not None:
            self.client.close()
        self.console.print("[green]Voice assistant stopped[/green]")


def main():
    VoiceAssistant().run()


if __name__ == "__main__":
    main()

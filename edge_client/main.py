"""Main voice assistant loop for Obsidian integration.

Press the configured hotkey to activate voice input.
Asks questions to your Obsidian vault through the brain server.
"""
import os
import sys
import keyboard
from rich.console import Console
from dotenv import load_dotenv

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api_client import BrainServerClient
from audio.recorder import record_audio
from audio.stt import SpeechRecognizer
from audio.tts import TextToSpeech

load_dotenv()


class VoiceAssistant:
    """
    Main voice assistant orchestrator.
    Manages all components and handles the voice interaction loop.
    """
    
    def __init__(self, config_path=".env"):
        """
        Initialize the voice assistant.
        
        Args:
            config_path (str): Path to .env configuration file
        """
        load_dotenv(config_path)
        
        # Load configuration
        self.brain_url = os.getenv("BRAIN_SERVER_URL", "http://localhost:8000")
        self.hotkey = os.getenv("ACTIVATION_HOTKEY", "ctrl+shift+space")
        self.recording_duration = int(os.getenv("MAX_RECORDING_DURATION", 5))
        
        # Initialize components (lazy - will be created on first use)
        self.client = None
        self.recognizer = None
        self.tts = None
        self.console = Console()
        
        self.is_running = False
    
    def _init_components(self):
        """Initialize all components on first use (lazy loading)"""
        if self.client is None:
            self.client = BrainServerClient(base_url=self.brain_url)
        
        if self.recognizer is None:
            model_size = os.getenv("WHISPER_MODEL", "base")
            self.recognizer = SpeechRecognizer(model_size=model_size, device="auto")
        
        if self.tts is None:
            self.tts = TextToSpeech(
                rate=int(os.getenv("TTS_RATE", 175)),
                volume=float(os.getenv("TTS_VOLUME", 0.9)),
                voice_index=1  # Female voice
            )
    
    def process_voice_query(self):
        """Handle one complete voice interaction cycle."""
        # Initialize components if needed
        self._init_components()
        
        audio_file = None
        
        try:
            self.console.print("\n[bold green]🎤 Listening...[/bold green]")
            
            # Step 1: Record audio
            audio_file = record_audio(
                duration=self.recording_duration,
                output_file="temp_query.wav"
            )
            
            # Step 2: Transcribe to text
            self.console.print("[blue]🎯 Transcribing...[/blue]")
            query_text = self.recognizer.transcribe(audio_file)
            
            if not query_text:
                self.console.print("[yellow]⚠️  No speech detected[/yellow]")
                self.tts.speak("I didn't hear anything.")
                return
            
            self.console.print(f"[cyan]📝 You asked: {query_text}[/cyan]")
            
            # Step 3: Send query to brain server
            self.console.print("[blue]🧠 Thinking...[/blue]")
            response = self.client.query(query_text, top_k=5)
            
            if response:
                # Display answer
                self.console.print(f"\n[bold green]💡 Answer:[/bold green]")
                self.console.print(f"[white]{response.answer}[/white]\n")
                self.console.print(f"[dim]📚 Sources: {len(response.sources)} documents[/dim]")
                self.console.print(f"[dim]⏱️  Processing time: {response.processing_time:.2f}s[/dim]\n")
                
                # Step 4: Speak the answer
                self.tts.speak(response.answer)
            else:
                self.console.print("[red]❌ Failed to get response from brain server[/red]")
                self.tts.speak("Sorry, I couldn't process your request. Please check the brain server.")
        
        except KeyboardInterrupt:
            raise  # Allow Ctrl+C to propagate
        except Exception as e:
            self.console.print(f"[red]❌ Error: {e}[/red]")
            try:
                self.tts.speak("Sorry, an error occurred.")
            except:
                pass  # TTS might have failed
        
        finally:
            # Cleanup temporary audio file
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except:
                    pass  # Ignore cleanup errors
    
    def check_health(self):
        """
        Verify that the brain server is running and healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        self._init_components()
        return self.client.check_health()
    
    def run(self):
        """
        Start the voice assistant main loop.
        Waits for hotkey activation.
        """
        self.console.print("[bold blue]🚀 Obsidian Voice Assistant Started[/bold blue]")
        self.console.print(f"[yellow]Press {self.hotkey} to activate voice input[/yellow]")
        self.console.print(f"[yellow]Press Ctrl+C to exit[/yellow]\n")
        
        # Check brain server health
        self.console.print("[dim]Checking brain server connection...[/dim]")
        if not self.check_health():
            self.console.print("[red]❌ Brain server is not healthy or not running![/red]")
            self.console.print("[yellow]Please start the brain server first:[/yellow]")
            self.console.print("[dim]  cd brain_server[/dim]")
            self.console.print("[dim]  docker-compose up -d[/dim]\n")
            return
        
        self.console.print("[green]✅ Brain server is healthy[/green]\n")
        self.console.print("[dim]Waiting for activation...[/dim]")
        
        # Register hotkey
        try:
            keyboard.add_hotkey(self.hotkey, self.process_voice_query)
            self.is_running = True
        except Exception as e:
            self.console.print(f"[red]❌ Failed to register hotkey '{self.hotkey}': {e}[/red]")
            self.console.print("[yellow]Try running as administrator or changing the hotkey in .env[/yellow]")
            return
        
        try:
            # Keep the program running
            keyboard.wait()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]👋 Shutting down...[/yellow]")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup resources on shutdown."""
        self.is_running = False
        # Components will auto-cleanup via __del__ methods
        self.console.print("[green]✅ Voice assistant stopped[/green]")


def main():
    """Entry point for the voice assistant."""
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()



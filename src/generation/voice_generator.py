#!/usr/bin/env python3
"""
Voice Generator - Speak jokes aloud using text-to-speech

Features:
- Read jokes from results files (JSON)
- Support multiple TTS engines (pyttsx3 offline, gTTS online)
- Adjustable speech rate and voice selection
- Play single jokes or entire collections
- Save audio files for later playback
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print(
        "[WARNING] pyttsx3 not available. Install with: pipenv install pyttsx3",
        file=sys.stderr,
    )

try:
    from gtts import gTTS

    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ----------------------------- Configuration -----------------------------
RESULTS_FILE_DEFAULT = os.path.join("results", "idiom_jokes_results.json")
OUTPUT_DIR_DEFAULT = os.path.join("results", "audio")


class VoiceGenerator:
    """Generate speech from joke text using various TTS engines"""

    def __init__(
        self,
        engine: str = "pyttsx3",
        rate: int = 175,
        voice_index: Optional[int] = None,
    ):
        """
        Initialize the voice generator

        Args:
            engine: TTS engine to use ("pyttsx3" or "gtts")
            rate: Speech rate (words per minute) for pyttsx3
            voice_index: Voice selection index (None for default)
        """
        self.engine_name = engine
        self.rate = rate
        self.voice_index = voice_index
        self.tts_engine = None

        if engine == "pyttsx3":
            if not PYTTSX3_AVAILABLE:
                raise RuntimeError(
                    "pyttsx3 is not installed. Run: pipenv install pyttsx3"
                )
            self._init_pyttsx3()
        elif engine == "gtts":
            if not GTTS_AVAILABLE:
                raise RuntimeError("gtts is not installed. Run: pipenv install gtts")
        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'pyttsx3' or 'gtts'")

    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine with settings"""
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", self.rate)

        voices = self.tts_engine.getProperty("voices")
        if self.voice_index is not None and 0 <= self.voice_index < len(voices):
            self.tts_engine.setProperty("voice", voices[self.voice_index].id)
            print(f"[INFO] Using voice: {voices[self.voice_index].name}")

    def speak(self, text: str, wait: bool = True):
        """
        Speak the given text

        Args:
            text: Text to speak
            wait: Whether to wait for speech to complete (pyttsx3 only)
        """
        if self.engine_name == "pyttsx3":
            self.tts_engine.say(text)
            if wait:
                self.tts_engine.runAndWait()
        elif self.engine_name == "gtts":
            raise NotImplementedError("Use save_and_play() for gtts engine")

    def save_audio(self, text: str, output_path: str):
        """
        Save speech to an audio file

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.engine_name == "pyttsx3":
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
        elif self.engine_name == "gtts":
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(output_path)

        print(f"[INFO] Saved audio to: {output_path}")

    def save_and_play(self, text: str, temp_file: str = None):
        """
        Save to temporary file and play (useful for gtts)

        Args:
            text: Text to speak
            temp_file: Temporary file path (auto-generated if None)
        """
        if temp_file is None:
            temp_file = os.path.join(OUTPUT_DIR_DEFAULT, "temp_joke.mp3")

        self.save_audio(text, temp_file)
        self.play_audio(temp_file)

    @staticmethod
    def play_audio(audio_path: str):
        """
        Play an audio file

        Args:
            audio_path: Path to the audio file
        """
        if not os.path.exists(audio_path):
            print(f"[ERROR] Audio file not found: {audio_path}")
            return

        # Try using pygame first, then fall back to system commands
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return
            except Exception as e:
                print(f"[WARNING] pygame playback failed: {e}", file=sys.stderr)

        # Fall back to system commands
        import platform

        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                os.system(f"afplay '{audio_path}'")
            elif system == "Linux":
                os.system(
                    f"mpg123 '{audio_path}' 2>/dev/null || ffplay -nodisp -autoexit '{audio_path}' 2>/dev/null"
                )
            elif system == "Windows":
                os.system(f"start {audio_path}")
            else:
                print(f"[WARNING] Unsupported system for audio playback: {system}")
        except Exception as e:
            print(f"[ERROR] Failed to play audio: {e}")

    def list_voices(self):
        """List available voices (pyttsx3 only)"""
        if self.engine_name != "pyttsx3":
            print("[INFO] Voice listing only available for pyttsx3 engine")
            return

        voices = self.tts_engine.getProperty("voices")
        print(f"\n[INFO] Available voices ({len(voices)}):")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name} ({voice.id})")
            print(f"      Languages: {voice.languages}")


def load_jokes(file_path: str) -> List[Dict]:
    """
    Load jokes from a JSON results file

    Args:
        file_path: Path to the JSON file

    Returns:
        List of joke dictionaries
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        jokes = json.load(f)

    print(f"[INFO] Loaded {len(jokes)} joke(s) from {file_path}")
    return jokes


def format_joke_for_speech(joke_dict: Dict) -> str:
    """
    Format a joke dictionary for natural speech

    Args:
        joke_dict: Joke dictionary with 'idiom' and 'joke' keys

    Returns:
        Formatted text for speech
    """
    idiom = joke_dict.get("idiom", "")
    joke = joke_dict.get("joke", "")

    # Option 1: Just the joke
    # return joke

    # Option 2: Idiom setup + joke
    if idiom:
        return f"Based on the idiom: {idiom}. {joke}"
    return joke


def main():
    parser = argparse.ArgumentParser(
        description="Speak jokes aloud using text-to-speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Speak all jokes from results file (offline, fast)
  python src/generation/voice_generator.py

  # Speak a specific joke by index
  python src/generation/voice_generator.py --index 0

  # Use Google TTS (requires internet, better quality)
  python src/generation/voice_generator.py --engine gtts

  # Adjust speech rate (pyttsx3 only)
  python src/generation/voice_generator.py --rate 150

  # List available voices
  python src/generation/voice_generator.py --list-voices

  # Save joke audio to file without playing
  python src/generation/voice_generator.py --index 0 --save-only --output my_joke.mp3

  # Use a different voice (after listing with --list-voices)
  python src/generation/voice_generator.py --voice 1
        """,
    )

    parser.add_argument(
        "--file",
        "-f",
        default=RESULTS_FILE_DEFAULT,
        help=f"Jokes file to read from (default: {RESULTS_FILE_DEFAULT})",
    )
    parser.add_argument(
        "--index", "-i", type=int, help="Index of specific joke to speak (0-based)"
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=["pyttsx3", "gtts"],
        default="pyttsx3",
        help="TTS engine to use (default: pyttsx3)",
    )
    parser.add_argument(
        "--rate",
        "-r",
        type=int,
        default=175,
        help="Speech rate in words per minute for pyttsx3 (default: 175)",
    )
    parser.add_argument(
        "--voice", "-v", type=int, help="Voice index to use (see --list-voices)"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List available voices and exit"
    )
    parser.add_argument(
        "--save-only", action="store_true", help="Save audio to file without playing"
    )
    parser.add_argument("--output", "-o", help="Output file path for saved audio")
    parser.add_argument(
        "--format",
        choices=["joke-only", "with-idiom"],
        default="joke-only",
        help="Format for speech output (default: joke-only)",
    )

    args = parser.parse_args()

    # Initialize voice generator
    try:
        generator = VoiceGenerator(
            engine=args.engine, rate=args.rate, voice_index=args.voice
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # List voices if requested
    if args.list_voices:
        generator.list_voices()
        return 0

    # Load jokes
    jokes = load_jokes(args.file)
    if not jokes:
        return 1

    # Determine which jokes to speak
    if args.index is not None:
        if args.index < 0 or args.index >= len(jokes):
            print(f"[ERROR] Index {args.index} out of range (0-{len(jokes)-1})")
            return 1
        jokes_to_speak = [jokes[args.index]]
        print(f"[INFO] Speaking joke #{args.index}")
    else:
        jokes_to_speak = jokes
        print(f"[INFO] Speaking {len(jokes_to_speak)} joke(s)")

    # Speak or save jokes
    for i, joke_dict in enumerate(jokes_to_speak):
        # Format joke for speech
        if args.format == "with-idiom":
            text = format_joke_for_speech(joke_dict)
        else:
            text = joke_dict.get("joke", "")

        print(f"\n[{i+1}/{len(jokes_to_speak)}] {text}")

        # Handle save vs. play
        if args.save_only:
            if args.output:
                output_path = args.output
            else:
                # Auto-generate filename
                idiom_slug = joke_dict.get("idiom", f"joke_{i}").replace(" ", "_")[:50]
                output_path = os.path.join(OUTPUT_DIR_DEFAULT, f"{idiom_slug}.mp3")

            generator.save_audio(text, output_path)
        else:
            # Play immediately
            if args.engine == "pyttsx3":
                generator.speak(text, wait=True)
            elif args.engine == "gtts":
                temp_file = os.path.join(OUTPUT_DIR_DEFAULT, f"temp_{i}.mp3")
                generator.save_and_play(text, temp_file)

        # Small pause between jokes if playing multiple
        if (
            len(jokes_to_speak) > 1
            and i < len(jokes_to_speak) - 1
            and not args.save_only
        ):
            import time

            time.sleep(1)

    print(f"\n[INFO] Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

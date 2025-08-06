import argparse
import os
import sys

from mlx_audio.tts.generate import generate_audio

def main():
    parser = argparse.ArgumentParser(description="Generate audio from text using an MLX TTS model.")
    parser.add_argument("--text", required=True, help="The text to synthesize.")
    parser.add_argument("--model-path", required=True, help="The path to the TTS model.")
    parser.add_argument("--output-path", default="generated_speech.wav", help="The path to save the generated audio file.")
    args = parser.parse_args()

    try:
        # Extract file prefix and format from the output path
        file_prefix, audio_format = os.path.splitext(args.output_path)
        # Remove the leading dot from the extension
        if audio_format.startswith("."):
            audio_format = audio_format[1:]

        generate_audio(
            text=args.text,
            model_path=args.model_path,
            file_prefix=file_prefix,
            audio_format=audio_format,
            verbose=True,
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
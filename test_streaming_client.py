#!/usr/bin/env python3
"""
Test client for the streaming TTS endpoint.
Demonstrates how to consume the /tts/stream API and save audio chunks.
"""

import requests
import wave
import io
import sys
from pathlib import Path


def stream_tts_to_file(
    text: str,
    output_file: str = "streaming_output.wav",
    server_url: str = "https://k1ho193a7ifqs6-8004.proxy.runpod.net",
    voice_mode: str = "predefined",
    predefined_voice_id: str = None,
    language: str = "en",
    chunk_size: int = 25,
    first_chunk_size: int = 5,
    temperature: float = 0.8,
    print_metrics: bool = True,
):
    """
    Stream TTS audio from the server and save to a file.
    
    Args:
        text: Text to synthesize
        output_file: Output filename
        server_url: Base URL of the TTS server
        voice_mode: 'predefined' or 'clone'
        predefined_voice_id: Voice filename (for predefined mode)
        language: Language code (e.g., 'en', 'es', 'fr')
        chunk_size: Tokens per chunk
        first_chunk_size: Tokens in first chunk
        temperature: Generation temperature
        print_metrics: Whether to print metrics on server side
    """
    endpoint = f"{server_url}/tts/stream"
    
    # Build request payload
    payload = {
        "text": text,
        "voice_mode": voice_mode,
        "language": language,
        "chunk_size": chunk_size,
        "first_chunk_size": first_chunk_size,
        "temperature": temperature,
        "print_metrics": print_metrics,
        "output_format": "wav",
    }
    
    if voice_mode == "predefined" and predefined_voice_id:
        payload["predefined_voice_id"] = predefined_voice_id
    
    print(f"📡 Connecting to {endpoint}...")
    print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"🎤 Voice mode: {voice_mode}")
    print(f"🌍 Language: {language}")
    print(f"⚙️  Chunk sizes: first={first_chunk_size}, subsequent={chunk_size}")
    print()
    
    # Make streaming request
    try:
        with requests.post(endpoint, json=payload, stream=True) as response:
            response.raise_for_status()
            
            # Collect all chunks
            chunks = []
            chunk_count = 0
            total_bytes = 0
            
            print("🎵 Receiving audio chunks...")
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
                    chunk_count += 1
                    total_bytes += len(chunk)
                    
                    # Print progress
                    if chunk_count == 1:
                        print(f"   ⚡ First chunk received! ({len(chunk)} bytes)")
                    elif chunk_count % 5 == 0:
                        print(f"   📦 Chunk {chunk_count} ({len(chunk)} bytes, {total_bytes:,} total)")
            
            print()
            print(f"✅ Streaming complete!")
            print(f"   • Total chunks: {chunk_count}")
            print(f"   • Total bytes: {total_bytes:,}")
            
            # Save to file
            if chunks:
                output_path = Path(output_file)
                with open(output_path, 'wb') as f:
                    for chunk in chunks:
                        f.write(chunk)
                
                print(f"💾 Saved to: {output_path.absolute()}")
                print(f"   • File size: {output_path.stat().st_size:,} bytes")
                
                # Try to get audio duration
                try:
                    with wave.open(str(output_path), 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        duration = frames / float(rate)
                        print(f"   • Duration: {duration:.2f} seconds")
                        print(f"   • Sample rate: {rate} Hz")
                except Exception as e:
                    print(f"   ⚠️  Could not read WAV info: {e}")
            else:
                print("❌ No audio chunks received")
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    import sys
    import os
    
    # Get server URL from environment variable or command line
    server_url = os.getenv("TTS_SERVER_URL", "https://k1ho193a7ifqs6-8004.proxy.runpod.net")
    
    # Check if voice file was provided as command line argument
    if len(sys.argv) > 1:
        voice_file = sys.argv[1]
        print(f"Using voice file from command line: {voice_file}")
    else:
        # Prompt user for voice file
        print("=" * 70)
        print("🎙️  STREAMING TTS CLIENT TEST")
        print("=" * 70)
        print()
        print("⚠️  You need to specify a voice file to use.")
        print()
        print("Usage:")
        print(f"  python {sys.argv[0]} <voice_file.wav>")
        print()
        print("Example:")
        print(f"  python {sys.argv[0]} my_voice.wav")
        print()
        print("You can also set the server URL:")
        print(f"  TTS_SERVER_URL=https://your-server.com python {sys.argv[0]} my_voice.wav")
        print()
        print("Or edit the script and set the voice file directly.")
        print()
        sys.exit(1)
    
    # Check if custom server URL was provided as second argument
    if len(sys.argv) > 2:
        server_url = sys.argv[2]
        print(f"Using custom server URL: {server_url}")
    
    # Example usage
    examples = [
        {
            "text": "Hello! This is a test of the streaming text to speech system. "
                   "Notice how the audio starts playing almost immediately, even though "
                   "the full text hasn't been processed yet. This is the power of streaming!",
            "output_file": "streaming_test_short.wav",
            "language": "en",
        }
    ]
    
    # Run first example
    example = examples[0]
    
    print("=" * 70)
    print("🎙️  STREAMING TTS CLIENT TEST")
    print("=" * 70)
    print()
    
    stream_tts_to_file(
        text=example["text"],
        output_file=example["output_file"],
        language=example["language"],
        server_url=server_url,
        voice_mode="predefined",
        predefined_voice_id=voice_file,  # Use the voice file from command line
        chunk_size=25,
        first_chunk_size=5,
        temperature=0.8,
        print_metrics=True,
    )
    
    print()
    print("=" * 70)
    print("🎉 Test complete!")
    print("=" * 70)
    print()
    print("💡 Tips:")
    print("   • Adjust chunk_size for different latency/quality tradeoffs")
    print("   • Use first_chunk_size=5 for fastest initial response")
    print("   • Try different languages with the language parameter")
    print("   • Monitor server logs to see streaming metrics")


if __name__ == "__main__":
    main()
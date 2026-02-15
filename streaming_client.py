#!/usr/bin/env python3
"""
Diagnostic script to check TTS server status and available voices.
Helps troubleshoot connection and configuration issues.
"""

import sys
import requests
import json
from typing import Optional


def check_server_connection(server_url: str) -> bool:
    """Check if the server is reachable."""
    print(f"\n🔍 Checking server connection to {server_url}...")
    try:
        response = requests.get(f"{server_url}/", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Server is reachable (HTTP {response.status_code})")
            return True
        else:
            print(f"   ⚠️  Server responded but with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to server. Is it running?")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Connection timeout. Server might be slow or unreachable.")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_model_info(server_url: str) -> Optional[dict]:
    """Get model information from the server."""
    print(f"\n📊 Fetching model information...")
    try:
        response = requests.get(f"{server_url}/api/model-info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Model information retrieved:")
            print(f"      • Model loaded: {info.get('loaded', False)}")
            print(f"      • Model type: {info.get('type', 'unknown')}")
            print(f"      • Class name: {info.get('class_name', 'unknown')}")
            print(f"      • Device: {info.get('device', 'unknown')}")
            print(f"      • Sample rate: {info.get('sample_rate', 'unknown')} Hz")
            print(f"      • Supports streaming: {info.get('type') == 'multilingual'}")
            print(f"      • Supports languages: {info.get('supports_language_param', False)}")
            
            if info.get('type') != 'multilingual':
                print(f"\n   ⚠️  WARNING: Streaming requires 'multilingual' model!")
                print(f"       Current model type: {info.get('type')}")
                print(f"       Please update config.yaml to use 'chatterbox-multilingual'")
            
            return info
        else:
            print(f"   ❌ Failed to get model info (HTTP {response.status_code})")
            print(f"      Response: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def list_predefined_voices(server_url: str) -> list:
    """List available predefined voices."""
    print(f"\n🎤 Fetching available predefined voices...")
    try:
        response = requests.get(f"{server_url}/get_predefined_voices", timeout=10)
        if response.status_code == 200:
            voices = response.json()
            if voices:
                print(f"   ✅ Found {len(voices)} predefined voice(s):")
                for i, voice in enumerate(voices, 1):
                    display_name = voice.get('displayName', 'Unknown')
                    filename = voice.get('filename', 'Unknown')
                    print(f"      {i}. {display_name} (file: {filename})")
                return voices
            else:
                print(f"   ⚠️  No predefined voices found.")
                print(f"      Upload voice files to the predefined_voices directory.")
                return []
        else:
            print(f"   ❌ Failed to list voices (HTTP {response.status_code})")
            return []
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


def list_reference_audio(server_url: str) -> list:
    """List available reference audio files."""
    print(f"\n📁 Fetching available reference audio files...")
    try:
        response = requests.get(f"{server_url}/get_reference_files", timeout=10)
        if response.status_code == 200:
            files = response.json()
            if files:
                print(f"   ✅ Found {len(files)} reference audio file(s):")
                for i, filename in enumerate(files, 1):
                    print(f"      {i}. {filename}")
                return files
            else:
                print(f"   ⚠️  No reference audio files found.")
                return []
        else:
            print(f"   ❌ Failed to list reference audio (HTTP {response.status_code})")
            return []
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []


def test_streaming_endpoint(server_url: str, voice_file: Optional[str] = None) -> bool:
    """Test if the streaming endpoint is available and properly configured."""
    print(f"\n🧪 Testing streaming endpoint availability...")
    
    # If no voice file provided, try to get one from the server
    if not voice_file:
        voices = list_predefined_voices(server_url)
        if voices:
            voice_file = voices[0].get('filename')
            print(f"   Using first available voice: {voice_file}")
        else:
            print(f"   ⚠️  No voice file available for testing.")
            print(f"      Skipping endpoint test.")
            return False
    
    try:
        # Send a minimal test request
        payload = {
            "text": "Test",
            "voice_mode": "predefined",
            "predefined_voice_id": voice_file,
            "language": "en",
            "chunk_size": 25,
            "first_chunk_size": 5,
        }
        
        print(f"   Sending test request to /tts/stream...")
        response = requests.post(
            f"{server_url}/tts/stream",
            json=payload,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"   ✅ Streaming endpoint is working!")
            
            # Read first chunk to verify streaming works
            chunk_count = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunk_count += 1
                    if chunk_count == 1:
                        print(f"      • First chunk received: {len(chunk)} bytes")
                    if chunk_count >= 3:  # Just test first few chunks
                        break
            
            print(f"      • Total chunks tested: {chunk_count}")
            return True
            
        elif response.status_code == 503:
            print(f"   ❌ Streaming not supported (HTTP 503)")
            print(f"      Likely reason: Model is not multilingual type")
            try:
                error_detail = response.json()
                print(f"      Server message: {error_detail.get('detail', 'No details')}")
            except:
                print(f"      Response: {response.text[:200]}")
            return False
            
        elif response.status_code == 400:
            print(f"   ❌ Bad request (HTTP 400)")
            try:
                error_detail = response.json()
                print(f"      Server message: {error_detail.get('detail', 'No details')}")
            except:
                print(f"      Response: {response.text[:200]}")
            return False
            
        else:
            print(f"   ❌ Unexpected response (HTTP {response.status_code})")
            print(f"      Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timeout. Server might be processing slowly.")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def print_usage_instructions(voices: list, model_info: dict):
    """Print instructions for using the streaming client."""
    print("\n" + "=" * 70)
    print("📝 HOW TO USE THE STREAMING CLIENT")
    print("=" * 70)
    
    if not voices:
        print("\n⚠️  You need to upload voice files first!")
        print("   1. Upload .wav or .mp3 files to the server's predefined_voices directory")
        print("   2. Or use the web UI to upload files")
        print("   3. Then run this diagnostic script again")
        return
    
    if model_info and model_info.get('type') != 'multilingual':
        print("\n⚠️  Streaming requires the multilingual model!")
        print("   1. Edit config.yaml on the server")
        print("   2. Set model.repo_id to 'chatterbox-multilingual'")
        print("   3. Restart the server")
        print("   4. Run this diagnostic script again")
        return
    
    # Everything looks good, provide usage instructions
    voice_file = voices[0].get('filename')
    
    print("\n✅ Your server is ready for streaming!")
    print(f"\nRun the test client with:")
    print(f"\n   python test_streaming_client.py {voice_file}")
    
    if "runpod" in server_url or "https" in server_url:
        print(f"\nFor your RunPod server, use:")
        print(f'\n   python test_streaming_client.py {voice_file} "{server_url}"')
        print(f"\nOr set environment variable:")
        print(f'\n   export TTS_SERVER_URL="{server_url}"')
        print(f'   python test_streaming_client.py {voice_file}')
    
    print(f"\nAvailable voices you can use:")
    for voice in voices:
        print(f"   • {voice.get('filename')}")


def main():
    import os
    
    print("=" * 70)
    print("🔧 TTS SERVER DIAGNOSTIC TOOL")
    print("=" * 70)
    
    # Get server URL from command line or environment
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        server_url = os.getenv("TTS_SERVER_URL", "http://localhost:8000")
    
    print(f"\n🌐 Target server: {server_url}")
    
    # Run all diagnostic checks
    is_connected = check_server_connection(server_url)
    
    if not is_connected:
        print("\n" + "=" * 70)
        print("❌ DIAGNOSTIC FAILED")
        print("=" * 70)
        print("\nThe server is not reachable. Please check:")
        print("   1. Is the server running?")
        print("   2. Is the URL correct?")
        print("   3. Are there any firewall/network issues?")
        sys.exit(1)
    
    model_info = check_model_info(server_url)
    voices = list_predefined_voices(server_url)
    reference_files = list_reference_audio(server_url)
    
    # Test streaming endpoint if possible
    streaming_works = test_streaming_endpoint(server_url)
    
    # Print summary and instructions
    print("\n" + "=" * 70)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Server connected: Yes")
    print(f"{'✅' if model_info and model_info.get('loaded') else '❌'} Model loaded: {model_info.get('loaded') if model_info else 'Unknown'}")
    print(f"{'✅' if model_info and model_info.get('type') == 'multilingual' else '⚠️ '} Model type: {model_info.get('type') if model_info else 'Unknown'}")
    print(f"{'✅' if voices else '⚠️ '} Predefined voices: {len(voices)}")
    print(f"{'✅' if reference_files else '⚠️ '} Reference audio: {len(reference_files)}")
    print(f"{'✅' if streaming_works else '❌'} Streaming endpoint: {'Working' if streaming_works else 'Not working'}")
    
    # Print usage instructions
    print_usage_instructions(voices, model_info or {})
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
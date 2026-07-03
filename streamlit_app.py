"""
Streamlit UI for Audio Customer Support Agent Testing

This application provides a comprehensive interface to test all the endpoints
of the Audio Customer Support Agent API server.

Features:
- Text-based chat with the agent
- Audio recording and playback
- File upload for audio testing
- Health monitoring
- Debug endpoints for individual components
- Real-time API status checking

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
import time
import io
import wave
import numpy as np
from typing import Dict, Any, Optional
import base64
from datetime import datetime

# Audio recording imports
try:
    import sounddevice as sd
    AUDIO_RECORDING_AVAILABLE = True
except ImportError:
    AUDIO_RECORDING_AVAILABLE = False
    st.warning("Audio recording not available. Install sounddevice: `pip install sounddevice`")

# Configuration
DEFAULT_SERVER_URL = "http://localhost:8000"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "server_url" not in st.session_state:
        st.session_state.server_url = DEFAULT_SERVER_URL
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "server_status" not in st.session_state:
        st.session_state.server_status = "Unknown"
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None

def check_server_status(server_url: str) -> Dict[str, Any]:
    """Check if the API server is running and get health status."""
    try:
        # Check root endpoint
        response = requests.get(f"{server_url}/", timeout=5)
        if response.status_code == 200:
            root_info = response.json()
        else:
            root_info = {"status": "error"}
        
        # Check health endpoint
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code == 200:
            health_info = health_response.json()
        else:
            health_info = {"status": "unhealthy", "message": "Health endpoint failed"}
        
        return {
            "server_running": True,
            "root_info": root_info,
            "health_info": health_info
        }
    except requests.exceptions.RequestException as e:
        return {
            "server_running": False,
            "error": str(e)
        }

def send_text_message(server_url: str, text: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Send text message to the chat endpoint."""
    try:
        payload = {
            "text": text,
            "parameters": parameters or {}
        }
        
        response = requests.post(
            f"{server_url}/chat/text",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e)
        }

def send_audio_message(server_url: str, audio_data: bytes) -> Dict[str, Any]:
    """Send audio data to the audio chat endpoint."""
    try:
        files = {
            'audio': ('audio.wav', audio_data, 'audio/wav')
        }
        
        response = requests.post(
            f"{server_url}/chat/audio",
            files=files,
            timeout=60  # Longer timeout for audio processing
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "audio_data": response.content,
                "content_type": response.headers.get('content-type', 'audio/mpeg')
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e)
        }


def record_audio(sample_rate: int = AUDIO_SAMPLE_RATE) -> Optional[bytes]:
    """Record audio using sounddevice with user control."""
    if not AUDIO_RECORDING_AVAILABLE:
        st.error("Audio recording not available. Please install sounddevice.")
        return None
    
    try:
        # Simple 10-second recording - user can stop early by clicking stop
        duration = 10  # Maximum duration
        st.info(" Recording will last up to 10 seconds. Speak now!")
        
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=AUDIO_CHANNELS,
                           dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        
        # Convert to WAV bytes
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(AUDIO_CHANNELS)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def create_audio_player(audio_data: bytes, label: str = "Audio Response"):
    """Create an audio player widget for audio data."""
    if audio_data:
        st.audio(audio_data, format='audio/mpeg')
        
        # Provide download button
        st.download_button(
            label=f"Download {label}",
            data=audio_data,
            file_name=f"{label.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime="audio/mpeg"
        )

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Audio Support Agent Tester",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Main title
    st.title(" Audio Customer Support Agent Tester")
    st.markdown("### Test all endpoints of your Audio Support Agent implementation")
    
    # Sidebar configuration
    with st.sidebar:
        st.header(" Configuration")
        
        # Server URL configuration
        server_url = st.text_input(
            "API Server URL",
            value=st.session_state.server_url,
            help="URL of your FastAPI server"
        )
        st.session_state.server_url = server_url
        
        # Server status check
        if st.button("🔄 Check Server Status"):
            with st.spinner("Checking server..."):
                status = check_server_status(server_url)
                st.session_state.server_status = status
        
        # Display server status
        if st.session_state.server_status != "Unknown":
            status = st.session_state.server_status
            if isinstance(status, dict) and status.get("server_running"):
                st.success(" Server is running")
                health = status.get("health_info", {})
                if health.get("status") == "healthy":
                    st.success(" All components ready")
                elif health.get("status") == "unhealthy":
                    st.warning(f" {health.get('message', 'Some components not ready')}")
                else:
                    st.info(" Health status unknown")
            else:
                st.error(" Server not accessible")
                if status.get("error"):
                    st.error(f"Error: {status['error']}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Text Chat", 
        " Audio Chat", 
        " Health Monitor",
        " Documentation"
    ])
    
    with tab1:
        st.header(" Text Chat Interface")
        st.markdown("Test the text-based conversation with your customer support agent.")
        
        # Chat interface
        user_message = st.text_input(
            "Your message:",
            placeholder="Ask about returns, shipping, support, etc.",
            key="text_input"
        )
        
        send_text = st.button("Send Message", type="primary", key="send_text_btn")
        
        # Send message
        if send_text and user_message:
            # Send request
            with st.spinner("Sending message..."):
                result = send_text_message(server_url, user_message)
            
            # Add to chat history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "user": user_message,
                "result": result
            })
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader(" Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
                with st.container():
                    st.markdown(f"**[{chat['timestamp']}] You:** {chat['user']}")
                    
                    if chat['result']['success']:
                        response_data = chat['result']['data']
                        st.markdown(f"** Agent:** {response_data.get('response_text', 'No response')}")
                        
                        # Show metadata
                        if 'processing_time_ms' in response_data:
                            st.caption(f" Processing time: {response_data['processing_time_ms']}ms")
                        if 'audio_available' in response_data:
                            st.caption(f" Audio available: {response_data['audio_available']}")
                    else:
                        st.error(f" Error: {chat['result']['error']}")
                    
                    st.divider()
        
        # Clear chat history
        if st.button(" Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with tab2:
        st.header(" Audio Chat Interface")
        st.markdown("Test the complete audio pipeline: record audio, get audio response.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(" Audio Input")
            
            # Recording controls
            if AUDIO_RECORDING_AVAILABLE:
                if st.button(" Record Audio", key="start_recording", type="primary"):
                    with st.spinner("Recording for 10 seconds... Speak now!"):
                        st.session_state.audio_data = record_audio()
                    if st.session_state.audio_data:
                        st.success(" Recording completed!")
            else:
                st.warning(" Audio recording not available. Install sounddevice to record audio.")
            
            # Show recorded audio
            if st.session_state.audio_data:
                st.audio(st.session_state.audio_data, format='audio/wav')
            
            # File upload
            st.markdown("**Or upload an audio file:**")
            uploaded_file = st.file_uploader(
                "Choose audio file",
                type=['wav', 'mp3', 'ogg', 'flac'],
                help="Upload an audio file to test"
            )
            
            if uploaded_file:
                st.session_state.audio_data = uploaded_file.read()
                st.audio(st.session_state.audio_data)
        
        with col2:
            st.subheader(" Audio Output")
            
            # Send audio for processing
            if st.session_state.audio_data:
                if st.button(" Send Audio to Agent", type="primary"):
                    with st.spinner("Processing audio... This may take a while."):
                        result = send_audio_message(server_url, st.session_state.audio_data)
                    
                    if result['success']:
                        st.success(" Audio processed successfully!")
                        create_audio_player(result['audio_data'], "Agent Response")
                    else:
                        st.error(f" Error: {result['error']}")
            else:
                st.info("👆 Record or upload audio first")
    
    with tab3:
        st.header(" Health Monitor")
        st.markdown("Monitor the health and status of all pipeline components.")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh every 10 seconds")
        
        # Manual refresh
        if st.button("🔄 Refresh Status") or auto_refresh:
            with st.spinner("Checking component health..."):
                status = check_server_status(server_url)
                st.session_state.server_status = status
        
        # Display detailed status
        if st.session_state.server_status != "Unknown":
            status = st.session_state.server_status
            
            if isinstance(status, dict) and status.get("server_running"):
                # Server info
                st.subheader(" Server Information")
                root_info = status.get("root_info", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Server Status", " Running")
                with col2:
                    st.metric("API Version", root_info.get("version", "Unknown"))
                with col3:
                    st.metric("Last Check", datetime.now().strftime("%H:%M:%S"))
                
                # Component health
                st.subheader(" Component Health")
                health_info = status.get("health_info", {})
                components = health_info.get("components", {})
                
                col1, col2 = st.columns(2)
                with col1:
                    for component, is_healthy in components.items():
                        icon = "" if is_healthy else ""
                        st.metric(f"{component}", f"{icon} {'Ready' if is_healthy else 'Not Ready'}")
                
                with col2:
                    overall_status = health_info.get("status", "unknown")
                    status_icon = "" if overall_status == "healthy" else "" if overall_status == "unhealthy" else ""
                    st.metric("Overall Status", f"{status_icon} {overall_status.title()}")
                    
                    if health_info.get("message"):
                        st.info(health_info["message"])
            else:
                st.error(" Server not accessible")
                if status.get("error"):
                    st.error(f"Connection error: {status['error']}")
                
                st.markdown("""
                **Troubleshooting:**
                1. Make sure your FastAPI server is running
                2. Check the server URL is correct
                3. Verify the server is accessible from this machine
                4. Check for any firewall or network issues
                """)
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(1)  # Small delay to prevent too frequent updates
            st.rerun()
    
    with tab4:
        st.header(" Documentation & Help")
        st.markdown("Quick reference for using the Audio Support Agent Tester.")
        
        st.subheader(" Getting Started")
        st.markdown("""
        1. **Start your API server:**
           ```bash
           cd audio_support_agent
           python -m src.api.server
           ```
        
        2. **Check server status** in the sidebar
        
        3. **Try text chat** to test the LLM agent
        
        4. **Test audio processing** with the Audio Chat tab
        """)
        
        st.subheader(" API Endpoints Available")
        endpoints = {
            "GET /": "Root endpoint with API information",
            "GET /health": "Health check for all components",
            "POST /chat/text": "Text-based conversation with the agent",
            "POST /chat/audio": "Complete audio pipeline (STT → LLM → TTS)",
        }
        
        for endpoint, description in endpoints.items():
            st.markdown(f"- **`{endpoint}`**: {description}")
        
        st.subheader(" Audio Requirements")
        st.markdown("""
        **For audio recording:**
        - Install sounddevice: `pip install sounddevice`
        - Microphone access required
        - Supported formats: WAV (recommended), MP3, OGG, FLAC
        
        **For best results:**
        - Use quiet environment
        - Speak clearly into microphone
        - Keep recordings between 2-10 seconds
        - Test with different audio files
        """)
        
        st.subheader(" Troubleshooting")
        st.markdown("""
        **Common issues:**
        
        - **Server not responding**: Check if FastAPI server is running on correct port
        - **Audio recording fails**: Install sounddevice and check microphone permissions
        - **STT errors**: Verify your STT implementation and API keys
        - **TTS errors**: Check TTS service configuration and API keys
        - **LLM errors**: Ensure LLM service is properly configured
        
        **Tips:**
        - Check the Health Monitor tab for component status
        - Use Debug Tools to test components individually
        - Review server logs for detailed error messages
        - Try different audio formats if one doesn't work
        """)

if __name__ == "__main__":
    main()
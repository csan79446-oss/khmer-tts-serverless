import runpod
import base64
import io
import soundfile as sf
import numpy as np
from backend_server import generate_segment, SpeakerConfig, SAMPLE_RATE

def encode_to_base64(audio_array):
    """មុខងារបំលែង Numpy Array ទៅជា Base64 string"""
    if len(audio_array) == 0:
        return ""
    
    # បំលែងទៅ WAV format ក្នុង Memory
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, SAMPLE_RATE, format='WAV')
    buffer.seek(0)
    
    # បំលែងទៅ Base64
    return base64.b64encode(buffer.read()).decode('utf-8')

def handler(event):
    """Function ស្នូលសម្រាប់ RunPod Serverless"""
    try:
        job_input = event.get('input', {})
        text = job_input.get("text", "")
        config_data = job_input.get("config", {})

        # បង្កើត SpeakerConfig object
        config = SpeakerConfig(**config_data)

        # ហៅមុខងារផលិតសំឡេងពី backend_server
        final_wav = generate_segment(text, config)

        # បំលែងលទ្ធផលទៅជា Base64
        audio_b64 = encode_to_base64(final_wav)

        return {
            "status": "success",
            "audio_base64": audio_b64
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ចាប់ផ្តើមរង់ចាំការបញ្ជា
runpod.serverless.start({"handler": handler})
import runpod
from backend_server import process_tts_request

def handler(event):
    """Entry point សម្រាប់ RunPod Serverless"""
    try:
        # ទទួល input ពី RunPod
        input_data = event.get('input', {})
        
        # ហៅ logic ពី backend_server
        result = process_tts_request(input_data)
        
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ចាប់ផ្តើម RunPod
runpod.serverless.start({"handler": handler})

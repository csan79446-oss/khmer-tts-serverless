import os
import runpod
import torch
import base64
import io
import logging

# ១. ការកំណត់បច្ចេកទេសដើម្បីជៀសវាង Error "Failed to find C compiler"
os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.suppress_errors = True

# កំណត់ Logger ដើម្បីងាយស្រួលមើល Error ក្នុង RunPod Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ២. ផ្ទុក Model (Global variable ដើម្បីកុំឱ្យផ្ទុកឡើងវិញរាល់ពេលមាន Request)
model = None

def init():
    """Function នេះដំណើរការតែម្តងគត់ នៅពេល Server ចាប់ផ្តើម"""
    global model
    logger.info("កំពុងផ្ទុក Model...")
    
    # ជំនួសកន្លែងនេះដោយកូដផ្ទុក Model ពិតប្រាកដរបស់អ្នក
    # ឧទាហរណ៍: from voxcpm import VoxCPM; model = VoxCPM.from_pretrained(...)
    # model.to("cuda")
    
    logger.info("Model បានផ្ទុកជោគជ័យ!")

def handler(job):
    """Function នេះដំណើរការរាល់ពេលមាន Request ផ្ញើមក"""
    job_input = job.get("input", {})
    text = job_input.get("text", "សួស្តី")
    
    try:
        # ៣. ដំណើរការ Inference (កន្លែងបង្កើតសំឡេង)
        # ជំនួសកន្លែងនេះដោយកូដ Inference របស់អ្នក
        # audio_tensor = model.generate(text)
        # audio_buffer = io.BytesIO()
        # save_audio(audio_tensor, audio_buffer)
        
        # នេះគឺជា Dummy Data (ត្រូវជំនួសដោយទិន្នន័យពិត)
        logger.info(f"កំពុងបង្កើតសំឡេងសម្រាប់អត្ថបទ៖ {text}")
        
        # ៤. បំលែងសំឡេងទៅជា Base64 string
        # audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
        
        return {
            "output": {
                "audio_base64": "SGVsbG8sIHRoaXMgaXMgdGVzdCBhdWRpby4uLg==", # ជំនួសដោយ data ពិត
                "status": "success"
            }
        }
        
    except Exception as e:
        logger.error(f"មានបញ្ហាក្នុងការបង្កើតសំឡេង៖ {str(e)}")
        return {"error": str(e)}

# ៥. ចាប់ផ្តើម Server
runpod.serverless.start({"handler": handler, "init": init})

import os
import runpod
import torch
import base64
import io
import soundfile as sf
import logging

os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.suppress_errors = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def init():
    global model
    logger.info("កំពុងផ្ទុក Model...")
    # ជំនួស path នេះដោយ path ពិតរបស់ model អ្នក
    # from voxcpm import VoxCPM
    # model = VoxCPM.from_pretrained("path_to_model")
    # model.to("cuda")
    logger.info("Model បានផ្ទុកជោគជ័យ!")

def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    mode = job_input.get("mode", "preset")
    
    try:
        logger.info(f"កំពុងបង្កើតសំឡេង៖ {text}")
        
        # ជំនួសដោយកូដ Inference ពិតប្រាកដរបស់អ្នក
        # audio_tensor = model.generate(text)
        
        # នេះជា Dummy data ដើម្បីកុំឱ្យ Error ពេលមិនទាន់មាន Model
        audio_base64 = "SGVsbG8sIHRoaXMgaXMgdGVzdCBhdWRpby4uLg==" 
        
        return {
            "output": {
                "audio_base64": audio_base64,
                "status": "success"
            }
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler, "init": init})

import os
import runpod
import torch
import base64
import io
import soundfile as sf
import logging
import numpy as np
from huggingface_hub import hf_hub_download

# សន្មតថាកញ្ចប់ voxcpm ត្រូវបានដំឡើងតាមរយៈ requirements.txt រួចរាល់
from voxcpm import VoxCPM 

os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.suppress_errors = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

# Voice Design Mapping សម្រាប់សំឡេងនីមួយៗ (យោងតាមទម្រង់ការងារ VoxCPM2)
PRESET_MAPPING = {
    "[ប្រុស១]": "(A mature Cambodian man, clear, calm and professional voice)",
    "[ស្រី១]": "(A young Cambodian woman, gentle, sweet and clear voice)",
    "[ប្រុស២]": "(A young Cambodian man, energetic and clear voice)",
    "[ស្រី២]": "(A mature Cambodian woman, professional and confident voice)",
    "[ប្រុស៣]": "(A deep-voiced Cambodian man, formal and serious tone)",
    "[ស្រី៣]": "(A warm and friendly Cambodian woman's voice)"
}

def init():
    global model
    logger.info("⚡ កំពុងផ្ទុកម៉ូដែល AI (VoxCPM) ទៅកាន់ GPU...")
    try:
        # ប្តូរ "openbmb/VoxCPM2" ទៅជា Repo ID ឬ Path ពិតប្រាកដរបស់ម៉ូដែលអ្នក
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2") 
        model.to("cuda")
        logger.info("✅ ម៉ូដែល AI បានផ្ទុកនិងរៀបចំរួចរាល់នៅលើ CUDA (GPU)!")
    except Exception as e:
        logger.error(f"❌ បរាជ័យក្នុងការផ្ទុកម៉ូដែល: {str(e)}")
        raise e

def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text", "").strip()
    mode = job_input.get("mode", "preset")
    preset_name = job_input.get("preset_name", "[ប្រុស១]")
    ref_audio_name = job_input.get("ref_audio_name", "")
    speaker_map = job_input.get("speaker_map", {})
    
    if not text:
        return {"error": "សូមបញ្ចូលអត្ថបទអក្សរ!"}
        
    try:
        logger.info(f"🎙️ កំពុងផលិតសំឡេង... Mode: {mode} | អត្ថបទ: {text[:30]}...")

        # --- ទម្រង់ទី ១: Mode Preset (Voice Design) ---
        if mode == "preset":
            description = PRESET_MAPPING.get(preset_name, "(A clear Cambodian voice)")
            full_text = f"{description}{text}"
            wav = model.generate(full_text)

        # --- ទម្រង់ទី ២: Mode Clone (Zero-Shot Voice Cloning) ---
        elif mode == "clone":
            if not ref_audio_name:
                return {"error": "មិនមានឈ្មោះឯកសារសំឡេងគំរូឡើយ!"}
            # ទាញយកឯកសារសំឡេងគំរូពី Hugging Face មកទុកនៅលើ Server ជាបណ្តោះអាសន្ន
            local_ref_path = hf_hub_download(repo_id="Tha456/VoxCPM2", filename=ref_audio_name, repo_type="model")
            wav = model.generate(text=text, reference_audio=local_ref_path)

        # --- ទម្រង់ទី ៣: Mode SRT (បំបែកសំឡេងតាម Tag) ---
        elif mode == "srt":
            lines = text.split('\n')
            combined_wavs = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                current_tag = None
                for tag in PRESET_MAPPING.keys():
                    if line.startswith(tag):
                        current_tag = tag
                        break
                        
                if current_tag:
                    line_text = line[len(current_tag):].strip()
                    speaker_info = speaker_map.get(current_tag, {"mode": "preset", "preset_name": current_tag})
                else:
                    line_text = line
                    speaker_info = {"mode": "preset", "preset_name": "[ប្រុស១]"}
                    
                if not line_text:
                    continue
                    
                if speaker_info.get("mode") == "clone":
                    r_name = speaker_info.get("ref_audio_name")
                    local_ref_path = hf_hub_download(repo_id="Tha456/VoxCPM2", filename=r_name, repo_type="model")
                    line_wav = model.generate(text=line_text, reference_audio=local_ref_path)
                else:
                    p_name = speaker_info.get("preset_name", "[ប្រុស១]")
                    desc = PRESET_MAPPING.get(p_name, "(A clear voice)")
                    full_line_text = f"{desc}{line_text}"
                    line_wav = model.generate(full_line_text)
                    
                combined_wavs.append(line_wav)
                
            if combined_wavs:
                wav = np.concatenate(combined_wavs, axis=0)
            else:
                return {"error": "រកមិនឃើញអត្ថបទដែលមានទម្រង់ Tag ត្រឹមត្រូវទេ!"}
        else:
            return {"error": f"មិនស្គាល់ Mode: {mode}"}

        # --- បំប្លែង Audio Array ទៅជា Base64 ទម្រង់ WAV ---
        bytes_io = io.BytesIO()
        sample_rate = 24000  # កម្រិត Sample Rate លំនាំដើមរបស់ VoxCPM
        sf.write(bytes_io, wav, sample_rate, format='WAV')
        audio_base64 = base64.b64encode(bytes_io.getvalue()).decode('utf-8')
        
        logger.info("✅ ការផលិតសំឡេងជោគជ័យ ១០០%!")
        return {
            "output": {
                "audio_base64": audio_base64,
                "status": "success"
            }
        }
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler, "init": init})

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import io
import re
import uuid
import base64
import torch
import numpy as np
from scipy.io import wavfile
import runpod
import shutil
import threading  # 🔥 បន្ថែម Threading ដើម្បីការពារការគាំង Rollout
from huggingface_hub import snapshot_download

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from voxcpm import VoxCPM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_INSTANCE = None
PRESET_DIR = "./presets"
TEMP_DIR = "/tmp/runpod_tts"

# បង្កើត Variable សម្រាប់តាមដានស្ថានភាពផ្ទុកម៉ូដែល
MODEL_LOADING_STATUS = "not_started"  # "loading", "ready", "error"
MODEL_ERROR_MESSAGE = ""

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

def sync_presets_from_hf():
    try:
        print("-> កំពុងទាញយក Presets ពី Hugging Face...")
        snapshot_download(
            repo_id="Tha456/VoxCPM2",
            allow_patterns=["*.wav", "**/*.wav"], 
            local_dir=PRESET_DIR,
            local_dir_use_symlinks=False
        )
        print("-> ទាញយក Presets ជោគជ័យ!")
    except Exception as e:
        print(f"Warning: មិនអាចទាញយក Presets ពី HF បានទេ: {str(e)}")

def load_tts_model():
    global MODEL_INSTANCE
    if MODEL_INSTANCE is None:
        sync_presets_from_hf()
        try:
            print("-> កំពុងផ្ទុកម៉ូដែល VoxCPM ចូលទៅក្នុង RAM/GPU...")
            model_path = os.environ.get("MODEL_PATH", "Tha456/VoxCPM2")
            MODEL_INSTANCE = VoxCPM.from_pretrained(
                model_path, 
                load_denoiser=True,
                optimize=False
            )
            print("-> ផ្ទុកម៉ូដែល VoxCPM រួចរាល់ និងត្រៀមខ្លួនដំណើរការ!")
        except Exception as e:
            raise RuntimeError(f"ការផ្ទុកម៉ូដែល VoxCPM បរាជ័យ: {str(e)}")
    return MODEL_INSTANCE

def _bg_load_model_task():
    """រត់ការទាញយក និងផ្ទុកម៉ូដែលនៅលើ Background Thread"""
    global MODEL_LOADING_STATUS, MODEL_ERROR_MESSAGE
    MODEL_LOADING_STATUS = "loading"
    try:
        load_tts_model()
        MODEL_LOADING_STATUS = "ready"
    except Exception as e:
        MODEL_LOADING_STATUS = "error"
        MODEL_ERROR_MESSAGE = str(e)
        print(f"--- [Background Error] --- {MODEL_ERROR_MESSAGE}")

def cleanup_temp_files():
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
    except Exception:
        pass

def get_speaker_audio_path(preset_name=None):
    default_path = os.path.join(PRESET_DIR, "default.wav")
    if preset_name:
        clean_name = preset_name.replace(".wav", "")
        preset_path = os.path.join(PRESET_DIR, f"{clean_name}.wav")
        return preset_path if os.path.exists(preset_path) else default_path
    return default_path

def parse_srt_tags(text):
    pattern = r'\[([^\]]+)\]:\s*([^\[]+)'
    matches = re.findall(pattern, text)
    if not matches:
        return [("default", text.strip())]
    return [(m[0].strip(), m[1].strip()) for m in matches if m[1].strip()]

def run_tts_inference(model, text, speaker_wav_path):
    try:
        if speaker_wav_path and os.path.exists(speaker_wav_path):
            wav = model.generate(text=text, reference_wav_path=speaker_wav_path, cfg_value=2.0, inference_timesteps=10)
        else:
            wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
            
        sample_rate = getattr(model.tts_model, 'sample_rate', 48000)
        return sample_rate, wav
    except Exception as e:
        raise RuntimeError(f"បញ្ហាក្នុងការបង្កាត់សំឡេង: {str(e)}")

def handler(job):
    cleanup_temp_files()
    
    # 🔥 ពិនិត្យស្ថានភាពម៉ូដែលដែលកំពុងដោនឡូតនៅ Background
    if MODEL_LOADING_STATUS == "loading":
        return {"status": "error", "message": "ម៉ាស៊ីនមេកំពុងផ្ទុកម៉ូដែលជាលើកដំបូង (Cold Start) សូមព្យាយាមម្តងទៀតក្នុងរយៈពេល ១ នាទី។"}
    elif MODEL_LOADING_STATUS == "error":
        return {"status": "error", "message": f"ការផ្ទុកម៉ូដែលបានបរាជ័យនៅលើ Server: {MODEL_ERROR_MESSAGE}"}
        
    model = MODEL_INSTANCE 
    if model is None:
        return {"status": "error", "message": "ម៉ូដែលមិនទាន់ត្រូវបានដំឡើងរួចរាល់ឡើយ។"}
        
    job_input = job['input']
    mode = job_input.get("mode", "preset").lower()
    text = job_input.get("text", "")
    preset_name = job_input.get("preset_name", "default")
    ref_audio_name = job_input.get("ref_audio_name", "")
    speaker_map = job_input.get("speaker_map", {})
    
    if not text.strip():
        return {"status": "error", "message": "សូមបញ្ចូលអត្ថបទដែលត្រូវអាន។"}
        
    final_audio_segments = []
    global_sample_rate = 48000
    
    try:
        if mode in ["preset", "clone"]:
            target_name = preset_name if mode == "preset" else ref_audio_name
            speaker_wav = get_speaker_audio_path(preset_name=target_name)
            
            sr, audio_np = run_tts_inference(model, text, speaker_wav)
            global_sample_rate = sr
            final_audio_segments.append(audio_np)
            
        elif mode == "srt":
            segments = parse_srt_tags(text)
            for speaker, segment_text in segments:
                speaker_info = speaker_map.get(speaker, {})
                s_mode = speaker_info.get("mode", "preset").lower()
                target_name = speaker_info.get("preset_name", speaker) if s_mode == "preset" else speaker_info.get("ref_audio_name", speaker)
                
                speaker_wav = get_speaker_audio_path(preset_name=target_name)
                sr, segment_audio = run_tts_inference(model, segment_text, speaker_wav)
                global_sample_rate = sr
                final_audio_segments.append(segment_audio)
                
        if final_audio_segments:
            combined_audio = np.concatenate(final_audio_segments, axis=0)
            max_val = np.max(np.abs(combined_audio))
            if max_val > 0:
                combined_audio = combined_audio / max_val
            
            byte_io = io.BytesIO()
            wav_data = (combined_audio * 32767).astype(np.int16)
            wavfile.write(byte_io, global_sample_rate, wav_data)
            
            audio_base64 = base64.b64encode(byte_io.getvalue()).decode('utf-8')
            cleanup_temp_files()
            
            return {
                "status": "success",
                "mode": mode,
                "audio_base64": audio_base64,
                "format": "wav"
            }
        else:
            return {"status": "error", "message": "ការរៀបចំទិន្នន័យសំឡេងមិនបានសម្រេច។"}
            
    except Exception as e:
        cleanup_temp_files()
        return {"status": "error", "message": f"កំហុសប្រព័ន្ធដំណើរការ៖ {str(e)}"}

if __name__ == "__main__":
    print("--- [RunPod Startup] បើកដំណើរការ Background Thread ដើម្បីទាញយកម៉ូដែល... ---")
    
    # 🔥 បង្កើត Thread ឲ្យដោនឡូតម៉ូដែលដាច់ដោយឡែក កុំឲ្យទាក់ដំណើរការរបស់ RunPod
    bg_thread = threading.Thread(target=_bg_load_model_task)
    bg_thread.daemon = True
    bg_thread.start()
    
    print("--- [RunPod Startup] ចាប់ផ្តើមប្រព័ន្ធ Serverless ភ្លាមៗ (Rollout នឹងឡើង 100%) ---")
    runpod.serverless.start({"handler": handler})

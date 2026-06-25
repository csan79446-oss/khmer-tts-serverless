import os
import io
import re
import uuid
import base64
import torch
import numpy as np
from scipy.io import wavfile
import runpod
import shutil
from huggingface_hub import snapshot_download

# បិទ torch.compile ការពារ Cold Start Error
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from voxcpm import VoxCPM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_INSTANCE = None
PRESET_DIR = "./presets"
TEMP_DIR = "/tmp/runpod_tts"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

def sync_presets_from_hf():
    """ទាញយកសំឡេងគំរូទាំងអស់ពី Hugging Face មកទុកក្នុងម៉ាស៊ីន"""
    try:
        snapshot_download(
            repo_id="Tha456/VoxCPM2",
            allow_patterns=["*.wav", "**/*.wav"], 
            local_dir=PRESET_DIR,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Warning: មិនអាចទាញយក Presets ពី HF បានទេ: {str(e)}")

def load_tts_model():
    """ផ្ទុកម៉ូដែល VoxCPM"""
    global MODEL_INSTANCE
    if MODEL_INSTANCE is None:
        sync_presets_from_hf() # ត្រូវតែ Sync មុនពេលផ្ទុកម៉ូដែល
        try:
            model_path = os.environ.get("MODEL_PATH", "Tha456/VoxCPM2")
            MODEL_INSTANCE = VoxCPM.from_pretrained(
                model_path, 
                load_denoiser=True,
                optimize=False
            )
        except Exception as e:
            raise RuntimeError(f"ការផ្ទុកម៉ូដែល VoxCPM បរាជ័យ: {str(e)}")
    return MODEL_INSTANCE

def cleanup_temp_files():
    """សម្អាត Cache"""
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
    except Exception:
        pass

def get_speaker_audio_path(preset_name=None):
    """ស្វែងរកផ្លូវសំឡេងដោយស្វ័យប្រវត្តិ (ទាំង Preset និង Clone ពី Cloud គឺដូចគ្នា)"""
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
    
    try:
        model = load_tts_model()
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    job_input = job['input']
    
    # ចាប់យកទិន្នន័យពី Frontend ឱ្យត្រូវ Format ១០០%
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
            # ប្រើឈ្មោះពី preset_name ឬ ref_audio_name អាស្រ័យលើ Mode
            target_name = preset_name if mode == "preset" else ref_audio_name
            speaker_wav = get_speaker_audio_path(preset_name=target_name)
            
            sr, audio_np = run_tts_inference(model, text, speaker_wav)
            global_sample_rate = sr
            final_audio_segments.append(audio_np)
            
        elif mode == "srt":
            segments = parse_srt_tags(text)
            for speaker, segment_text in segments:
                # ទាញយកការកំណត់ពី UI (Speaker Map)
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
    runpod.serverless.start({"handler": handler})

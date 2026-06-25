import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import io
import re
import base64
import torch
import numpy as np
from scipy.io import wavfile
import runpod
from voxcpm import VoxCPM
import traceback 

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRESET_DIR = "./presets"

# ផ្នែកកែសម្រួលដើម្បីចាប់កំហុសនៅពេល Load Model
try:
    print("-> កំពុងផ្ទុកម៉ូដែល VoxCPM...")
    MODEL_INSTANCE = VoxCPM.from_pretrained(
        "Tha456/VoxCPM2", 
        load_denoiser=True,
        optimize=False
    )
    print("-> ផ្ទុកម៉ូដែលរួចរាល់ ១០០%!")
except Exception as e:
    print("!!! កំហុសធ្ងន់ធ្ងរពេលផ្ទុកម៉ូដែល !!!")
    traceback.print_exc() # បង្ហាញ Error លម្អិតក្នុង Log
    exit(1) # បញ្ឈប់កម្មវិធី ដើម្បីកុំឱ្យវា Restart loop ហើយអ្នកអាចឃើញ Error

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
    if speaker_wav_path and os.path.exists(speaker_wav_path):
        wav = model.generate(text=text, reference_wav_path=speaker_wav_path, cfg_value=2.0, inference_timesteps=10)
    else:
        wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10)
    sample_rate = getattr(model.tts_model, 'sample_rate', 48000)
    return sample_rate, wav

def handler(job):
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
            sr, audio_np = run_tts_inference(MODEL_INSTANCE, text, speaker_wav)
            global_sample_rate = sr
            final_audio_segments.append(audio_np)
            
        elif mode == "srt":
            segments = parse_srt_tags(text)
            for speaker, segment_text in segments:
                speaker_info = speaker_map.get(speaker, {})
                s_mode = speaker_info.get("mode", "preset").lower()
                target_name = speaker_info.get("preset_name", speaker) if s_mode == "preset" else speaker_info.get("ref_audio_name", speaker)
                
                speaker_wav = get_speaker_audio_path(preset_name=target_name)
                sr, segment_audio = run_tts_inference(MODEL_INSTANCE, segment_text, speaker_wav)
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
            return {
                "status": "success",
                "mode": mode,
                "audio_base64": audio_base64,
                "format": "wav"
            }
        else:
            return {"status": "error", "message": "ការរៀបចំទិន្នន័យសំឡេងមិនបានសម្រេច។"}
            
    except Exception as e:
        return {"status": "error", "message": f"កំហុសប្រព័ន្ធដំណើរការ៖ {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

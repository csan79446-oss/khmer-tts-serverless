import os
import io
import re
import uuid
import base64
import torch
import shutil
import requests
import numpy as np
from scipy.io import wavfile
import runpod

# --- ១. ការរៀបចំដំឡើង និងផ្ទុក Model (Global Init) ---
# ឧបមាថាប្រើប្រាស់គំរូ TTS ដូចជា XTTS, StyleTTS2 ឬ Coqui TTS
# (កូដផ្នែកនេះនឹងរក្សាដំណើរការដដែល តែធានាការផ្ទុកបានត្រឹមត្រូវ)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None

def load_tts_model():
    global MODEL
    if MODEL is None:
        print("----> 🚀 កំពុងចាប់ផ្តើមផ្ទុកម៉ូដែល AI ចូលទៅ VRAM...")
        # លុបជួរកូដ torch.compile(..., backend='inductor') ចោល ដើម្បីកុំឱ្យទាមទារ C Compiler
        # ឧទាហរណ៍៖ MODEL = TTSCore.load_checkpoint(...)
        # ជំនួសមកវិញនូវការដំណើរការល្បឿនធម្មតា ឬ Eager Mode
        MODEL = "INITIALIZED" 
        print("----> 🎉 ម៉ូដែល AI ត្រូវបានផ្ទុកដោយជោគជ័យ!")
    return MODEL

# បង្កើតតំបន់ផ្ទុកហ្វាយបណ្តោះអាសន្ន និងសម្អាតដើម្បីការពារឌីសពេញ
TEMP_DIR = "/tmp/runpod_tts"
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_files():
    """សម្អាតហ្វាយសំឡេងចាស់ៗចោល ដើម្បីការពារកុំឱ្យពេញឌីស (0.36GB Free)"""
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        print(print(f"Error cleaning Cache: {str(e)}"))

# --- ២. មុខងារស្នូលសម្រាប់ទាញយក លក្ខណៈសំឡេង (Speaker Embeddings) ---
def get_speaker_conditioning(mode, preset_name=None, reference_audio_b64=None):
    """
    ធានាការទាញយកលក្ខណៈសំឡេង (Voice Embeddings) មកប្រើប្រាស់ឱ្យបានត្រឹមត្រូវ ១០០%
    មិនឱ្យមានការឡូឡំសំឡេងគ្នាឡើយ។
    """
    # កំណត់ផ្លូវថតទុកសំឡេង Preset
    PRESET_DIR = "./presets"
    
    if mode == "Preset" and preset_name:
        preset_path = os.path.join(PRESET_DIR, f"{preset_name}.wav")
        if os.path.exists(preset_path):
            return preset_path # បញ្ជូនផ្លូវហ្វាយសំឡេងគំរូ Preset ទៅឱ្យ Model
        else:
            # បើរកមិនឃើញ យកសំឡេងលំនាំដើម (Default Preset)
            return os.path.join(PRESET_DIR, "default.wav")
            
    elif mode == "Clone" and reference_audio_b64:
        # បំប្លែងពី Base64 ទៅជាហ្វាយ .wav បណ្តោះអាសន្នដាច់ដោយឡែកសម្រាប់ Request នេះ
        unique_id = str(uuid.uuid4())
        temp_wave_path = os.path.join(TEMP_DIR, f"ref_{unique_id}.wav")
        
        with open(temp_wave_path, "wb") as fh:
            fh.write(base64.b64decode(reference_audio_b64))
        return temp_wave_path
        
    return os.path.join(PRESET_DIR, "default.wav")

def mock_tts_generate(text, speaker_ref_path):
    """
    នេះជាមុខងារសន្មតសម្រាប់ការដុតសំឡេងចេញពី Model (Core Inference)
    សូមបងជំនួសត្រង់កន្លែងនេះដោយមុខងារគណនារបស់ Model TTS ពិតប្រាកដរបស់បង
    (ឧទាហរណ៍៖ model.synthesize(text, speaker_ref_path))
    """
    # ក្នុងកូដពិតរបស់បង ត្រូវធានាថាបានបញ្ជូន speaker_ref_path ចូលទៅគ្រប់ទម្រង់
    # ដើម្បីកុំឱ្យវាប្តូរសំឡេងចៃដន្យ
    sr = 24000
    dummy_wav = np.zeros(int(sr * 2), dtype=np.float32) # សំឡេងគំរូ ២ វិនាទី
    return sr, dummy_wav

# --- ៣. មុខងារបំបែកអក្សរតាម Tag សម្រាប់រឿង SRT ---
def parse_srt_tags(text):
    """
    ស្វែងរក Tag សំឡេងនៅក្នុងអត្ថបទ ឧទាហរណ៍៖ [ពិសិដ្ឋ]: សួស្តីបង ឬ [ស្រីនា]: ចាសសួស្តី
    រួចបំបែកវាជាកញ្ចប់ៗ (Speaker, Text) ដើម្បីផលិតម្តងម្នាក់ៗជៀសវាងការច្រឡំសំឡេង។
    """
    # ស្វែងរកទម្រង់ [ឈ្មោះអ្នកនិយាយ]: អត្ថបទ
    pattern = r'\[([^\]]+)\]:\s*([^\[]+)'
    matches = re.findall(pattern, text)
    
    if not matches:
        # បើគ្មាន Tag ទេ ឱ្យអានជាអត្ថបទធម្មតាទាំងអស់ដោយប្រើសំឡេង Default
        return [("default", text.strip())]
    
    segments = []
    for match in matches:
        speaker = match[0].strip()
        segment_text = match[1].strip()
        if segment_text:
            segments.append((speaker, segment_text))
    return segments

# --- ៤. មុខងារចម្បងរបស់ RunPod Handler ---
def handler(job):
    # សម្អាត Cache ចាស់ៗមុនរត់ការងារថ្មី
    cleanup_temp_files()
    
    # ផ្ទុក Model (បើមិនទាន់បានផ្ទុក)
    load_tts_model()
    
    job_input = job['input']
    mode = job_input.get("mode", "Preset") # ជម្រើស៖ Preset, Clone, SRT
    text = job_input.get("text", "")
    preset_name = job_input.get("speaker_preset", "default")
    reference_audio = job_input.get("reference_audio", None) # ទម្រង់ Base64 String
    
    if not text:
        return {"error": "សូមបញ្ចូលអត្ថបទអាន (Text input is required)."}
    
    final_audio_segments = []
    sample_rate = 24000
    
    try:
        # --- ទម្រង់ទី ១ & ទី ២៖ អត្ថបទធម្មតា (Preset ឬ Clone) ---
        if mode in ["Preset", "Clone"]:
            # ទាញយកសំឡេងគំរូតែមួយគត់មកប្រើរហូតដល់ចប់អត្ថបទ
            speaker_ref = get_speaker_conditioning(mode, preset_name, reference_audio)
            
            # ផលិតសំឡេងចេញមក (ធានាថាប្រើលក្ខណៈសំឡេងតែមួយមិនប្រែប្រួល)
            sr, audio_data = mock_tts_generate(text, speaker_ref)
            sample_rate = sr
            final_audio_segments.append(audio_data)
            
        # --- ទម្រង់ទី ៣៖ អត្ថបទរឿង SRT (បំបែកសំឡេងតាម Tag) ---
        elif mode == "SRT":
            segments = parse_srt_tags(text)
            
            for speaker, seg_text in segments:
                # ស្វែងរកសំឡេងគំរូតាមឈ្មោះ Tag (អាចជាឈ្មោះ Preset ដូចជា 'ពិសិដ្ឋ', 'ស្រីនា')
                speaker_ref = get_speaker_conditioning("Preset", preset_name=speaker)
                
                # ផលិតសំឡេងដាច់ដោយឡែកសម្រាប់កថាខណ្ឌនីមួយៗ
                sr, seg_audio = mock_tts_generate(seg_text, speaker_ref)
                sample_rate = sr
                final_audio_segments.append(seg_audio)
        
        # រួបរួមរាល់បំណែកសំឡេងទាំងអស់ចូលគ្នាជាហ្វាយតែមួយ
        if final_audio_segments:
            combined_audio = np.concatenate(final_audio_segments, axis=0)
            
            # បំប្លែងលទ្ធផលទៅជា Base64 ដើម្បីផ្ញើត្រឡប់ទៅកម្មវិធីបញ្ជាវិញ
            byte_io = io.BytesIO()
            wavfile.write(byte_io, sample_rate, (combined_audio * 32767).astype(np.int16))
            audio_bytes = byte_io.getvalue()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # សម្អាតហ្វាយបណ្តោះអាសន្នក្រោយពេលធ្វើការរួចរាល់
            cleanup_temp_files()
            
            return {
                "status": "success",
                "mode": mode,
                "audio_base64": audio_base64,
                "format": "wav"
            }
        else:
            return {"error": "មិនអាចផលិតសំឡេងបានឡើយ។"}
            
    except Exception as e:
        cleanup_temp_files()
        return {"error": f"ការផលិតសំឡេងបរាជ័យ៖ {str(e)}"}

# ចាប់ផ្តើមដំណើរការប្រព័ន្ធ RunPod Serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

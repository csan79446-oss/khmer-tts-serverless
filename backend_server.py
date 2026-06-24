import os
import re
import io
import base64
import tempfile
import numpy as np
import soundfile as sf
import runpod
from pydantic import BaseModel
from typing import Optional
from voxcpm import VoxCPM

# បង្កើតប្រថាប់ (Global Variable) ទុកសម្រាប់ផ្ទុកម៉ូដែល
model = None
SAMPLE_RATE = 24000
init_error = None

NUMBERS_MAP = {
    '0': 'សូន្យ', '1': 'មួយ', '2': 'ពីរ', '3': 'បី', '4': 'បួន',
    '5': 'ប្រាំ', '6': 'ប្រាំមួយ', '7': 'ប្រាំពីរ', '8': 'ប្រាំបី', '9': 'ប្រាំបួន',
    '០': 'សូន្យ', '១': 'មួយ', '២': 'ពីរ', '៣': 'បី', '៤': 'បួន',
    '៥': 'ប្រាំ', '៦': 'ប្រាំមួយ', '៧': 'ប្រាំពីរ', '៨': 'ប្រាំបី', '៩': 'ប្រាំបួន'
}

def clean_khmer_text(raw_text: str) -> str:
    text = raw_text
    for digit, word in NUMBERS_MAP.items():
        text = text.replace(digit, word)
    text = re.sub(r'[^\u1780-\u17F9\sAlign។ៗ?!,a-zA-Z]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def srt_time_to_seconds(time_str: str) -> float:
    try:
        parts = time_str.replace(',', '.').split(':')
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0

class SpeakerConfig(BaseModel):
    mode: str = "preset"
    preset_name: Optional[str] = "[ប្រុស១]"
    ref_audio_base64: Optional[str] = None

# --- មុខងាររត់ដំបូងបង្អស់ដើម្បីផ្ទុកម៉ូដែល (Explicit Init) ---
def initialize_model():
    global model, SAMPLE_RATE, init_error
    print("⚙️ [STARTUP] កំពុងផ្ទុកម៉ូដែល VoxCPM2 ទៅកាន់ GPU...")
    try:
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        if hasattr(model, 'to'):
            model.to("cuda")
        SAMPLE_RATE = model.tts_model.sample_rate 
        print(f"✅ [SUCCESS] ម៉ូដែលរួចរាល់! {SAMPLE_RATE}Hz")
    except Exception as e:
        init_error = str(e)
        print(f"❌ [ERROR] បរាជ័យក្នុងការផ្ទុកម៉ូដែល: {init_error}")

def _generate_single_audio(text_chunk: str, temp_ref_path: str, fallback_ref_path: str) -> np.ndarray:
    kwargs = {
        "text": text_chunk,
        "cfg_value": 2.5,
        "inference_timesteps": 25,
        "normalize": True,
        "retry_badcase": True
    }
    
    ref_path = temp_ref_path if (temp_ref_path and os.path.exists(temp_ref_path)) else fallback_ref_path
    if ref_path and os.path.exists(ref_path):
        kwargs["reference_audio"] = ref_path 
        kwargs["reference_wav_path"] = ref_path 
    
    try:
        return model.generate(**kwargs)
    except Exception as gen_err:
        print(f"❌ កំហុសពេល Model កំពុងដំណើរការ៖ {str(gen_err)}")
        return np.array([], dtype=np.float32)

def generate_segment(text: str, config: SpeakerConfig) -> np.ndarray:
    clean_txt = clean_khmer_text(text)
    if not clean_txt or len(clean_txt.strip()) == 0:
        return np.array([], dtype=np.float32)
    
    temp_ref_path = None
    fallback_ref_path = None
    
    if config.mode == "clone" and config.ref_audio_base64 and len(config.ref_audio_base64.strip()) > 100:
        try:
            b64_str = config.ref_audio_base64.split(",")[1] if "," in config.ref_audio_base64 else config.ref_audio_base64
            audio_data, orig_sr = sf.read(io.BytesIO(base64.b64decode(b64_str)))
            if len(audio_data.shape) > 1: audio_data = np.mean(audio_data, axis=1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_data, orig_sr)
                temp_ref_path = tmp.name
        except Exception as e: 
            print(f"⚠️ Clone error: {e}")

    if not temp_ref_path:
        voice_name = re.sub(r'[\[\]]', '', config.preset_name or "ប្រុស១")
        preset_file = f"{voice_name}.wav"
        if os.path.exists(preset_file): 
            fallback_ref_path = preset_file
        else:
            fallback_files = [f for f in os.listdir('.') if f.endswith('.wav')]
            if fallback_files: fallback_ref_path = fallback_files[0]

    chunked_text = clean_txt.replace('។', '|').replace('\n', '|').replace(',', '|')
    sentences = [s.strip() for s in chunked_text.split('|') if s.strip()]
    
    master_audio_chunks = []
    silence_array = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        seg_wav = _generate_single_audio(sentence, temp_ref_path, fallback_ref_path)
        if len(seg_wav) > 0:
            master_audio_chunks.append(seg_wav)
            if i < len(sentences) - 1: 
                master_audio_chunks.append(silence_array)

    if temp_ref_path and os.path.exists(temp_ref_path): 
        os.remove(temp_ref_path)
        
    return np.concatenate(master_audio_chunks) if master_audio_chunks else np.array([], dtype=np.float32)

# --- មុខងារចម្បងរបស់ RunPod Worker ---
def handler(job):
    global model, init_error
    
    # ប្រសិនបើម៉ូដែលផ្ទុកមិនចូល GPU ផ្ញើសារប្រាប់ Client ភ្លាម
    if model is None:
        return {"error": f"ម៉ាស៊ីនមិនទាន់មានម៉ូដែល AI សម្រាប់ដំណើរការឡើយ! មូលហេតុពិតប្រាកដ៖ {init_error if init_error else 'ប្រព័ន្ធទាញយកមិនទាន់ចប់ ឬទំហំ GPU មិនគ្រាន់'}"}

    input_data = job.get("input", {})
    text = input_data.get("text", "")
    mode = input_data.get("mode", "preset")
    preset_name = input_data.get("preset_name", "[ប្រុស១]")
    ref_audio_base64 = input_data.get("ref_audio_base64", None)
    
    if not text:
        return {"error": "សូមបញ្ចូលអត្ថបទអក្សរ!"}

    try:
        if mode == "srt":
            blocks = re.split(r'\n\s*\n', text.strip())
            master_audio = []
            last_end_time = 0.0
            for block in blocks:
                lines = [l.strip() for l in block.split('\n') if l.strip()]
                if len(lines) < 3: continue
                time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if not time_match: continue
                start_sec = srt_time_to_seconds(time_match.group(1))
                content_line = " ".join(lines[2:])
                tag_match = re.match(r'^(\[.*?\])\s*(.*)', content_line)
                speaker_tag = tag_match.group(1) if tag_match else "default"
                speech_text = tag_match.group(2) if tag_match else content_line
                
                if start_sec > last_end_time:
                    master_audio.append(np.zeros(int((start_sec - last_end_time) * SAMPLE_RATE), dtype=np.float32))
                
                cfg = SpeakerConfig(mode="preset", preset_name=speaker_tag)
                seg_wav = generate_segment(speech_text, cfg)
                if len(seg_wav) > 0:
                    master_audio.append(seg_wav)
                    last_end_time = start_sec + (len(seg_wav) / SAMPLE_RATE)
            
            if master_audio:
                final_wav = np.concatenate(master_audio)
            else:
                return {"error": "រកមិនឃើញឃ្លាដែលមានទម្រង់ SRT ត្រឹមត្រូវទេ"}
        else:
            cfg = SpeakerConfig(mode=mode, preset_name=preset_name, ref_audio_base64=ref_audio_base64)
            final_wav = generate_segment(text, cfg)

        if len(final_wav) == 0: 
            return {"error": "ដំណើរការបរាជ័យ មិនអាចបង្កើតសំឡេងបានឡើយ (Model Returned Empty)"}
        
        # Normalize
        max_amp = np.max(np.abs(final_wav))
        if max_amp > 0: 
            final_wav = final_wav / max_amp
        
        # បំប្លែងទៅជា Base64
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, SAMPLE_RATE)
            out_b64 = base64.b64encode(open(tmp.name, 'rb').read()).decode('utf-8')
            os.remove(tmp.name)
        
        return {
            "output": {
                "audio_base64": out_b64,
                "status": "success"
            }
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------------
# 🔥 គន្លឹះដោះស្រាយ៖ បង្ខំឱ្យរត់មុខងារ Load Model ភ្លាមៗពេលកូដចាប់ផ្តើម
# -------------------------------------------------------------
initialize_model()

# ចាប់ផ្តើមស្តាប់ការងារពី RunPod (ដាក់តែ handler មួយបានហើយ)
runpod.serverless.start({"handler": handler})

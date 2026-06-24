import os
import re
import io
import sys
import base64
import tempfile
import traceback
import numpy as np
import soundfile as sf
import runpod
from pydantic import BaseModel
from typing import Optional
from voxcpm import VoxCPM

# កំណត់អថេរ Global សម្រាប់ប្រព័ន្ធ
model = None
SAMPLE_RATE = 24000
init_error_message = None

NUMBERS_MAP = {
    '0': 'សូន្យ', '1': 'មួយ', '2': 'ពីរ', '3': 'បី', '4': 'បួន',
    '5': 'ប្រាំ', '6': 'ប្រាំមួយ', '7': 'ប្រាំពីរ', '8': 'ប្រាំបី', '9': 'ប្រាំបួន',
    '០': 'សូន្យ', '១': 'មួយ', '២': 'ពីរ', '៣': 'បី', '៤': 'បួន',
    '៥': 'ប្រាំ', '៦': 'ប្រាំមួយ', '៧': 'ប្រាំពីរ', '៨': 'ប្រាំបី', '៩': 'ប្រាំបួន'
}

def clean_khmer_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text
    for digit, word in NUMBERS_MAP.items():
        text = text.replace(digit, word)
    text = re.sub(r'[^\u1780-\u17F9\s។ៗ?!,a-zA-Z]', '', text)
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

# --- មុខងារផ្ទុកម៉ូដែលជាមុន (Pre-loading Model) ---
def initialize_model_safely():
    global model, SAMPLE_RATE, init_error_message
    print("⚙️ [STARTUP] កំពុងចាប់ផ្តើមផ្ទុកម៉ូដែល VoxCPM2 ទៅលើ GPU...", flush=True)
    try:
        # ផ្ទុកម៉ូដែល AI
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        
        # បញ្ជូនទៅកាន់ CUDA (GPU) ប្រសិនបើមានទម្រង់ .to()
        if hasattr(model, 'to'):
            try:
                model.to("cuda")
                print("⚡ [GPU] បានប្តូរការរត់ទៅកាន់ CUDA ដោយជោគជ័យ។", flush=True)
            except Exception as gpu_err:
                print(f"⚠️ [GPU Warning] មិនអាចរត់លើ CUDA បានទេ: {gpu_err}។ ប្រព័ន្ធនឹងរត់លើ CPU ជំនួស។", flush=True)
        
        SAMPLE_RATE = getattr(model.tts_model, 'sample_rate', 24000)
        print(f"✅ [READY] ម៉ូដែលបានផ្ទុករួចរាល់! Sample Rate: {SAMPLE_RATE}Hz", flush=True)
    except Exception as e:
        error_trace = traceback.format_exc()
        init_error_message = f"{str(e)}\n{error_trace}"
        print(f"❌ [CRITICAL ERROR] បរាជ័យក្នុងការផ្ទុកម៉ូដែល AI: {init_error_message}", sys.stderr, flush=True)

def _generate_single_audio(text_chunk: str, temp_ref_path: str, fallback_ref_path: str) -> np.ndarray:
    if not text_chunk or len(text_chunk.strip()) == 0:
        return np.array([], dtype=np.float32)
        
    kwargs = {
        "text": text_chunk,
        "cfg_value": 2.5,
        "inference_timesteps": 25,
        "normalize": True,
        "retry_badcase": True
    }
    
    ref_path = temp_ref_path if (temp_ref_path and os.path.exists(temp_ref_path)) else fallback_ref_path
    if ref_path and os.path.exists(ref_path):
        # ការពារការខុសជំនាន់កូដ ( VoxCPM ខ្លះប្រើ reference_audio ខ្លះប្រើ reference_wav_path )
        kwargs["reference_audio"] = ref_path
        kwargs["reference_wav_path"] = ref_path
    
    try:
        return model.generate(**kwargs)
    except Exception as gen_err:
        print(f"❌ [MODEL GENERATION ERROR] កំហុសពេលកំពុងផលិតឃ្លា ({text_chunk}): {str(gen_err)}", flush=True)
        return np.array([], dtype=np.float32)

def generate_segment(text: str, config: SpeakerConfig) -> np.ndarray:
    clean_txt = clean_khmer_text(text)
    if not clean_txt or len(clean_txt.strip()) == 0:
        return np.array([], dtype=np.float32)
    
    temp_ref_path = None
    fallback_ref_path = None
    
    # ត្រៀមឯកសារសំឡេង Clone (Base64)
    if config.mode == "clone" and config.ref_audio_base64 and len(config.ref_audio_base64.strip()) > 100:
        try:
            b64_str = config.ref_audio_base64.split(",")[1] if "," in config.ref_audio_base64 else config.ref_audio_base64
            audio_data, orig_sr = sf.read(io.BytesIO(base64.b64decode(b64_str)))
            if len(audio_data.shape) > 1: 
                audio_data = np.mean(audio_data, axis=1)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_data, orig_sr)
                temp_ref_path = tmp.name
        except Exception as e: 
            print(f"⚠️ [CLONE WARNING] មិនអាចបំប្លែងសំឡេង Clone បានទេ: {e}", flush=True)

    # ត្រៀមឯកសារសំឡេង Preset
    if not temp_ref_path:
        voice_name = re.sub(r'[\[\]]', '', config.preset_name or "ប្រុស១")
        preset_file = f"{voice_name}.wav"
        if os.path.exists(preset_file): 
            fallback_ref_path = preset_file
        else:
            fallback_files = [f for f in os.listdir('.') if f.endswith('.wav')]
            if fallback_files: 
                fallback_ref_path = fallback_files[0]

    # កាត់ឃ្លាអត្ថបទដើម្បីផលិតម្តងមួយកង់ៗ
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
        try: os.remove(temp_ref_path)
        except: pass
        
    return np.concatenate(master_audio_chunks) if master_audio_chunks else np.array([], dtype=np.float32)

# --- មុខងារចម្បងសម្រាប់ដោះស្រាយការងារពី RunPod ---
def handler(job):
    global model, init_error_message
    
    # ការពារបញ្ហា NoneType Object Error
    if model is None:
        error_msg = f"ម៉ាស៊ីនមិនទាន់មានម៉ូដែល AI សម្រាប់ដំណើរការឡើយ! មូលហេតុពិត៖ {init_error_message if init_error_message else 'កំពុងទាញយក ឬទំហំ GPU មិនគ្រាន់'}"
        return {"error": error_msg, "audio_base64": "", "status": "error"}

    input_data = job.get("input", {})
    text = input_data.get("text", "")
    mode = input_data.get("mode", "preset")
    preset_name = input_data.get("preset_name", "[ប្រុស១]")
    ref_audio_base64 = input_data.get("ref_audio_base64", None)
    
    if not text or len(text.strip()) == 0:
        return {"error": "សូមបញ្ចូលអត្ថបទអក្សរខ្មែរ!", "audio_base64": "", "status": "error"}

    try:
        # ករណីដំណើរការឯកសារទម្រង់ SRT
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
                return {"error": "រកមិនឃើញឃ្លាដែលមានទម្រង់ SRT ត្រឹមត្រូវទេ!", "audio_base64": "", "status": "error"}
        
        # ករណីដំណើរការអត្ថបទធម្មតា ឬសំឡេង Clone
        else:
            cfg = SpeakerConfig(mode=mode, preset_name=preset_name, ref_audio_base64=ref_audio_base64)
            final_wav = generate_segment(text, cfg)

        # ការពារបញ្ហាផលិតបានឯកសារទទេ 72 Bytes
        if len(final_wav) == 0: 
            return {"error": "ដំណើរការបរាជ័យ ម៉ូដែល AI មិនអាចបង្កើតសំឡេងចេញពីអត្ថបទនេះបានទេ!", "audio_base64": "", "status": "error"}
        
        # ធ្វើឱ្យសំឡេងឮច្បាស់ស្មើគ្នា (Normalize Audio)
        max_amp = np.max(np.abs(final_wav))
        if max_amp > 0: 
            final_wav = final_wav / max_amp
        
        # រក្សាទុកជាឯកសារបណ្តោះអាសន្ន រួចបម្លែងជា Base64
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, final_wav, SAMPLE_RATE)
            with open(tmp.name, 'rb') as f:
                out_b64 = base64.b64encode(f.read()).decode('utf-8')
            try: os.remove(tmp.name)
            except: pass
        
        # បោះលទ្ធផលត្រឡប់ទៅវិញដោយរៀបចំគន្លឹះ (Keys) ឱ្យមានគ្រប់ទម្រង់ដើម្បីកុំឱ្យទាស់ជាមួយ App
        return {
            "audio_base64": out_b64,
            "status": "success",
            "output": {
                "audio_base64": out_b64,
                "status": "success"
            }
        }
        
    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}", "audio_base64": "", "status": "error"}

# =============================================================
# 🔥 ជំហានដោះស្រាយដ៏សំខាន់៖ ផ្ទុកម៉ូដែល AI ភ្លាមៗពេលដំណើរការ Script
# =============================================================
initialize_model_safely()

# ចាប់ផ្តើមដំណើរការប្រព័ន្ធរង់ចាំការងាររបស់ RunPod Serverless
runpod.serverless.start({"handler": handler})

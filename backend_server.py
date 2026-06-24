import os
import re
import io
import base64
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
from voxcpm import VoxCPM

app = FastAPI(title="Khmer TTS Comprehensive Production Engine (RunPod Optimized)")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Khmer TTS Server is running successfully!"}

# --- ផ្នែកទាញយក និងត្រៀមលក្ខណៈ Model ---
print("⚙️ កំពុងដំណើរការដំឡើង និងផ្ទៀងផ្ទាត់ម៉ូដែល VoxCPM2 លើ RunPod...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SAMPLE_RATE = model.tts_model.sample_rate 
print(f"✅ ម៉ូដែលរួចរាល់ ១០០%! ដំណើរការលើល្បឿនស្តង់ដារ៖ {SAMPLE_RATE}Hz")

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
    
    # ជម្រះអក្សរចម្លែកៗចេញ ប៉ុន្តែរក្សាសញ្ញាខណ្ឌ។ និងក្បៀសសម្រាប់បំបែកប្រយោគ
    text = re.sub(r'[^\u1780-\u17F9\s។ៗ?!,a-zA-Z]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def srt_time_to_seconds(time_str: str) -> float:
    try:
        parts = time_str.replace(',', '.').split(':')
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0

# --- ប្រព័ន្ធរចនាសម្ព័ន្ធទិន្នន័យ ---
class SpeakerConfig(BaseModel):
    mode: str = "preset"
    preset_name: Optional[str] = "[ប្រុស១]"
    ref_audio_base64: Optional[str] = None

class TTSRequest(BaseModel):
    text: str
    mode: str = "preset"
    preset_name: Optional[str] = "[ប្រុស១]"
    ref_audio_base64: Optional[str] = None
    speaker_map: Optional[Dict[str, SpeakerConfig]] = None

# --- មុខងារផលិតសំឡេងស្នូល (Core Generation Function) ---
def _generate_single_audio(text_chunk: str, temp_ref_path: str, fallback_ref_path: str) -> np.ndarray:
    """មុខងារសម្រាប់ Generate អត្ថបទខ្លីមួយកង់"""
    kwargs = {
        "text": text_chunk,
        "cfg_value": 2.5,               # រក្សាភាពច្បាស់នៃសំឡេង
        "inference_timesteps": 25,      # ត្រឹម 25 គឺគ្រប់គ្រាន់សម្រាប់គុណភាព និងល្បឿនដំណើរការ
        "normalize": True,
        "retry_badcase": True
    }
    
    if temp_ref_path and os.path.exists(temp_ref_path):
        kwargs["reference_wav_path"] = temp_ref_path
    elif fallback_ref_path and os.path.exists(fallback_ref_path):
        kwargs["reference_wav_path"] = fallback_ref_path
    
    try:
        return model.generate(**kwargs)
    except Exception as gen_err:
        print(f"❌ កំហុសពេល Model កំពុងដំណើរការ chunk នេះ៖ {str(gen_err)}")
        return np.array([], dtype=np.float32)

# --- មុខងារចម្អិនសំឡេងលំដាប់លម្អិត (កែបញ្ហាអានញាប់នៅទីនេះ) ---
def generate_segment(text: str, config: SpeakerConfig) -> np.ndarray:
    clean_txt = clean_khmer_text(text)
    if not clean_txt or len(clean_txt.strip()) == 0:
        return np.array([], dtype=np.float32)
        
    temp_ref_path = None
    fallback_ref_path = None
    
    # ត្រៀម Reference Audio
    if config.mode == "clone" and config.ref_audio_base64 and len(config.ref_audio_base64.strip()) > 100:
        try:
            b64_str = config.ref_audio_base64.split(",")[1] if "," in config.ref_audio_base64 else config.ref_audio_base64
            audio_data, orig_sr = sf.read(io.BytesIO(base64.b64decode(b64_str)))
            if len(audio_data.shape) > 1: 
                audio_data = np.mean(audio_data, axis=1) 
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_data, orig_sr)
                temp_ref_path = tmp.name
        except Exception as clone_err:
            print(f"⚠️ ការបំលែងសំឡេង Clone បរាជ័យ៖ {str(clone_err)}")

    if not temp_ref_path:
        voice_name = re.sub(r'[\[\]]', '', config.preset_name or "ប្រុស១")
        preset_file = f"{voice_name}.wav"
        if os.path.exists(preset_file):
            fallback_ref_path = preset_file
        else:
            fallback_files = [f for f in os.listdir('.') if f.endswith('.wav')]
            if fallback_files:
                fallback_ref_path = fallback_files[0]

    # 🛠️ ដំណោះស្រាយអានញាប់៖ បំបែកអត្ថបទជាកង់ៗ (Chunking) តាមសញ្ញាខណ្ឌ និងក្បៀស
    # ជំនួសសញ្ញាខណ្ឌដោយសញ្ញាបំបែកពិសេស ដើម្បីងាយស្រួល split
    chunked_text = clean_txt.replace('។', '|').replace('\n', '|').replace(',', '|')
    sentences = [s.strip() for s in chunked_text.split('|') if s.strip()]
    
    master_audio_chunks = []
    
    # បង្កើតចន្លោះស្ងាត់ (Silence) 0.5 វិនាទី សម្រាប់ការដកដង្ហើមរបស់ AI
    silence_duration_seconds = 0.5
    silence_array = np.zeros(int(silence_duration_seconds * SAMPLE_RATE), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        print(f"   -> កំពុងអានប្រយោគទី {i+1}: {sentence[:30]}...")
        seg_wav = _generate_single_audio(sentence, temp_ref_path, fallback_ref_path)
        
        if len(seg_wav) > 0:
            master_audio_chunks.append(seg_wav)
            # បញ្ចូលចន្លោះស្ងាត់ បើមិនមែនជាប្រយោគចុងក្រោយ
            if i < len(sentences) - 1:
                master_audio_chunks.append(silence_array)

    # លុបឯកសារបណ្តោះអាសន្ន
    if temp_ref_path and os.path.exists(temp_ref_path): 
        os.remove(temp_ref_path)

    if not master_audio_chunks:
        return np.array([], dtype=np.float32)
        
    return np.concatenate(master_audio_chunks)

# --- API Entry Point ---
@app.post("/generate")
async def generate_speech(req: TTSRequest):
    try:
        if req.mode == "srt":
            print("\n🎬 [SRT Mode] កំពុងវិភាគឯកសាររឿង និងចម្អិនសំឡេងតួអង្គ...")
            blocks = re.split(r'\n\s*\n', req.text.strip())
            master_audio = []
            last_end_time = 0.0
            
            for block in blocks:
                lines = [l.strip() for l in block.split('\n') if l.strip()]
                if len(lines) < 3: continue
                
                time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if not time_match: continue
                
                start_sec = srt_time_to_seconds(time_match.group(1))
                end_sec = srt_time_to_seconds(time_match.group(2))
                content_line = " ".join(lines[2:])
                
                tag_match = re.match(r'^(\[.*?\])\s*(.*)', content_line)
                speaker_tag = tag_match.group(1) if tag_match else "default"
                speech_text = tag_match.group(2) if tag_match else content_line
                
                if start_sec > last_end_time:
                    silence_duration = start_sec - last_end_time
                    silence_samples = np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.float32)
                    master_audio.append(silence_samples)
                
                cfg = SpeakerConfig()
                if req.speaker_map and speaker_tag in req.speaker_map:
                    cfg = req.speaker_map[speaker_tag]
                elif speaker_tag != "default":
                    cfg = SpeakerConfig(mode="preset", preset_name=speaker_tag)
                
                print(f"🗣️ តួអង្គ {speaker_tag} ៖ {speech_text[:25]}...")
                seg_wav = generate_segment(speech_text, cfg)
                if len(seg_wav) > 0:
                    master_audio.append(seg_wav)
                    actual_audio_duration = len(seg_wav) / SAMPLE_RATE
                    last_end_time = start_sec + actual_audio_duration
                else:
                    last_end_time = end_sec
                
            if not master_audio:
                raise HTTPException(status_code=400, detail="រចនាសម្ព័ន្ធអត្ថបទ SRT មិនត្រឹមត្រូវ!")
                
            final_wav = np.concatenate(master_audio)
            
        else:
            print(f"\n📝 [Normal Mode] ប្រភេទ៖ {req.mode}")
            default_cfg = SpeakerConfig(mode=req.mode, preset_name=req.preset_name, ref_audio_base64=req.ref_audio_base64)
            final_wav = generate_segment(req.text, default_cfg)

        if len(final_wav) == 0:
            raise HTTPException(status_code=500, detail="ការផលិតសំឡេងមិនទទួលបានលទ្ធផលអ្វីឡើយ។")

        # សម្រួលកម្រិតសំឡេងកុំឱ្យបែក (Normalize Volume)
        max_amp = np.max(np.abs(final_wav))
        if max_amp > 0: 
            final_wav = final_wav / max_amp

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, final_wav, SAMPLE_RATE)
            tmp_out.seek(0)
            out_b64 = base64.b64encode(tmp_out.read()).decode('utf-8')
            tmp_out_path = tmp_out.name
            
        if os.path.exists(tmp_out_path): 
            os.remove(tmp_out_path)
            
        return {"status": "success", "audio_base64": out_b64}

    except Exception as e:
        print(f"❌ Server Error Log: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n🚀 [RUNPOD ENGINE] កំពុងបើកដំណើរការ Server លើ Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
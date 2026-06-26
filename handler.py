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
from huggingface_hub import hf_hub_download

app = FastAPI(title="Khmer TTS Comprehensive Production Engine (RunPod Optimized)")

# --- មុខងារសម្រាប់ទាញយកសំឡេងពី Hugging Face និងរក្សាទុកក្នុង Cache ---
def prepare_reference_audio(filename: str) -> str:
    """ទាញយកឯកសារពី Hugging Face មកទុកក្នុង Local Storage ដើម្បីកុំឱ្យទាញញឹកញាប់"""
    local_dir = "/workspace/reference_audio_cache"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    # បើមិនទាន់មានក្នុងម៉ាស៊ីន ទើបធ្វើការទាញយក
    if not os.path.exists(local_path):
        print(f"📥 កំពុងទាញយក {filename} ពី Hugging Face...")
        hf_hub_download(
            repo_id="Tha456/VoxCPM2", 
            filename=filename, 
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    return local_path

# --- មុខងារថ្មី៖ រក្សាទុក base64 audio ទៅ local file ---
def save_base64_audio(b64_string: str, filename: str) -> str:
    """រក្សាទុក base64 audio string ទៅជាឯកសារ .wav"""
    if not b64_string:
        return None
    local_dir = "/workspace/reference_audio_cache"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)

    try:
        audio_bytes = base64.b64decode(b64_string)
        with open(local_path, "wb") as f:
            f.write(audio_bytes)
        return local_path
    except Exception as e:
        print(f"⚠️ ការរក្សាទុក base64 audio បរាជ័យ៖ {str(e)}")
        return None

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
    text = re.sub(r'[^\u1780-\u17F9\sa-zA-Z]', '', text)
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
    ref_audio_name: Optional[str] = None  # សម្រាប់ Hugging Face filename
    ref_audio_base64: Optional[str] = None  # ✅ បន្ថែម៖ សម្រាប់ base64 ពី Frontend

class TTSRequest(BaseModel):
    text: str
    mode: str = "preset"
    preset_name: Optional[str] = "[ប្រុស១]"
    ref_audio_name: Optional[str] = None
    ref_audio_base64: Optional[str] = None  # ✅ បន្ថែម
    speaker_map: Optional[Dict[str, SpeakerConfig]] = None

# --- មុខងារផលិតសំឡេងស្នូល ---
def _generate_single_audio(text_chunk: str, ref_wav_path: str) -> np.ndarray:
    kwargs = {
        "text": text_chunk,
        "cfg_value": 2.5,
        "inference_timesteps": 25,
        "normalize": True,
        "retry_badcase": True
    }

    if ref_wav_path and os.path.exists(ref_wav_path):
        kwargs["reference_wav_path"] = ref_wav_path

    try:
        return model.generate(**kwargs)
    except Exception as gen_err:
        print(f"❌ កំហុសពេល Model កំពុងដំណើរការ៖ {str(gen_err)}")
        return np.array([], dtype=np.float32)

# --- មុខងារថ្មី៖ ទទួលបាន path នៃ reference audio ---
def get_reference_path(config: SpeakerConfig) -> str:
    """
    ទទួលបាន path នៃ reference audio ពី config
    គាំទ្រទាំង filename (Hugging Face) និង base64 (Frontend upload)
    """
    ref_wav_path = None

    # ពិនិត្យ base64 ជាមុន (ពី Frontend)
    if config.ref_audio_base64:
        temp_filename = f"uploaded_{hash(config.ref_audio_base64) % 10000000}.wav"
        ref_wav_path = save_base64_audio(config.ref_audio_base64, temp_filename)
        if ref_wav_path:
            print(f"📁 បានរក្សាទុក base64 audio ទៅ៖ {ref_wav_path}")
            return ref_wav_path

    # ពិនិត្យ filename (Hugging Face)
    if config.ref_audio_name:
        try:
            ref_wav_path = prepare_reference_audio(config.ref_audio_name)
            if os.path.exists(ref_wav_path):
                return ref_wav_path
        except Exception as e:
            print(f"⚠️ ការទាញយកសំឡេង Clone បរាជ័យ៖ {str(e)}")

    return None

# --- មុខងារចម្អិនសំឡេងលំដាប់លម្អិត ---
def generate_segment(text: str, config: SpeakerConfig) -> np.ndarray:
    clean_txt = clean_khmer_text(text)
    if not clean_txt or len(clean_txt.strip()) == 0:
        return np.array([], dtype=np.float32)

    # ទទួលបាន reference audio path
    ref_wav_path = get_reference_path(config)

    # ត្រៀម Preset Audio ប្រសិនបើមិនមាន clone audio
    if not ref_wav_path:
        voice_name = re.sub(r'[\[\]]', '', config.preset_name or "ប្រុស១")
        preset_file = f"{voice_name}.wav"
        if os.path.exists(preset_file):
            ref_wav_path = preset_file
        else:
            fallback_files = [f for f in os.listdir('.') if f.endswith('.wav')]
            if fallback_files:
                ref_wav_path = fallback_files[0]

    # បំបែកអត្ថបទជាកង់ៗ
    chunked_text = clean_txt.replace('។', '|').replace('\n', '|').replace(',', '|')
    sentences = [s.strip() for s in chunked_text.split('|') if s.strip()]

    master_audio_chunks = []
    silence_array = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        print(f"   -> កំពុងអានប្រយោគទី {i+1}: {sentence[:30]}...")
        seg_wav = _generate_single_audio(sentence, ref_wav_path)

        if len(seg_wav) > 0:
            master_audio_chunks.append(seg_wav)
            if i < len(sentences) - 1:
                master_audio_chunks.append(silence_array)

    if not master_audio_chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(master_audio_chunks)

# --- API Entry Point ---
@app.post("/generate")
async def generate_speech(req: TTSRequest):
    try:
        if req.mode == "srt":
            print("\n🎬 [SRT Mode] កំពុងវិភាគឯកសាររឿង...")
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

                # ✅ បន្ថែម base64 support សម្រាប់ speaker_map
                if req.speaker_map and speaker_tag in req.speaker_map:
                    speaker_cfg = req.speaker_map[speaker_tag]
                    cfg.ref_audio_base64 = speaker_cfg.ref_audio_base64
                    cfg.ref_audio_name = speaker_cfg.ref_audio_name

                seg_wav = generate_segment(speech_text, cfg)
                if len(seg_wav) > 0:
                    master_audio.append(seg_wav)
                    actual_audio_duration = len(seg_wav) / SAMPLE_RATE
                    last_end_time = start_sec + actual_audio_duration
                else:
                    last_end_time = end_sec

            # ✅ ការពារ master_audio ទទេ
            if not master_audio:
                raise HTTPException(status_code=500, detail="មិនមានសំឡេងដែលបានបង្កើត។")
            final_wav = np.concatenate(master_audio)

        else:
            print(f"\n📝 [Normal Mode] ប្រភេទ៖ {req.mode}")
            default_cfg = SpeakerConfig(
                mode=req.mode, 
                preset_name=req.preset_name, 
                ref_audio_name=req.ref_audio_name,
                ref_audio_base64=req.ref_audio_base64  # ✅ បន្ថែម base64 support
            )
            final_wav = generate_segment(req.text, default_cfg)

        if len(final_wav) == 0:
            raise HTTPException(status_code=500, detail="ការផលិតសំឡេងបរាជ័យ។")

        # Normalize Volume
        max_amp = np.max(np.abs(final_wav))
        if max_amp > 0: 
            final_wav = final_wav / max_amp

        # ✅ កែបញ្ហា base64 - សរសេរទៅ file ហើយអានពី file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            tmp_path = tmp_out.name

        # សរសេរទិន្នន័យសំឡេងទៅ file
        sf.write(tmp_path, final_wav, SAMPLE_RATE)

        # អានពី file ហើយ encode base64
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
            out_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # លុប file បណ្តោះអាសន្ន
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {"status": "success", "audio_base64": out_b64}

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n🚀 [RUNPOD ENGINE] កំពុងបើកដំណើរការ Server លើ Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

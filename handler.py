import os
import re
import base64
import tempfile
import numpy as np
import soundfile as sf
import runpod
import torch

# ✅ Import voxcpm (official package!)
from voxcpm import VoxCPM
from huggingface_hub import hf_hub_download

# ==========================================
# ✅ Device setup
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 PyTorch CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔥 CUDA Version: {torch.version.cuda}")
    print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.backends.cudnn.benchmark = True
else:
    print("⚠️ កំពុងដំណើរការលើ CPU!")

# ==========================================
# ✅ Load VoxCPM2 Model
# ==========================================
print("⚙️ កំពុងដំឡើង VoxCPM2 Model...")
REPO_ID = "openbmb/VoxCPM2"  # ✅ Official repo!

try:
    # ✅ ប្រើ VoxCPM.from_pretrained ត្រឹមត្រូវ
    model = VoxCPM.from_pretrained(
        REPO_ID,
        load_denoiser=False,
        device=DEVICE
    )
    print(f"✅ Model loaded successfully!")
    print(f"✅ Running on: {DEVICE}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

SAMPLE_RATE = model.tts_model.sample_rate
print(f"✅ Sample Rate: {SAMPLE_RATE}Hz")

# ==========================================
# Helper functions
# ==========================================
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
    text = re.sub(r'[^\u1780-\u17F9\s។ៗ?!,a-zA-Z]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def srt_time_to_seconds(time_str: str) -> float:
    try:
        parts = time_str.replace(',', '.').split(':')
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0

def prepare_reference_audio(filename: str) -> str:
    local_dir = "/workspace/reference_audio_cache"
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    
    if not os.path.exists(local_path):
        print(f"📥 កំពុងទាញយក {filename}...")
        try:
            hf_hub_download(repo_id="Tha456/VoxCPM2", filename=filename, local_dir=local_dir, local_dir_use_symlinks=False)
        except Exception as e:
            print(f"⚠️ ការទាញយកបរាជ័យ: {e}")
            return None
    
    if filename.lower().endswith(".mp3"):
        wav_path = local_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(local_path)
                audio.export(wav_path, format="wav")
            except:
                return local_path
        return wav_path
    return local_path

def _generate_single_audio(text_chunk: str, ref_wav_path: str = None) -> np.ndarray:
    try:
        kwargs = {
            "text": text_chunk,
            "cfg_value": 2.5,
            "inference_timesteps": 25,
            "normalize": True,
            "retry_badcase": True
        }
        
        if ref_wav_path and os.path.exists(ref_wav_path):
            kwargs["reference_wav_path"] = ref_wav_path
        
        return model.generate(**kwargs)
    except Exception as gen_err:
        print(f"❌ កំហុស: {str(gen_err)}")
        return np.array([], dtype=np.float32)

def generate_segment(text: str, config: dict) -> np.ndarray:
    clean_txt = clean_khmer_text(text)
    if not clean_txt or len(clean_txt.strip()) == 0:
        return np.array([], dtype=np.float32)
        
    ref_wav_path = None
    temp_wav_file = None
    
    mode = config.get("mode", "preset")
    ref_audio_name = config.get("ref_audio_name")
    ref_audio_base64 = config.get("ref_audio_base64")
    preset_name = config.get("preset_name", "[ប្រុស១]")
    
    if mode == "clone":
        if ref_audio_base64:
            try:
                audio_bytes = base64.b64decode(ref_audio_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    ref_wav_path = tmp.name
                    temp_wav_file = tmp.name
            except Exception as e:
                print(f"⚠️ Base64 decode បរាជ័យ: {e}")
        elif ref_audio_name:
            ref_wav_path = prepare_reference_audio(ref_audio_name)

    if not ref_wav_path:
        voice_name = re.sub(r'[\[\]]', '', preset_name)
        preset_file = f"{voice_name}.wav"
        ref_wav_path = prepare_reference_audio(preset_file)
        
        if not ref_wav_path:
            fallback_files = [f for f in os.listdir('/workspace/reference_audio_cache') if f.endswith('.wav')]
            if fallback_files:
                ref_wav_path = os.path.join('/workspace/reference_audio_cache', fallback_files[0])

    chunked_text = clean_txt.replace('។', '|').replace('\n', '|').replace(',', '|')
    sentences = [s.strip() for s in chunked_text.split('|') if s.strip()]
    
    master_audio_chunks = []
    silence_array = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        print(f"   -> ប្រយោគទី {i+1}: {sentence[:30]}...")
        seg_wav = _generate_single_audio(sentence, ref_wav_path)
        
        if len(seg_wav) > 0:
            master_audio_chunks.append(seg_wav)
            if i < len(sentences) - 1:
                master_audio_chunks.append(silence_array)

    if temp_wav_file and os.path.exists(temp_wav_file):
        try:
            os.remove(temp_wav_file)
        except:
            pass

    if not master_audio_chunks:
        return np.array([], dtype=np.float32)
        
    return np.concatenate(master_audio_chunks)

# ==========================================
# RunPod Handler
# ==========================================
def handler(job):
    try:
        job_input = job.get("input", {})
        
        req_text = job_input.get("text", "")
        req_mode = job_input.get("mode", "preset")
        req_preset_name = job_input.get("preset_name", "[ប្រុស១]")
        req_ref_audio_name = job_input.get("ref_audio_name")
        req_speaker_map = job_input.get("speaker_map", {})

        if not req_text:
            return {"error": "សូមបញ្ចូលអត្ថបទ (text) នៅក្នុង input។"}

        if req_mode == "srt":
            blocks = re.split(r'\n\s*\n', req_text.strip())
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
                
                cfg = {"mode": "preset", "preset_name": "[ប្រុស១]"}
                if req_speaker_map and speaker_tag in req_speaker_map:
                    cfg = req_speaker_map[speaker_tag]
                elif speaker_tag != "default":
                    cfg = {"mode": "preset", "preset_name": speaker_tag}
                
                seg_wav = generate_segment(speech_text, cfg)
                
                if len(seg_wav) > 0:
                    master_audio.append(seg_wav)
                    actual_audio_duration = len(seg_wav) / SAMPLE_RATE
                    last_end_time = start_sec + actual_audio_duration
                else:
                    last_end_time = end_sec
            
            if not master_audio:
                return {"error": "ការផលិតសំឡេង SRT បរាជ័យ។"}
            final_wav = np.concatenate(master_audio)
            
        else:
            default_cfg = {
                "mode": req_mode, 
                "preset_name": req_preset_name, 
                "ref_audio_name": req_ref_audio_name,
                "ref_audio_base64": job_input.get("ref_audio_base64")
            }
            final_wav = generate_segment(req_text, default_cfg)

        if len(final_wav) == 0:
            return {"error": "ការផលិតសំឡេងបរាជ័យ។"}

        max_amp = np.max(np.abs(final_wav))
        if max_amp > 0: final_wav = final_wav / max_amp

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            sf.write(tmp_out.name, final_wav, SAMPLE_RATE)
            tmp_out.seek(0)
            out_b64 = base64.b64encode(tmp_out.read()).decode('utf-8')
            tmp_out_path = tmp_out.name
            
        if os.path.exists(tmp_out_path): os.remove(tmp_out_path)
        
        return {"status": "success", "audio_base64": out_b64}

    except Exception as e:
        import traceback
        print(f"❌ Handler Error: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

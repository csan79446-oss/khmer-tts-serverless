import os
import re
import base64
import tempfile
import numpy as np
import soundfile as sf
import runpod
from pydub import AudioSegment
from voxcpm import VoxCPM
from huggingface_hub import hf_hub_download

# --- មុខងារសម្រាប់ទាញយកសំឡេងពី Hugging Face និងរក្សាទុកក្នុង Cache ---
def prepare_reference_audio(filename: str) -> str:
    local_dir = "/workspace/reference_audio_cache"
    os.makedirs(local_dir, exist_ok=True)
    
    local_path = os.path.join(local_dir, filename)
    
    if not os.path.exists(local_path):
        print(f"📥 កំពុងទាញយក {filename} ពី Hugging Face...")
        hf_hub_download(
            repo_id="Tha456/VoxCPM2", 
            filename=filename, 
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    
    # --- ការដោះស្រាយ MP3 ទៅជា WAV ---
    if filename.lower().endswith(".mp3"):
        wav_path = local_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            print(f"🔄 កំពុងបម្លែង {filename} ទៅជា WAV...")
            audio = AudioSegment.from_mp3(local_path)
            audio.export(wav_path, format="wav")
        return wav_path
        
    return local_path

# --- ផ្នែកទាញយក និងត្រៀមលក្ខណៈ Model ---
print("⚙️ កំពុងដំណើរការដំឡើង និងផ្ទៀងផ្ទាត់ម៉ូដែល VoxCPM2 លើ RunPod Serverless...")
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
    text = re.sub(r'[^\u1780-\u17F9\s។ៗ?!,a-zA-Z]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def srt_time_to_seconds(time_str: str) -> float:
    try:
        parts = time_str.replace(',', '.').split(':')
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0

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

def generate_segment(text: str, config: dict) -> np.ndarray:
    clean_txt = clean_khmer_text(text)
    if not clean_txt or len(clean_txt.strip()) == 0:
        return np.array([], dtype=np.float32)
        
    ref_wav_path = None
    tmp_b64_audio_path = None # អញ្ញត្តិសម្រាប់រក្សាទុកទីតាំង File បណ្តោះអាសន្ន ដើម្បីលុបចោលវិញ
    
    mode = config.get("mode", "preset")
    ref_audio_name = config.get("ref_audio_name")
    ref_audio_base64 = config.get("ref_audio_base64") # ចាប់យកទិន្នន័យ Base64 ដែល Frontend ផ្ញើមក
    preset_name = config.get("preset_name", "[ប្រុស១]")
    
    # ១. ករណី Clone ដោយប្រើសំឡេងដែល Upload ពី Frontend (បំប្លែង Base64 ទៅ File)
    if mode == "clone" and ref_audio_base64:
        try:
            audio_bytes = base64.b64decode(ref_audio_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                ref_wav_path = tmp.name
                tmp_b64_audio_path = tmp.name
        except Exception as e:
            print(f"⚠️ ការបំប្លែង Base64 បរាជ័យ៖ {str(e)}")

    # ២. ករណី Clone ដោយទាញយកពី Hugging Face (ប្រើឈ្មោះហ្វាល់)
    elif mode == "clone" and ref_audio_name:
        try:
            ref_wav_path = prepare_reference_audio(ref_audio_name)
        except Exception as e:
            print(f"⚠️ ការទាញយកសំឡេង Clone បរាជ័យ៖ {str(e)}")

    # ៣. ករណីស្វែងរកហ្វាល់ Preset ធម្មតា ប្រសិនបើរកសំឡេង Clone មិនឃើញ
    if not ref_wav_path:
        voice_name = re.sub(r'[\[\]]', '', preset_name)
        preset_file = f"{voice_name}.wav"
        if os.path.exists(preset_file):
            ref_wav_path = preset_file
        else:
            fallback_files = [f for f in os.listdir('.') if f.endswith('.wav')]
            if fallback_files:
                ref_wav_path = fallback_files[0]

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

    # លុបឯកសារសំឡេងបណ្តោះអាសន្នចេញវិញ ដើម្បីកុំឱ្យពេញទំហំផ្ទុក (Memory) របស់ Server
    if tmp_b64_audio_path and os.path.exists(tmp_b64_audio_path):
        os.remove(tmp_b64_audio_path)

    if not master_audio_chunks:
        return np.array([], dtype=np.float32)
        
    return np.concatenate(master_audio_chunks)

def handler(job):
    try:
        job_input = job.get("input", {})
        
        req_text = job_input.get("text", "")
        req_mode = job_input.get("mode", "preset")
        req_preset_name = job_input.get("preset_name", "[ប្រុស១]")
        req_ref_audio_name = job_input.get("ref_audio_name")
        req_ref_audio_base64 = job_input.get("ref_audio_base64") # ចាប់យក Base64
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
            
            final_wav = np.concatenate(master_audio)
            
        else:
            default_cfg = {
                "mode": req_mode, 
                "preset_name": req_preset_name, 
                "ref_audio_name": req_ref_audio_name,
                "ref_audio_base64": req_ref_audio_base64 # បញ្ជូនបន្តទៅកាន់ប្រព័ន្ធផលិតសំឡេង
            }
            final_wav = generate_segment(req_text, default_cfg)

        if len(final_wav) == 0:
            return {"error": "ការផលិតសំឡេងបរាជ័យ (គ្មានទិន្នន័យត្រឡប់មកវិញ)។"}

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
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

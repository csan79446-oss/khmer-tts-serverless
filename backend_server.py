import os
import io
import re
import uuid
import base64
import torch
import shutil
import numpy as np
from scipy.io import wavfile
import runpod

# --- ១. ការគ្រប់គ្រង និងផ្ទុកម៉ូដែល VoxCPM (Global Initialization) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_INSTANCE = None
PRESET_DIR = "./presets"
TEMP_DIR = "/tmp/runpod_tts"

# បង្កើត Folder សម្រាប់ទុកហ្វាយបណ្តោះអាសន្ន និងប្រភពសំឡេង Preset
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)

def load_tts_model():
    """ផ្ទុកម៉ូដែល VoxCPM តែម្តងគត់ចូលទៅកាន់ VRAM"""
    global MODEL_INSTANCE
    if MODEL_INSTANCE is None:
        print("----> 🚀 កំពុងផ្ទុកម៉ូដែល VoxCPM ចូលទៅកាន់ VRAM...")
        try:
            import torch._dynamo
            # បិទ torch.compile ដើម្បីជៀសវាងបញ្ហាដាច់ខាតជាមួយ Compiler រំខានល្បឿន Cold Start
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True
            
            from voxcpm import VoxCPM
            
            # ប្រព័ន្ធនឹងទាញយកពី "openbmb/VoxCPM2" ឬបងអាចប្តូរទៅកាន់ផ្លូវ فولឌ័រ local របស់បងបាន
            model_path = os.environ.get("MODEL_PATH", "openbmb/VoxCPM2")
            
            MODEL_INSTANCE = VoxCPM.from_pretrained(
                model_path, 
                load_denoiser=False,
                optimize=False # រត់តាមបែប Eager mode មិនប្រើ torch.compile នាំឱ្យគាំង
            )
            print("----> 🎉 ម៉ូដែល VoxCPM ផ្ទុកជោគជ័យ និងត្រៀមខ្លួនរួចរាល់!")
        except Exception as e:
            print(f"----> ❌ ការផ្ទុកម៉ូដែល VoxCPM បរាជ័យ: {str(e)}")
            raise e
    return MODEL_INSTANCE

def cleanup_temp_files():
    """សម្អាតហ្វាយបណ្តោះអាសន្នភ្លាមៗ ការពារឌីសពេញ (Disk Space Error)"""
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning cache: {str(e)}")

# --- ២. មុខងារទាញយកលក្ខណៈសំឡេង (Speaker Routing Engine) ---
def get_speaker_audio_path(mode, preset_name=None, reference_audio_b64=None):
    """ធានាការផ្គូផ្គងសំឡេងគំរូ និងបំប្លែង Base64 សម្រាប់ VoxCPM ឱ្យបានត្រឹមត្រូវ ១០០%"""
    default_path = os.path.join(PRESET_DIR, "default.wav")
    
    if mode == "Preset" and preset_name:
        preset_path = os.path.join(PRESET_DIR, f"{preset_name}.wav")
        if os.path.exists(preset_path):
            return preset_path
        print(f"រកមិនឃើញសំឡេង Preset: {preset_name}, ប្រព័ន្ធនឹងប្រើសំឡេងលំនាំដើម")
        return default_path
        
    elif mode == "Clone" and reference_audio_b64:
        unique_id = str(uuid.uuid4())
        temp_wave_path = os.path.join(TEMP_DIR, f"clone_{unique_id}.wav")
        
        # សម្អាត Header របស់ Base64 ប្រសិនបើវាលោតមកជាមួយពី Frontend (ឧ. data:audio/wav;base64,...)
        clean_b64 = reference_audio_b64.split(",")[-1] if "," in reference_audio_b64 else reference_audio_b64
        with open(temp_wave_path, "wb") as fh:
            fh.write(base64.b64decode(clean_b64))
        return temp_wave_path
        
    return default_path

# --- ៣. មុខងារបំបែក និងវិភាគ Tag សម្រាប់រឿង SRT ---
def parse_srt_tags(text):
    """
    បំបែកអក្សរ និង Tag សំឡេងចេញពីគ្នាឱ្យដាច់ស្រឡះ ជៀសវាងការច្រឡំសំឡេងគ្នា។
    គំរូ៖ [piseth]: សួស្តីបង [sreyna]: ចាសសួស្តី
    """
    pattern = r'\[([^\]]+)\]:\s*([^\[]+)'
    matches = re.findall(pattern, text)
    
    if not matches:
        return [("default", text.strip())]
    
    return [(m[0].strip(), m[1].strip()) for m in matches if m[1].strip()]

# --- ៤. មុខងារស្នូលសម្រាប់ហៅសាច់កូដ VoxCPM ពិតប្រាកដ ---
def run_tts_inference(model, text, speaker_wav_path):
    """បញ្ជូនទិន្នន័យទៅឱ្យ API របស់ VoxCPM ផលិតសំឡេង"""
    try:
        # ការហៅមុខងារ .generate របស់ VoxCPM ពិតប្រាកដ
        # គាំទ្រការបោះផ្លូវហ្វាយសំឡេងចូលទៅកាន់ reference_wav_path សម្រាប់ Clone និង Preset
        if speaker_wav_path and os.path.exists(speaker_wav_path):
            wav = model.generate(
                text=text,
                reference_wav_path=speaker_wav_path,
                cfg_value=2.0,
                inference_timesteps=10
            )
        else:
            wav = model.generate(
                text=text,
                cfg_value=2.0,
                inference_timesteps=10
            )
            
        # ទាញយក Sample Rate ពិតប្រាកដរបស់ម៉ូដែល VoxCPM (ជាទូទៅគឺ 48000Hz Studio Quality)
        sample_rate = getattr(model.tts_model, 'sample_rate', 48000)
        return sample_rate, wav
        
    except Exception as e:
        raise RuntimeError(f"បញ្ហាក្នុងការបង្កាត់សំឡេងរបស់ម៉ូដែល VoxCPM: {str(e)}")

# --- ៥. មុខងារចម្បងរបស់ RunPod Handler ---
def handler(job):
    cleanup_temp_files()
    
    try:
        model = load_tts_model()
    except Exception as e:
        return {"status": "error", "message": f"មិនអាចដំណើរការម៉ាស៊ីន AI VoxCPM បានទេ: {str(e)}"}
        
    job_input = job['input']
    mode = job_input.get("mode", "Preset")  # ទម្រង់៖ Preset, Clone, SRT
    text = job_input.get("text", "")
    preset_name = job_input.get("speaker_preset", "default")
    reference_audio = job_input.get("reference_audio", None)
    
    if not text.strip():
        return {"status": "error", "message": "សូមបញ្ចូលអត្ថបទដែលត្រូវអាន។"}
        
    final_audio_segments = []
    global_sample_rate = 48000
    
    try:
        # --- ទម្រង់ទី ១ & ទី ២៖ Preset Mode ឬ Clone Mode ---
        if mode in ["Preset", "Clone"]:
            speaker_wav = get_speaker_audio_path(mode, preset_name, reference_audio)
            sr, audio_np = run_tts_inference(model, text, speaker_wav)
            global_sample_rate = sr
            final_audio_segments.append(audio_np)
            
        # --- ទម្រង់ទី ៣៖ SRT Mode (បំបែកសំឡេងតាម Tag) ---
        elif mode == "SRT":
            segments = parse_srt_tags(text)
            
            for speaker, segment_text in segments:
                # ផ្គូផ្គង Tag នីមួយៗទៅកាន់ហ្វាយ Preset (ឧទាហរណ៍៖ [piseth] -> ./presets/piseth.wav)
                speaker_wav = get_speaker_audio_path("Preset", preset_name=speaker)
                
                # ផលិតសំឡេងដាច់ដោយឡែកសម្រាប់ឃ្លានីមួយៗ រួចយកមកតម្រៀបគ្នា
                sr, segment_audio = run_tts_inference(model, segment_text, speaker_wav)
                global_sample_rate = sr
                final_audio_segments.append(segment_audio)
                
        # --- ដំណើរការរួបរួម និងផ្ញើលទ្ធផលត្រឡប់ ---
        if final_audio_segments:
            combined_audio = np.concatenate(final_audio_segments, axis=0)
            
            # ធ្វើការ Normalize សំឡេងកុំឱ្យបែក ឬឆ្នេរ
            max_val = np.max(np.abs(combined_audio))
            if max_val > 0:
                combined_audio = combined_audio / max_val
            
            # បម្លែងទៅជាទម្រង់ 16-bit PCM WAV
            byte_io = io.BytesIO()
            wav_data = (combined_audio * 32767).astype(np.int16)
            wavfile.write(byte_io, global_sample_rate, wav_data)
            
            audio_bytes = byte_io.getvalue()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
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
        return {"status": "error", "message": f"កំហុសប្រព័ន្ធដំណើរការ VoxCPM៖ {str(e)}"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

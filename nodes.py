import asyncio

from .src.inference.inference import Inference


class LoadMuyanTTSModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_type": ("STRING", {"default": "base"}),
            "model_path": ("STRING", {"multiline": False, "default": "pretrained_models/Muyan-TTS"}),
            "cnhubert_model_path": ("STRING", {"multiline": False, "default": "pretrained_models/chinese-hubert-base"}),
        }}

    RETURN_TYPES = ("MODEL", "MODELTYPE")
    RETURN_NAMES = ("model", "model_type")
    FUNCTION = "load_model"
    CATEGORY = "Muyan-TTS"

    def load_model(self, model_type, model_path):
        try:
          if model_type == "base":
              model_path = "pretrained_models/Muyan-TTS"
          elif model_type == "sft":
              model_path = "pretrained_models/Muyan-TTS-SFT"
          else:
              print(f"Invalid model type: '{model_type}'. Please specify either 'base' or 'sft'.")
          print(f"Model downloaded successfully to {model_path}")
        except Exception as e:
          print(f"Error downloading model: {str(e)}")
  
        model = model_path
        return (model, model_type)
    

class LoadRefAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_path": ("STRING", {"default": "assets/Claire.wav"}),
            }
        }

    RETURN_TYPES = ("RefAudio",)
    RETURN_NAMES = ("ref_audio",)
    FUNCTION = "load_audio"
    CATEGORY = "Muyan-TTS"

    def load_audio(self, audio_path):
        ref_audio = audio_path
        return (ref_audio,)


class PromptText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt_text": ("STRING", {"multiline": False, "default": "Although the campaign was not a complete success, it did provide Napoleon with valuable experience and prestige."}),
        }}

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Muyan-TTS"

    def load_prompt(self, prompt_text):
        prompt = prompt_text
        return (prompt,)


class InputText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_prompt": ("STRING", {"default": "Welcome to the captivating world of podcasts, let's embark on this exciting journey together."})
            }
        }

    RETURN_TYPES = ("TEXT",)
    RETURN_NAMES = ("input_text",)
    FUNCTION = "load_text"
    CATEGORY = "Muyan-TTS"

    def load_text(self, text_prompt):
        input_text = text_prompt
        return (input_text,)


class Generate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": ("MODELTYPE",),
                "ref_audio": ("RefAudio",),
                "prompt": ("PROMPT",),
                "input_text": ("TEXT",),
            }
        }

    RETURN_TYPES = ("TTS",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Muyan-TTS"

    def generate(self, model, model_type, ref_audio, prompt, input_text):
        tts = Inference(model_type, model_path, enable_vllm_acc=False)
        audio = tts.generate(
            ref_wav_path=ref_audio,
            prompt_text=prompt,
            text=input_text
        )
        return (audio,)


class SaveMuyanTTSAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("TTS",),
                "output_path": ("STRING", {"default": "tts.wav"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "Muyan-TTS"

    def save(self, audio, output_path):
        with open(output_path, "wb") as f:
          f.write(next(audio))  
        print(f"Speech generated in {output_path}")
        return ()

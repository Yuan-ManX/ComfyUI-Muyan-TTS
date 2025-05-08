from .nodes import LoadMuyanTTSModel, LoadRefAudio, PromptText, InputText, Generate, SaveMuyanTTSAudio

NODE_CLASS_MAPPINGS = {
    "LoadMuyanTTSModel": LoadMuyanTTSModel,
    "LoadRefAudio": LoadRefAudio,
    "PromptText": PromptText,
    "InputText": InputText,
    "Generate": Generate,
    "SaveMuyanTTSAudio": SaveMuyanTTSAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMuyanTTSModel": "Load MuyanTTS Model",
    "LoadRefAudio": "Load Ref Audio",
    "PromptText": "Prompt Text",
    "InputText": "Input Text",
    "Generate": "Generate",
    "SaveMuyanTTSAudio": "Save MuyanTTS Audio",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

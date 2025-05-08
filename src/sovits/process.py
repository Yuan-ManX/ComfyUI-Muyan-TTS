import re
import torch
import librosa
import types
import sys
import numpy as np
from io import BytesIO
from sovits.models import SynthesizerTrn
import logging
from sovits.utils import *
import sovits.cnhubert as cnhubert

utils_module = types.ModuleType('utils')
from sovits.utils import HParams
setattr(utils_module, 'HParams', HParams)
sys.modules['utils'] = utils_module

class Speaker:
    def __init__(self, name, sovits, phones = None, bert = None, prompt = None):
        self.name = name
        self.sovits = sovits
        self.phones = phones
        self.bert = bert
        self.prompt = prompt

class Sovits:
    def __init__(self, vq_model, hps):
        self.vq_model = vq_model
        self.hps = hps

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class Processor:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Processor, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        is_half=True,
        stream_mode="close",
        is_int32=False,
        sovits_path="pretrained_models/Muyan-TTS/sovits.pth",
        cnhubert_path="pretrained_models/chinese-hubert-base",
        default_cut_punc="",
    ):
        if not Processor._initialized:
            self.spec_cache = {}
            self.speaker_list = {}
            self.audio_token_cache = {}
            self.is_int32 = is_int32
            self.stream_mode = stream_mode
            self.device = device
            self.is_half = is_half
            self.default_cut_punc = default_cut_punc

            # set sovits path
            if sovits_path is not None:
                self.sovits_path = sovits_path
            else:
                raise FileNotFoundError("No valid sovits.pth found.")

            # load sovits model
            sovits = self.get_sovits_weights(self.sovits_path, self.device)
            self.speaker_list["default"] = Speaker(name="default", sovits=sovits)

            # load cnhubert model
            cnhubert.cnhubert_base_path = cnhubert_path
            ssl_model = cnhubert.get_model()
            if self.is_half:
                self.ssl_model = ssl_model.half().to(self.device)
            else:
                self.ssl_model = ssl_model.to(self.device)

            Processor._initialized = True

    def generate_audio_token(self, ref_wav_path, spk="default"):
        if ref_wav_path in self.audio_token_cache:
            return self.audio_token_cache[ref_wav_path]

        infer_sovits = self.speaker_list[spk].sovits
        vq_model = infer_sovits.vq_model
        hps = infer_sovits.hps
        zero_wav = np.zeros(
            int(hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half == True else np.float32,
        )
        
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if self.is_half == True:
                wav16k = wav16k.half().to(self.device)
                zero_wav_torch = zero_wav_torch.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
                zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(self.device)
            
            audio_token = tensor_to_audio_tokens(prompt)
            self.audio_token_cache[ref_wav_path] = audio_token
            return audio_token
    
    def get_sovits_weights(self, sovits_path, device="cpu"):
        dict_s2 = torch.load(sovits_path, map_location=device)
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"
        model_params_dict = vars(hps.model)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict
        )
        if ("pretrained" not in sovits_path):
            del vq_model.enc_q
        if self.is_half == True:
            vq_model = vq_model.half().to(self.device)
        else:
            vq_model = vq_model.to(self.device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        sovits = Sovits(vq_model, hps)
        return sovits

    def get_tts_wav(self, predict, vits_wav_path, prompt_text, prompt_language, text, text_language, speed=1, inp_refs=[], spk="default", scaling_factor=1.0):
        infer_sovits = self.speaker_list[spk].sovits
        vq_model = infer_sovits.vq_model
        hps = infer_sovits.hps

        dtype = torch.float16 if self.is_half == True else torch.float32
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if self.is_half == True else np.float32)
        refers=[]
        with torch.no_grad():
            for path in [vits_wav_path] + inp_refs:
                # try:
                    if path not in self.spec_cache:
                        refer = get_spepc(hps, path).to(dtype).to(self.device)
                        
                        self.spec_cache[path] = refer
                    else:
                        refer = self.spec_cache[path]
                    refers.append(refer)
                # except Exception as e:
                #     logging.error(e)
        version = vq_model.version
        prompt_language = prompt_language.lower()
        text_language = text_language.lower()
        texts = text.split("\n")
        audio_bytes = BytesIO()

        splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
        for text in texts:
            if only_punc(text):
                continue

            audio_opt = []
            if (text[-1] not in splits): text += "。" if text_language != "en" else "."
            phones2, _, _ = get_phone(text, text_language, version)
            pred_token = []
            for item in re.findall(r"<\|.+?\|>", predict):
                if item == "<|eot_id|>" or item == "<|audio_token_end|>":
                    continue
                pred_token.append(int(item.split("_")[-1].split("|>")[0]))
            pred_semantic = torch.LongTensor(pred_token).to(self.device).unsqueeze(0).unsqueeze(0)
            audio = vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                                refers,speed=speed).detach().cpu().numpy()[0, 0]
            max_audio=np.abs(audio).max()
            if max_audio>1:
                audio /= max_audio
            audio *= scaling_factor  # adjust volumn
            audio = np.clip(audio, -1.0, 1.0)
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            if self.is_int32:
                audio_bytes = pack_audio(audio_bytes,(np.concatenate(audio_opt, 0) * 2147483647).astype(np.int32),hps.data.sampling_rate, self.is_int32)
            else:
                audio_bytes = pack_audio(audio_bytes,(np.concatenate(audio_opt, 0) * 32768).astype(np.int16),hps.data.sampling_rate, self.is_int32)
            if self.stream_mode == "normal":
                audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
                yield audio_chunk
        
        if not self.stream_mode == "normal": 
            audio_bytes = pack_wav(audio_bytes,hps.data.sampling_rate, self.is_int32)
            yield audio_bytes.getvalue()


    def handle(self, pred_semantic, vits_wav_path, prompt_text, prompt_language, text, text_language, cut_punc, speed, inp_refs, scaling_factor):
        if cut_punc == None:
            text = cut_text(text, self.default_cut_punc)
        else:
            text = cut_text(text, cut_punc)
        res = self.get_tts_wav(pred_semantic, vits_wav_path, prompt_text, prompt_language, text, text_language, speed, inp_refs, scaling_factor=scaling_factor)
        return res

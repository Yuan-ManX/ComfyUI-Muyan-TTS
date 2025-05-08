import sys,os
import torch
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import math, traceback
import sys, pdb

import sovits
import logging, librosa
from sovits.models import SynthesizerTrn
from sovits.utils import clean_path
logging.getLogger("numba").setLevel(logging.WARNING)

import types
utils_module = types.ModuleType('utils')
from sovits.utils import HParams
setattr(utils_module, 'HParams', HParams)
sys.modules['utils'] = utils_module

def audio_tokenizer(input_dir = "data",
                 output_dir = "data",
                 s2config_path = "sovits/configs/s2.json", 
                 sovits_path=None):
    if sovits_path==None:
        if os.path.exists("pretrained_models/Muyan-TTS/sovits.pth"):
            sovits_path = "pretrained_models/Muyan-TTS/sovits.pth" 
        elif os.path.exists("pretrained_models/Muyan-TTS-SFT/sovits.pth"):
            sovits_path = "pretrained_models/Muyan-TTS-SFT/sovits.pth"
        else:
            raise FileNotFoundError(sovits_path)

    if os.path.exists(os.path.join(output_dir, "tmp", "text", "name2semantic.tsv")) == False:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        hps = sovits.utils.get_hparams_from_file(s2config_path)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            version="v2",
            **hps.model
        )
        if is_half == True:
            vq_model = vq_model.half().to(device)
        else:
            vq_model = vq_model.to(device)
        vq_model.eval()
        
        vq_model.load_state_dict(
            torch.load(sovits_path, map_location="cpu")["weight"], strict=False
        )
        

        def name2go(wav_name, lines):
            hubert_path = "%s/%s.pt" % (os.path.join(input_dir, "tmp", "hubert_embedding"), wav_name)
            if os.path.exists(hubert_path) == False:
                return
            ssl_content = torch.load(hubert_path, map_location="cpu")
            if is_half == True:
                ssl_content = ssl_content.half().to(device)
            else:
                ssl_content = ssl_content.to(device)
            codes = vq_model.extract_latent(ssl_content)
            semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
            lines.append("%s\t%s" % (wav_name, semantic))

        with open(os.path.join(input_dir, "tmp", "text", "raw_data.list"), "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        lines1 = []
        for line in lines:
            try:
                wav_name, spk_name, language, text = line.split("|")
                wav_name=clean_path(wav_name)
                wav_name = os.path.basename(wav_name)
                name2go(wav_name, lines1)
            except:
                print(line, traceback.format_exc())
        with open(os.path.join(output_dir, "tmp", "text", "name2semantic.tsv"), "w", encoding="utf8") as f:
            f.write("\n".join(lines1))

# -*- coding: utf-8 -*-

import sys,os

import sovits.cnhubert as cnhubert
import torch


import pdb,traceback,numpy as np,logging
from scipy.io import wavfile
import librosa
from sovits.utils import load_audio,clean_path

from time import time as ttime
import shutil

def my_save(fea,path):
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    tmp_path="%s.pth"%(ttime())
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

def audio_hubert_embedding(input_dir="data", output_dir="data", cnhubert_base_path="pretrained_models/chinese-hubert-base"):
    cnhubert.cnhubert_base_path = cnhubert_base_path
    os.makedirs(os.path.join(output_dir, "tmp", "hubert_embedding"), exist_ok=True)
    is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
    maxx=0.95
    alpha=0.5
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model=cnhubert.get_model()
    if(is_half==True):
        model=model.half().to(device)
    else:
        model = model.to(device)

    nan_fails=[]
    def name2go(wav_name,wav_path):
        hubert_path="%s/%s.pt"%(os.path.join(output_dir, "tmp", "hubert_embedding"),wav_name)
        if(os.path.exists(hubert_path)):return
        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.2:
            print("%s-filtered,%s" % (wav_name, tmp_max))
            return
        tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
        tmp_audio = librosa.resample(
            tmp_audio32b, orig_sr=32000, target_sr=16000
        )
        tensor_wav16 = torch.from_numpy(tmp_audio)
        if (is_half == True):
            tensor_wav16=tensor_wav16.half().to(device)
        else:
            tensor_wav16 = tensor_wav16.to(device)
        ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
        if np.isnan(ssl.detach().numpy()).sum()!= 0:
            nan_fails.append((wav_name,wav_path))
            print("nan filtered:%s"%wav_name)
            return
        my_save(ssl,hubert_path)

    with open(os.path.join(input_dir, "tmp", "text", "raw_data.list"),"r",encoding="utf8")as f:
        lines=f.read().strip("\n").split("\n")

    for line in lines:
        try:
            wav_name, spk_name, language, text = line.split("|")
            wav_name=clean_path(wav_name)
            wav_path=wav_name
            wav_name = os.path.basename(wav_name)
            name2go(wav_name,wav_path)
        except:
            print(line,traceback.format_exc())

    if(len(nan_fails)>0 and is_half==True):
        is_half=False
        model=model.float()
        for wav in nan_fails:
            try:
                name2go(wav[0],wav[1])
            except:
                print(wav_name,traceback.format_exc())

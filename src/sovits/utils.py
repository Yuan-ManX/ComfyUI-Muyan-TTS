import os,re
import traceback
import librosa
import numpy as np
import torch
import ffmpeg
import threading
import subprocess
from sovits.module.mel_processing import spectrogram_torch
from sovits.text.cleaner import clean_text
from sovits.text import cleaned_text_to_sequence
import soundfile as sf
from io import BytesIO
from sovits.LangSegment import LangSegment
import json

def clean_text_inf_normed_text(text, language, version):
    language = language.replace('all_', '')
    _, _, norm_text = clean_text(text, language, version)
    return norm_text

def clean_text_inf_phone(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_normed_text(text,language,version,final=False):
    text = text.replace("&", "and")
    text = text.replace("%", " percent")

    formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")
    formattext = re.sub(r'\s+([,.!?;:"-])', r'\1', formattext)
    
    norm_text = clean_text_inf_normed_text(formattext, language, version)
    norm_text = norm_text.replace(" ' ", "'")
    return norm_text


def tensor_to_audio_tokens(tensor):
    """
    Converts a Torch Tensor into a sequence of strings in the form <|audio_token_x|>.
    :param tensor: Torch Tensor containing a sequence of integers
    :return: String representation of the audio token sequence
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a Torch Tensor")
    # Ensure the Tensor is one-dimensional
    tensor = tensor.flatten()
    # Convert each integer to the <|audio_token_x|> format
    tokens = [f"<|audio_token_{int(val)}|>" for val in tensor.tolist()]
    # Join tokens with no spaces
    result = "".join(tokens)
    return result

def cut_text(text, punc):
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        if len(items)%2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    return text

def read_clean_buffer(audio_bytes):
    audio_chunk = audio_bytes.getvalue()
    audio_bytes.truncate(0)
    audio_bytes.seek(0)
    return audio_bytes, audio_chunk

def pack_audio(audio_bytes, data, rate, media_type="wav", is_int32=False):
    if media_type == "ogg":
        audio_bytes = pack_ogg(audio_bytes, data, rate)
    elif media_type == "aac":
        audio_bytes = pack_aac(audio_bytes, data, rate, is_int32)
    else:
        audio_bytes = pack_raw(audio_bytes, data, rate)
    return audio_bytes

def pack_ogg(audio_bytes, data, rate):
    def handle_pack_ogg():
        with sf.SoundFile(audio_bytes, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
            audio_file.write(data)
    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as e:
        # If changing the thread stack size is unsupported, a RuntimeError is raised.
        print("RuntimeError: {}".format(e))
        print("Changing the thread stack size is unsupported.")
    except ValueError as e:
        # If the specified stack size is invalid, a ValueError is raised and the stack size is unmodified.
        print("ValueError: {}".format(e))
        print("The specified stack size is invalid.")
    return audio_bytes

def pack_raw(audio_bytes, data, rate):
    audio_bytes.write(data.tobytes())
    return audio_bytes

def pack_wav(audio_bytes, rate, is_int32=False):
    if is_int32:
        data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int32)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV', subtype='PCM_32')
    else:
        data = np.frombuffer(audio_bytes.getvalue(),dtype=np.int16)
        wav_bytes = BytesIO()
        sf.write(wav_bytes, data, rate, format='WAV')
    return wav_bytes

def pack_aac(audio_bytes, data, rate, is_int32=False):
    if is_int32:
        pcm = 's32le'
        bit_rate = '256k'
    else:
        pcm = 's16le'
        bit_rate = '128k'
    process = subprocess.Popen([
        'ffmpeg',
        '-f', pcm,  
        '-ar', str(rate),  
        '-ac', '1',  
        '-i', 'pipe:0',  
        '-c:a', 'aac',  
        '-b:a', bit_rate, 
        '-vn',  
        '-f', 'adts',  
        'pipe:1'  
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    audio_bytes.write(out)
    return audio_bytes

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams

def should_skip_period(text, dot_index):
    """
    Determines whether to skip the period at text[dot_index] == '.'.
    Rules:
    1) If there is only one letter before the period (from the last space to the period), skip it.
    2) If the word before the period is Mr / mr / Dr / dr / Mrs / mrs / st, etc., skip it.
    """
    skip_abbrs = {"mr", "dr", "mrs", "st"}  

    last_space_index = text.rfind(' ', 0, dot_index)
    if last_space_index == -1:
        start_idx = 0
    else:
        start_idx = last_space_index + 1

    word_before_dot = text[start_idx:dot_index].strip()
    word_lower = word_before_dot.lower()

    if len(word_lower) == 1:
        return True

    if word_lower in skip_abbrs:
        return True

    return False

def fix_spaced_caps(text):
    words = text.split()
    result = []
    i = 0
    
    while i < len(words):
        current_word = words[i]
        # Check if the word is a single uppercase letter
        if len(current_word) == 1 and current_word.isupper():
            combined = current_word
            j = i + 1
            # Check subsequent words to merge consecutive single uppercase letters
            while j < len(words):
                next_word = words[j]
                # If it's a single uppercase letter, continue merging
                if len(next_word) == 1 and next_word.isupper():
                    combined += next_word
                    j += 1
                # If the next word starts with an uppercase letter and has a suffix (e.g., 's), merge the first letter and keep the suffix
                elif next_word[0].isupper() and len(next_word) > 1 and not next_word[1:].isalnum():
                    combined += next_word[0]  # Take only the first letter
                    combined += next_word[1:]  # Append the remaining part
                    j += 1
                    break
                else:
                    break
            # If multiple words were merged, add the combined result
            if j > i + 1:
                result.append(combined)
                i = j
            else:
                # If only one uppercase letter and no merging conditions are met, add it separately
                result.append(current_word)
                i += 1
        else:
            result.append(current_word)
            i += 1
    
    return ' '.join(result)

def clean_and_split_text(text):
    cleaned_text = re.sub(r"[^\w\s\.,!?:']", " ", text)
    
    sentences = []
    current_buffer = []

    i = 0
    while i < len(cleaned_text):
        ch = cleaned_text[i]
        if ch == '.' :
            if should_skip_period(cleaned_text, i):
                current_buffer.append(ch)
                i += 1
                continue
            else:
                current_buffer.append(ch)
                sentence = "".join(current_buffer).strip()
                if sentence:
                    sentences.append(sentence)
                current_buffer = []
                i += 1
                continue

        elif ch == '!' :
            current_buffer.append(ch)
            sentence = "".join(current_buffer).strip()
            if sentence:
                sentences.append(sentence)
            current_buffer = []
            i += 1
            continue
        
        elif ch == '?' :
            current_buffer.append(ch)
            sentence = "".join(current_buffer).strip()
            if sentence:
                sentences.append(sentence)
            current_buffer = []
            i += 1
            continue
        
        else:
            current_buffer.append(ch)
            i += 1

    leftover = "".join(current_buffer).strip()
    
    if leftover:
        sentences.append(leftover)

    sentences = [
        fix_spaced_caps(s.replace(':', ','))
        for s in sentences
    ]

    return sentences



def get_phone(text,language,version):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        
        phones, word2ph, norm_text = clean_text_inf_phone(formattext, language, version)
        
    return phones,None,norm_text

def get_bert_feature(text, word2ph, bert_model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)

def get_spepc(hps, filename):
    audio,_ = librosa.load(filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):
        audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec

def load_audio(file, sr):
    try:
        file = clean_path(file)
        if os.path.exists(file) == False:
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Error in audio loading")
    return np.frombuffer(out, np.float32).flatten()

def clean_path(path_str:str):
    if path_str.endswith(('\\','/')):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")

def load_wav_to_torch(full_path):
    data, sampling_rate = librosa.load(full_path, sr=None)
    return torch.FloatTensor(data), sampling_rate

def merge_sentences_minimum_n(sentences, n):
    merged_results = []
    buffer = []
    buffer_words = 0

    for s in sentences:
        words_count = len(s.split())
        buffer.append(s)
        buffer_words += words_count

        if buffer_words >= n:
            merged_results.append(" ".join(buffer))
            buffer = []
            buffer_words = 0

    if buffer:
        leftover_words = sum(len(x.split()) for x in buffer)
        if leftover_words < n and merged_results:
            merged_results[-1] += " " + " ".join(buffer)
        else:
            merged_results.append(" ".join(buffer))

    return merged_results


def count_audio_tokens(strings):
    return [s.count('<|audio_token_') for s in strings]

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

import sys
import os
from sovits.process import Processor
from sovits.utils import *
from fastapi.responses import StreamingResponse
import re, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MIN_SENTENCE_LENGTH = 12


class Inference():
    def __init__(self, 
                 model_type, model_path, ref_wav_path="assets/Claire.wav", 
                 prompt_text="Although the campaign was not a complete success, it did provide Napoleon with valuable experience and prestige.",
                 enable_vllm_acc=False):
        # ref_wav_path and prompt_text are used here only to initialize sovits (otherwise the first run would be slow)
        # new ref_wav_path and prompt_text can still be specified later using call_tts
        self.sovits_processor = Processor(sovits_path=os.path.join(model_path, "sovits.pth"))
        self.sovits_processor.generate_audio_token(ref_wav_path)
        clean_text_inf_normed_text(prompt_text, 'en', 'v1') 
        logging.info("init vits finish")

        # Used to distinguish between API mode and regular TTS mode
        self.enable_vllm_acc = enable_vllm_acc
        if self.enable_vllm_acc == True:
            from inference.inference_llama import InferenceLlamaVllm
            self.llama = InferenceLlamaVllm(model_path, model_type)
        else:
            from inference.inference_llama import InferenceLlamaHf
            self.llama = InferenceLlamaHf(model_path, model_type)
            
        self.model_type = model_type
        
    def _create_prompt(self, prompt_text, text, audio_tokens):
        if self.model_type == "base":
            return " " + prompt_text.strip() + " " + text.strip() + " " + audio_tokens.strip()
            
        elif self.model_type == "sft":
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"\
                    + " "\
                    + text.strip()\
                    + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        else:
            raise ValueError(f"Error model type: {self.model_type}")    
        
    def _process_prompt(self, ref_wav_path, prompt_text, text):
        audio_tokens = self.sovits_processor.generate_audio_token(ref_wav_path)
        prompt_text = get_normed_text(prompt_text, 'en', 'v1')
        text = get_normed_text(text, 'en', 'v1')
        
        # Used to handle overly long sentences by splitting them into multiple shorter sentences
        batch_texts = clean_and_split_text(text)
        batch_texts = merge_sentences_minimum_n(batch_texts, MIN_SENTENCE_LENGTH) 
        
        batch_prompts = []
        for i in range(len(batch_texts)):
            batch_prompts.append(self._create_prompt(prompt_text, batch_texts[i], audio_tokens))
            
        return batch_prompts
        
    async def generate(self, ref_wav_path, prompt_text, text, temperature=0.6, 
                 repetition_penalty=1.0, cut_punc=None,
                 speed=1.0, scaling_factor=1.0):
        batch_prompts = self._process_prompt(ref_wav_path, prompt_text, text)
        
        results = await self.llama.cal_tts(batch_prompts, temperature, repetition_penalty)
        pred_semantic = "".join(results)
        wavs = self.sovits_processor.handle(pred_semantic, ref_wav_path, prompt_text, 'en', text, 'en', cut_punc, speed, [], scaling_factor)
        return wavs

    def init_vits(self, ref_wav_path, prompt_text):
        logging.info("init vits...")
        self.sovits_processor.generate_audio_token(ref_wav_path)
        clean_text_inf_normed_text(prompt_text, 'en', 'v1') 
        logging.info("init vits finish")


    
    
    

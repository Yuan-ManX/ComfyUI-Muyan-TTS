from data_process import audio_hubert_embedding, audio_tokenizer, text_audio_combine, text_format_conversion
import os

class DataProcess():
    def __init__(self):
        self.librispeech_text_format_conversion = text_format_conversion.generate_raw_data_list
        self.audio_hubert_embedding = audio_hubert_embedding.audio_hubert_embedding
        self.audio_tokenizer = audio_tokenizer.audio_tokenizer
        self.text_audio_combine = text_audio_combine.text_audio_combine
        
    def pipeline(self, librispeech_dir=None, input_dir="data", output_dir="data"):
        # 1. Use librispeech as examples
        print("step 1: librispeech_text_format_conversion")
        if librispeech_dir==None:
            print("Please specify the path of librispeech")
        self.librispeech_text_format_conversion(librispeech_dir=librispeech_dir, output_dir=output_dir, subset="dev-clean")
        
        # 2. This step is to get hubert embedding
        print("step 2: audio_hubert_embedding")
        self.audio_hubert_embedding(input_dir=input_dir, output_dir=output_dir)
        
        # 3. This step is to get semantic tokens from hubert embedding
        print("step 3: audio_tokenizer")
        self.audio_tokenizer(input_dir=input_dir, output_dir=output_dir, sovits_path="pretrained_models/Muyan-TTS/sovits.pth")
        
        # 4. This step is to organize data into llama training format
        print("step 4: text_audio_combine")
        self.text_audio_combine(input_dir=input_dir, output_dir=output_dir)
        
        print(f"finished process, saved at: {os.path.join(input_dir, 'tts_sft_data.json')}")
        
        

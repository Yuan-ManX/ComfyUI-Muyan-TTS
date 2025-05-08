import subprocess
import os
import time
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import AsyncOpenAI
import asyncio
import logging
import socket

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def send_request_llama(model_type, prompt_text, llama_port, temperature=0.6, repetition_penalty=1.0):
    # Call the OpenAI ChatCompletion endpoint asynchronously
    url_chat = f"http://localhost:{llama_port}/v1"
    client = AsyncOpenAI(api_key="EMPTY", base_url=url_chat)
    
    if model_type == "base":
        response = await client.completions.create(
            model="llamaar",
            prompt=prompt_text,
            temperature=temperature,
            max_tokens=512,  
            stop=["<|audio_token_end|>","<|end_header_id|>","<|end_of_text|>"],
            extra_body={
                "skip_special_tokens": False,
                "repetition_penalty": repetition_penalty
            },
        )
        return response.choices[0].text
    
    else: # sft
        response = await client.chat.completions.create(
            model="llamaar",
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            extra_body={
                "skip_special_tokens": False,
                "repetition_penalty": repetition_penalty
            },
            temperature=temperature,
        )
        return response.choices[0].message.content

class InferenceLlamaVllm:
    def __init__(self, model_path, model_type):
        self.llama_port = self._get_available_port()
        self.model_type = model_type
        os.makedirs('logs', exist_ok=True)

        pid_file = "logs/vllm_pid.txt"
        cmd = (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} --served-model-name llamaar --enable-prefix-caching "
            f"--host 0.0.0.0 --port {self.llama_port} > logs/llm.log 2>&1 & echo $! > {pid_file}"
        )

        subprocess.run(cmd, shell=True)
        
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                self.pid = int(f.read().strip())
        else:
            raise RuntimeError("Failed to capture PID")

        logging.info("initializing llama, it may take some time...")

        self._wait_for_service(self.llama_port, timeout=300)
        logging.info(f"init llama finish in IPD:{self.pid}")
        
    def _is_port_available(self, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return True
            except OSError:
                return False

    def _get_available_port(self, start_port=None, max_tries=10):
        env_port = os.getenv('LLAMA_PORT', '8021')
        try:
            port = int(env_port)
            if self._is_port_available(port):
                return str(port)  
        except ValueError:
            pass  

        start_port = start_port or int(env_port)  
        port = start_port
        for _ in range(max_tries):
            if self._is_port_available(port):
                return str(port)
            port += 1
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_tries - 1}")
        
    def _wait_for_service(self, port, timeout):
        start_time = time.time()
        url = f"http://localhost:{port}/health" 
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1) 
        raise TimeoutError("Service startup timeout!")
    
    async def cal_tts(self, batch_prompts, temperature, repetition_penalty):
        tasks = [send_request_llama(self.model_type, prompt, self.llama_port, temperature, repetition_penalty) for prompt in batch_prompts]
            
        results = await asyncio.gather(*tasks)
        
        return results

    


class InferenceLlamaHf:
    def __init__(self, model_path, model_type):
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llama = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16, 
            trust_remote_code=True     
        ).to(self.device)
        
        self.model_type = model_type
        
    async def cal_tts(self, batch_prompts, temperature=0.6, repetition_penalty=1.0):
        results = []
        
        async def process_prompt(prompt):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[1]
            
            with torch.no_grad():
                outputs = await asyncio.to_thread(
                    self.llama.generate,
                    **inputs,
                    max_length=512,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty
                )
            
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
            return generated_text

        tasks = [process_prompt(prompt) for prompt in batch_prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

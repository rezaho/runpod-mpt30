import runpod
import os
import time
# import subprocess
# subprocess.call("nvidia-smi")

from pydantic import BaseModel
from typing import Dict, Optional, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import re
      
model_name = 'mosaicml/mpt-30b-chat'

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16, # Load model weights in bfloat16
    #load_in_4bit=True,
    trust_remote_code=True,
    device_map='cuda:0'
)

tokenizer = AutoTokenizer.from_pretrained(model_name)



class Item(BaseModel):
    message: str
    system_message: Optional[str] = None
    return_only_output: Optional[bool] = True
    temperature: Optional[float] = 0.7
    max_new_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    do_sample: Optional[bool] = True


def predict(item):
    item = Item(**item)
    # build the prompt
    prompt = ""
    if item.system_message:
        prompt+=f"<|im_start|>system\n{item.system_message}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{item.message}<|im_end|>\n<|im_start|>assistant\n"
    #tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(input_ids=inputs, max_new_tokens=item.max_new_tokens, do_sample=item.do_sample, temperature=item.temperature, top_p=item.top_p, eos_token_id= 0, pad_token_id= 0)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    if item.return_only_output:
        response = re.search(r"assistant\n([\S\s]+)", response)[1]
    return response



## load your model(s) into vram here

def handler(event):
    response = predict(event['input'])
    # do the things
    return {"result":response}

# def handler(event):
#     time.sleep(3.0)
#     return {'result': 'This is a test 7'}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })
    # print(handler(
    #     {
    #         'input':{
    #             'system_message':'You are a helpful assistant.',
    #             'message':'Hello, how are you?'
    #         }
    #     }
    # ))
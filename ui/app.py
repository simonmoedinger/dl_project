from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS if you need to handle cross-origin requests
import math
from numba import cuda
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, GenerationConfig
import torch
import pandas as pd
import math
from sklearn.metrics import accuracy_score
from transformers import BitsAndBytesConfig


model_name = "../leolm/fully_trained_model/"
#model_name = "LeoLM/leo-mistral-hessianai-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = False
tokenizer.add_bos_token, tokenizer.add_eos_token

generation_config = GenerationConfig(max_new_tokens=1024,
                                    temperature=0.4,
                                    top_p=0.95,
                                    top_k=40,
                                    repetition_penalty=1.3,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    do_sample=True,
                                    use_cache=True,
                                    output_attentions=False,
                                    output_hidden_states=False,
                                    output_scores=False,
                                    remove_invalid_values=True
                                    )

app = Flask(__name__)
CORS(app)  # Initialize CORS on the Flask app if needed


@app.route('/')
def index():
    return render_template('index.html')

import time
@app.route('/model_select', methods=['POST'])
def model_select():
    # Get the message from the POST request's body
    global model, tokenizer, generation_config
    del model, tokenizer, generation_config

    model_name = request.json['model']
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token, tokenizer.add_eos_token

    generation_config = GenerationConfig(max_new_tokens=1024,
                                         temperature=0.4,
                                         top_p=0.95,
                                         top_k=40,
                                         repetition_penalty=1.3,
                                         bos_token_id=tokenizer.bos_token_id,
                                         eos_token_id=tokenizer.eos_token_id,
                                         do_sample=True,
                                         use_cache=True,
                                         output_attentions=False,
                                         output_hidden_states=False,
                                         output_scores=False,
                                         remove_invalid_values=True
                                         )
    return jsonify({'message': 'Model changed successfully'})

@app.route('/chat', methods=['POST'])
def chat():
    
    # Get the message from the POST request's body
    user_message = request.json['message']

    input_tokens=tokenizer(user_message, return_tensors="pt").to(model.device)
    output_tokens=model.generate(**input_tokens, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)[0]
    answer=tokenizer.decode(output_tokens, skip_special_tokens=True)


    # Return the bot's response in JSON format
    return jsonify({'message': answer})

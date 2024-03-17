import os
from tqdm import tqdm 
import json 
import csv
import gc
import random
import string
import difflib

import html 
import base64 
import re 

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, default_data_collator
import datasets 
from contextlib import nullcontext
from peft import PeftConfig, PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from IPython.display import display_html


torch.autograd.set_detect_anomaly(True)

class LLMAttributor:
    def __init__(self, 
                 llama2_dir=None, 
                 tokenizer_dir=None,
                 model_save_dir=None, 
                 ckpt_names=None,
                 ckpt_prefix = "checkpoint-",
                 device=None,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 train_config=None,
                 block_size=64,
                 split_data_into_multiple_batches=False,
                 max_new_tokens=100
                 ) -> None:
        # set model dir and ckpt names
        if llama2_dir is None: llama2_dir = model_save_dir
        if tokenizer_dir is None: tokenizer_dir = llama2_dir
        assert model_save_dir is not None 
        self.llama2_dir = llama2_dir 
        self.model_save_dir = model_save_dir
        self.ckpt_names = ckpt_names
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)
        elif self.ckpt_names is None:
            self.ckpt_names = [f for f in os.listdir(self.model_save_dir) if f.startswith(ckpt_prefix) and f[len(ckpt_prefix):].isdigit()]
            self.ckpt_names = sorted(self.ckpt_names, key=lambda x: int(x[len(ckpt_prefix):]))
            if len(self.ckpt_names) == 0: self.ckpt_names = None

        self.loaded_model_dir = None
        self.model = None

        # set tokenizer 
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if self.device.startswith("cuda:"): torch.cuda.set_device(int(self.device[5:]))
        elif self.device not in ["auto","cuda"]: torch.cuda.set_device(self.device)

        # set PEFT related configs
        self.load_in_8bit = True 
        self.torch_dtype = torch.float16

        # set train(finetuning)/dataset configs
        self.train_config = train_config
        self.block_size = block_size
        
        # set generate configs
        self.max_new_tokens = max_new_tokens

        # set dataset 
        self.set_dataset(train_dataset, val_dataset, test_dataset, split_data_into_multiple_batches=split_data_into_multiple_batches)

        # TBU: if a model directory/checkpoint have been specified, load the model

        # location of this project
        self.project_dir = os.path.dirname(__file__)

        # for attribution 
        self.attribution_score_dir = os.path.join(self.model_save_dir, "attribution_score")
        self.text_hash_filepath = os.path.join(self.attribution_score_dir, "text_hash.csv")
        self.text_hash_dict = dict()
        if not os.path.exists(self.attribution_score_dir): os.makedirs(self.attribution_score_dir)
        elif os.path.exists(self.text_hash_filepath):
            with open(self.text_hash_filepath, "r") as f:
                csv_reader = csv.reader(f)
                self.text_hash_dict = {k:v for (k,v) in csv_reader}

    def set_dataset(self, train_dataset, val_dataset=None, test_dataset=None, split_data_into_multiple_batches=False):
        self.train_dataset = self._tokenize_dataset(train_dataset)
        self.val_dataset = self._tokenize_dataset(val_dataset)
        self.test_dataset = self._tokenize_dataset(test_dataset)

    def _tokenize_dataset(self, dataset):
        if dataset is None: return None 
        
        if type(dataset) == datasets.arrow_dataset.Dataset: pass 
        elif type(dataset) == dict: dataset = datasets.Dataset.from_dict(dataset)
        elif type(dataset) == list: dataset = datasets.Dataset.from_list(dataset)
        
        if "text" in dataset.features:
            if 'input_ids' in dataset.features: return dataset 
            dataset = dataset.map(lambda data: self.tokenizer(data["text"], max_length=None), remove_columns=["text"])
            dataset = dataset.map(lambda data: self._split_tokens(data=data))

            final_dataset = {key: [] for key in dataset.column_names}
            
            for data in dataset:
                num_batches = len(data["input_ids"])
                for i in range(num_batches):
                    for key in final_dataset: final_dataset[key].append(data[key][i])
            final_dataset["text"] = [self._ids_to_text(input_ids) for input_ids in final_dataset["input_ids"]]
            final_dataset["labels"] = final_dataset["input_ids"]

        elif "prompt" in dataset.features and "output" in dataset.features:
            dataset = dataset.rename_column("input", "input_text")
            dataset = dataset.map(lambda data: self.tokenizer(data["prompt"], max_length=None))
            dataset = dataset.rename_column("input_ids", "prompt_ids")
            dataset = dataset.rename_column("prompt", "prompt_text")
            dataset = dataset.map(lambda data: self.tokenizer(data["output"], max_length=None))
            dataset = dataset.rename_column("input_ids", "output_ids")
            dataset = dataset.rename_column("output", "output_text")
            
            final_dataset = {key: [] for key in dataset.column_names}  # prompt_ids, output_ids, prompt, output, attention_mask
            final_dataset["text"] = []
            final_dataset["input_ids"] = []
            for data in dataset:
                final_dataset["input_text"].append(data["input_text"])
                final_dataset["output_text"].append(data["output_text"])
                final_dataset["prompt_text"].append(data["prompt_text"])
                final_dataset["text"].append(data["prompt_text"] + data["output_text"])
                
                if data["prompt_ids"][0]!=self.tokenizer.bos_token_id: data["prompt_ids"] = [self.tokenizer.bos_token_id] + data["prompt_ids"]
                final_dataset["prompt_ids"].append(data["prompt_ids"])
                final_dataset["output_ids"].append(data["output_ids"])

                # input_ids = np.concatenate([data["prompt_ids"], data["output_ids"]])
                input_ids = data["prompt_ids"] + data["output_ids"]
                if len(input_ids) > self.block_size: input_ids = input_ids[:self.block_size]
                elif len(input_ids)<self.block_size: 
                    num_padding = (self.block_size - len(input_ids))
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
                final_dataset["input_ids"].append(input_ids)

                attention_mask = [0]*len(data["prompt_ids"]) + [1]*len(data["output_ids"])
                if len(attention_mask) > self.block_size: attention_mask = attention_mask[:self.block_size]
                elif len(attention_mask)<self.block_size: 
                    assert num_padding == self.block_size-len(attention_mask)
                    attention_mask = attention_mask + [0]*(self.block_size-len(attention_mask))
                final_dataset["attention_mask"].append(attention_mask)
            
            final_dataset["labels"] = final_dataset["input_ids"]

        else: raise ValueError("Please provide a dataset with 'text' or 'prompt' and 'output' keys")
        
        dataset = datasets.Dataset.from_dict(final_dataset)

        return dataset 
    
    def _split_tokens(self, data=None, tokens=None, block_size=None):
        assert data is not None or tokens is not None
        
        if tokens is None: tokens = data["input_ids"]
        attention_mask = data["attention_mask"] if data is not None and "attention_mask" in data else np.ones_like(tokens)

        # title = data["title"] if data is not None and "title" in data else None
        
        if block_size is None: block_size = self.block_size
        if tokens[0] != self.tokenizer.bos_token_id: tokens = [self.tokenizer.bos_token_id] + tokens 
        if tokens[-1] != self.tokenizer.eos_token_id: tokens = tokens + [self.tokenizer.eos_token_id]

        total_length = ((len(tokens) + block_size - 1) // block_size) * block_size
        if total_length > len(tokens): 
            tokens = tokens + [self.tokenizer.pad_token_id] * (total_length - len(tokens))
            attention_mask = np.concatenate([attention_mask, np.zeros(total_length - len(attention_mask))])
        
        result = {
            "input_ids": [tokens[i:i+block_size] for i in range(0, total_length, block_size)],
            "attention_mask": [attention_mask[i:i+block_size] for i in range(0, total_length, block_size)],
        }

        # other meta data (e.g., source url, title)
        for key in data:
            if key not in ["input_ids", "attention_mask"]: result[key] = [data[key]] * len(result["input_ids"])

        return result

    def set_model(self, pretrained=False, pretrained_dir=None, verbose=False, peft=True, peft_config=None):
        if pretrained and hasattr(self, "loaded_model_dir") and pretrained_dir==self.loaded_model_dir: return
        
        self.clear_model()
        
        if pretrained: 
            assert pretrained_dir is not None
            model_load_dir = pretrained_dir
        else: model_load_dir = self.llama2_dir
        
        if verbose: print("Loading model from", model_load_dir)
        self.model = LlamaForCausalLM.from_pretrained(model_load_dir, load_in_8bit=self.load_in_8bit, torch_dtype=self.torch_dtype, device_map=self.device)
        
        if peft:
            peft_config = self.set_model_peft(pretrained=pretrained, peft_config_dir=model_load_dir, peft_config=peft_config)
            if verbose:
                print(peft_config)
                self.model.print_trainable_parameters()
        
        self.loaded_model_dir = model_load_dir
        self.attr_grad = None
        self.scores = None 

    def set_model_peft(self, pretrained=False, peft_config=None, peft_config_dir=None):
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

        if pretrained:
            if peft_config is None and peft_config_dir is not None: 
                try: peft_config = PeftConfig.from_pretrained(peft_config_dir)
                except: peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
            elif peft_config is None: peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
            peft_config.inference_mode = False 
            self.model = PeftModel.from_pretrained(self.model, peft_config_dir, is_trainable=True, config=peft_config)
        else:
            if peft_config is None: peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
            peft_config.inference_mode = False 
            self.model = get_peft_model(self.model, peft_config)

        # self.model = self.model.to(self.device)

        return peft_config

    def clear_model(self):
        if hasattr(self, "model") and self.model is not None: del self.model 
        gc.collect()
        torch.cuda.empty_cache()
    
    def check_model(self, eval_i=0, eval_prompt=None):
        # if eval_prompt is None, should input eval_i and compare the result with the groundtruth
        # if eval_prompt is given, generate with the input prompt
        self.model.eval()
        device = "cuda" if self.device=="auto" else self.device
        if eval_prompt is None and self.val_dataset is not None: eval_prompt = self.val_dataset[eval_i]['text']
        elif eval_prompt is None and self.test_dataset is not None: eval_prompt = self.test_dataset[eval_i]['text']
        elif eval_prompt is None: eval_prompt = self.train_dataset[eval_i]['text']
        model_input = self.tokenizer(eval_prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = self.model.generate(**model_input, max_new_tokens=100)[0]
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            print("Generated:", decoded)
        
        if eval_prompt is None: print("Groundtruth:", self.val_dataset[eval_i]['answer'])

    def finetune(self, config=None, overwrite=False, bf16=False, logging_strategy="steps", logging_steps=10, save_strategy="epoch", save_steps=50, optim="adamw_torch_fused", max_steps=-1, learning_rate=1e-4, num_train_epochs=5, gradient_accumulation_steps=2, per_device_train_batch_size=2, gradient_checkpointing=False):
        if not hasattr(self, "model") or self.model is None: self.set_model(pretrained=False)
        if config is None and self.train_config is not None: config = self.train_config
        else:
            training_args = TrainingArguments(
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                gradient_accumulation_steps=gradient_accumulation_steps,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_checkpointing=gradient_checkpointing,
                output_dir=self.model_save_dir, 
                overwrite_output_dir=overwrite, 
                bf16=bf16, 
                logging_dir=os.path.join(self.model_save_dir, "logs"), 
                logging_strategy=logging_strategy,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                save_steps=save_steps,
                optim=optim,
                max_steps=max_steps)
        
        profiler = nullcontext()
        self.loaded_model_dir = None 

        # finetune
        with profiler:
            self.trainer = Trainer(model=self.model, args=training_args, train_dataset=self.train_dataset, data_collator=default_data_collator, callbacks=[])
            self.trainer.train()

        # model has been changed
        self.model.save_pretrained(self.model_save_dir)
        self.loaded_model_dir = self.model_save_dir
        ckpt_prefix = "checkpoint-"
        self.ckpt_names = [f for f in os.listdir(self.loaded_model_dir) if f.startswith(ckpt_prefix) and f[len(ckpt_prefix):].isdigit()]

    def generate(self, prompt=None, max_new_tokens=1000, return_decoded=True):
        device = "cuda" if self.device=="auto" else self.device
        model_input = self.tokenizer(prompt, return_tensors="pt").to(device)
        if self.model==None: self.set_model(pretrained=True, pretrained_dir=self.model_save_dir)
        output = self.model.generate(**model_input, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)[0]
        prompt_len = model_input["input_ids"].shape[1]
        output = output[prompt_len:]
        if return_decoded: output = self._ids_to_text(output)
        return output

    def get_text_to_hash_key(self, prompt, text, attr_tokens_pos):
        if type(attr_tokens_pos) == np.ndarray: attr_tokens_pos = attr_tokens_pos.reshape(-1).tolist()
        elif type(attr_tokens_pos) == torch.Tensor: attr_tokens_pos = attr_tokens_pos.reshape(-1).tolist()
        assert type(attr_tokens_pos) == list
        return f"[ATTR PROMPT] {prompt} [ATTR TEXT] {text} [ATTR TOKEN POS] {','.join([str(x) for x in attr_tokens_pos])}"

    def get_text_to_hash_val(self, prompt, text, attr_tokens_pos):
        hash_key = self.get_text_to_hash_key(prompt, text, attr_tokens_pos)
        if hash_key in self.text_hash_dict: return self.text_hash_dict[hash_key]
        else:
            random_string = ''.join(random.choice(string.ascii_letters) for i in range(10))
            while random_string in self.text_hash_dict.values(): random_string = ''.join(random.choice(string.ascii_letters) for i in range(10))
            self.text_hash_dict[hash_key] = random_string
            with open(self.text_hash_filepath, "a") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([hash_key, random_string])
            print(f"Added new hash value for {hash_key}")
            return random_string
    
    def set_attr_text(self, prompt=None, generated_text=None, attr_tokens_pos=None): 
        self.prompt = prompt 
        self.generated_text = generated_text 
        device = "cuda" if self.device=="auto" else self.device
        
        prompt_ids = self._text_to_ids(prompt)
        generated_ids = self._text_to_ids(generated_text)
        self.prompt_ids = torch.LongTensor(prompt_ids).to(device).reshape(1,-1)
        self.generated_ids = torch.LongTensor(generated_ids).to(device).reshape(1,-1)
        self.prompt_token_num = self.prompt_ids.shape[1]

        # self.scores = None 
        if attr_tokens_pos is not None: attr_tokens_pos = torch.LongTensor(attr_tokens_pos).to(device)
        else: 
            self.select_tokens(prompt=prompt, generated_text=generated_text)
            # attr_tokens_pos = torch.arange(self.generated_ids.shape[1]).to(self.device)
        # self.hash_val = self.get_text_to_hash_val(self.prompt, self.generated_text, attr_tokens_pos)

    def _text_to_ids(self, text):
        return torch.LongTensor(self.tokenizer(text, return_tensors="pt")["input_ids"])[:,1:]

    def _ids_to_text(self, tokens):
        if type(tokens) == torch.Tensor: tokens = tokens.detach().cpu().numpy()
        tokens = np.array(tokens).reshape(-1).tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def set_attr_tokens_pos(self, attr_tokens_pos):
        device = "cuda" if self.device=="auto" else self.device
        if hasattr(self, "attr_tokens_pos") and self.attr_tokens_pos is not None and torch.equal(self.attr_tokens_pos, torch.LongTensor(attr_tokens_pos).to(device)): return
        self.attr_tokens_pos = torch.LongTensor(attr_tokens_pos).to(device)
        self.attr_grad = None
        # self.scores = None

    def get_tokens_html_code(self, token_ids, attention_mask=None):
        if type(token_ids) in [np.array, torch.Tensor] and token_ids.ndim==2: token_ids = token_ids[0]
        if attention_mask is None: attention_mask = np.ones(len(token_ids))

        random_int = random.randint(0, 1000000)
        random_id = f"tokens-container-{random_int}"
        text_html_code = f"<div class='tokens-container' id='{random_id}'>"
        nobr_closed = True

        previous_space_flag = False

        for i, token_id in enumerate(token_ids):
            token_decoded = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            class_name = "token attended-token" if attention_mask[i] == 1 else "token unattended-token"
            if token_decoded=="<0x0A>": 
                class_name += " line-break-token"
                # if nobr_closed : text_html_code += f"<div class='{class_name}' id='token-{random_int}-{i}'>&#182</div><br>"
                # else: text_html_code += f"<div class='{class_name}' id='token-{random_int}-{i}'>&#182</div></nobr><br>"
                if nobr_closed : text_html_code += f"<div class='{class_name}' id='token-{random_int}-{i}'></div><br>"
                else: text_html_code += f"<div class='{class_name}' id='token-{random_int}-{i}'></div></nobr><br>"
                nobr_closed = True
                continue

            if "<" in token_decoded: token_decoded = token_decoded.replace("<", "&lt;")
            if ">" in token_decoded: token_decoded = token_decoded.replace(">", "&gt;")

            if "▁" == token_decoded:
                if i == 0: continue
                class_name += " space-token"
                token_decoded = "&nbsp;"
                html_code = f"<div class='{class_name}' id='token-{random_int}-{i}'>{token_decoded}</div>"
                previous_space_flag = True
                if not nobr_closed: html_code = html_code + "</nobr>"
            elif "▁" == token_decoded[0]: 
                if not nobr_closed: html_code = html_code + "</nobr>"
                if i>0: class_name += " left-space-token"
                token_decoded = token_decoded[1:]
                html_code = f"<nobr><div class='{class_name}' id='token-{random_int}-{i}'>{token_decoded}</div>"
                nobr_closed = False 
                previous_space_flag = False
            else:
                if previous_space_flag:
                    html_code = f"<nobr><div class='{class_name}' id='token-{random_int}-{i}'>{token_decoded}</div>"
                    nobr_closed = False
                else: html_code = f"<div class='{class_name}' id='token-{random_int}-{i}'>{token_decoded}</div>"
                previous_space_flag = False
            
            text_html_code += html_code
        
        if not nobr_closed: text_html_code += "</nobr>"
        text_html_code += "</div>"

        return text_html_code, random_id 

    def select_tokens(self, prompt=None, generated_text=None, prompt_ids=None, generated_ids=None):
        assert prompt is not None or prompt_ids is not None
        assert generated_text is not None or generated_ids is not None
        device = "cuda" if self.device=="auto" else self.device
        
        if prompt_ids is None: prompt_ids = self._text_to_ids(prompt)
        else: prompt_ids = torch.LongTensor(prompt_ids).reshape(1,-1).to(device)
        if prompt is None: prompt = self._ids_to_text(prompt_ids)
        if generated_ids is None: generated_ids = self._text_to_ids(generated_text)
        else: generated_ids = torch.LongTensor(generated_ids).reshape(1,-1).to(device)
        prompt_token_num = prompt_ids.shape[1]
        generated_token_num = generated_ids.shape[1]
        
        
        generated_mask = np.hstack([np.zeros([prompt_token_num]), np.ones([generated_token_num])])
        entire_text_ids = torch.cat([prompt_ids, generated_ids], dim=-1)
        text_html_code, tokens_container_id = self.get_tokens_html_code(entire_text_ids, generated_mask)

        html_code_filename = os.path.join(self.project_dir, "visualization/html/token_position_selector.html")
        html_code = open(html_code_filename, "r").read()

        html_code = html_code.replace("<!--tokens-slot-->", text_html_code)

        css_code_filename = os.path.join(self.project_dir, "visualization/css/select.css")
        css_code = "<style>" + open(css_code_filename, "r").read() + "</style>"
        html_code = html_code.replace("<!--style-slot-->", css_code)
        
        random_int_id = tokens_container_id.split("-")[-1]
        button_code = f"""<button class="copy-selected-token-idx-button" id="copy-selected-token-idx-button-{random_int_id}" type="button">Copy Selected Token Indices</button>
        <div class="selected-token-indices" id="selected-token-indices-{random_int_id}"></div>"""
        html_code = html_code.replace("<!--button-slot-->", button_code)

        message_js = f"""
        (function() {{
            const event = new Event('selectAttrTokensPos');
            event.token_container_id = '{tokens_container_id}';
            event.prompt_token_num = {prompt_token_num};
            event.total_token_num = {prompt_token_num + generated_token_num};
            document.dispatchEvent(event);
        }}())
        """
        message_js = message_js.encode()
        messenger_js_base64 = base64.b64encode(message_js).decode("utf-8")
        message_js = f"""<script src='data:text/javascript;base64,{messenger_js_base64}'></script>"""
        html_msg_code = html_code.replace("<!--message-slot-->", message_js)
        
        js_code_filename = os.path.join(self.project_dir, "visualization/js/token_select_events.js")
        js_string = open(js_code_filename, "r").read()
        js_b = bytes(js_string, encoding="utf-8")
        js_base64 = base64.b64encode(js_b).decode("utf-8")
        html_msg_js_code = html_msg_code.replace("<!--js-slot-->", f"""<script data-notebookMode="true" data-package="{__name__}" src='data:text/javascript;base64,{js_base64}'></script>""")

        iframe = f"""
        <iframe 
            srcdoc="{html.escape(html_msg_js_code)}" 
            frameBorder="0" 
            height="1000px"
            width="100%">
        """
        display_html(iframe, raw=True)

    def get_attr_grad(self, prompt_ids, gen_ids, attr_tokens_pos):
        device = "cuda" if self.device=="auto" else self.device
        entire_text_ids = torch.cat([prompt_ids, gen_ids], dim=-1)
        attr_attention_mask = torch.ones_like(entire_text_ids).to(device)
        prompt_token_num = prompt_ids.shape[1]
        
        logsoftmax = nn.LogSoftmax(dim=-1)
        self.model.eval()
        self.model.zero_grad()
        out = self.model.base_model(entire_text_ids, attr_attention_mask)
        attr_logits = out.logits 
        attr_logprobs = logsoftmax(attr_logits) 
        attr_logprobs = attr_logprobs[0, (prompt_token_num+attr_tokens_pos-1), entire_text_ids[0, prompt_token_num+attr_tokens_pos]]
        attr_logprob = attr_logprobs.sum() 
        attr_grad = torch.autograd.grad(attr_logprob, [param for param in self.model.parameters() if param.requires_grad])
        self.model.zero_grad()
        return attr_grad
    
    def save_ckpt_gradients(self, ckpt_names=None, ckpt_name=None, verbose=False):
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name] 
        else: ckpt_names = self.ckpt_names

        device = "cuda" if self.device=="auto" else self.device
        
        # save gradient values for the model parameters for each training data for each checkpointed model
        for ckpt in ckpt_names:
            if verbose: print("Saving gradients for", ckpt)

            # create grad_dir if it doesn't exists
            grad_dir = os.path.join(self.attribution_score_dir, ckpt, "training_gradients")
            if not os.path.exists(grad_dir): os.makedirs(grad_dir)
            
            # check if the gradients are already computed
            grad_files = [filename for filename in os.listdir(grad_dir) if filename.endswith(".pt") or filename.endswith(".pth")]
            grad_computed = (len(grad_files) == len(self.train_dataset))
            if grad_computed: continue

            # compute gradient and save
            ckpt_dir = os.path.join(self.model_save_dir, ckpt)
            self.set_model(pretrained=True, pretrained_dir=ckpt_dir)
            self.model.eval()
            
            for i, data in enumerate(tqdm(self.train_dataset)):
                # get the Delta_theta when we update the model with "data"
                input_ids = torch.LongTensor(data["input_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.LongTensor(data["attention_mask"]).unsqueeze(0).to(device)
                labels = torch.LongTensor(data["labels"]).unsqueeze(0).to(device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                grad_loss = torch.autograd.grad(loss, [param for param in self.model.parameters() if param.requires_grad])
                torch.save(grad_loss, f"{grad_dir}/{i}.pt")
            
    def get_datainf_scores(self, prompt_ids, gen_ids, attr_tokens_pos, ckpt_name=None, ckpt_names=None, integrated=True, integration="mean", weighted=False, weight=None, verbose=False, ckpt_num=-1):
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name] 
        elif ckpt_num>=0: ckpt_names = self.ckpt_names[:ckpt_num]
        else: ckpt_names = self.ckpt_names

        prompt = self._ids_to_text(prompt_ids)
        generated_text = self._ids_to_text(gen_ids)
        attr_hash = self.get_text_to_hash_val(prompt, generated_text, attr_tokens_pos)
        # if os.path.exists(os.path.join(self.attribution_score_dir, f"score_{attr_hash}.json")) and integrated:
        #     with open(os.path.join(self.attribution_score_dir, f"score_{attr_hash}.json"), "r") as f: integrated_scores = json.load(f)
        #     self.scores = np.array(integrated_scores)
        #     return self.scores

        all_scores = dict()
        for ckpt_name in ckpt_names:
            if verbose: print("Computing datainf scores for", ckpt_name)
            
            score_dir = os.path.join(self.attribution_score_dir, ckpt_name, f"score_{attr_hash}.json")
            if not os.path.exists(score_dir): self.save_datainf_scores(prompt_ids, gen_ids, attr_tokens_pos, attr_hash, ckpt_name=ckpt_name)
            with open(score_dir, "r") as f: scores = json.load(f)
            
            scores = np.array(scores)
            all_scores[ckpt_name] = scores
            
        if integrated and weighted: 
            assert weight is not None
            integrated_scores = None 
            raise NotImplementedError
        elif integrated and integration=="mean": 
            integrated_scores = np.mean(list(all_scores.values()), axis=0)
            self.scores = integrated_scores
        elif integrated and integration=="median": 
            integrated_scores = np.median(list(all_scores.values()), axis=0)
            self.scores = integrated_scores
        else: self.scores = all_scores
        return self.scores
        
    def get_inner_scores(self, ckpt_name=None, ckpt_names=None, integrated=True, integration="mean", weighted=False, weight=None, verbose=False):
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name] 
        else: ckpt_names = self.ckpt_names

        if os.path.exists(os.path.join(self.attribution_score_dir, f"inner_{self.attr_hash}.json")) and integrated:
            with open(os.path.join(self.attribution_score_dir, f"inner_{self.attr_hash}.json"), "r") as f: integrated_scores = json.load(f)
            self.scores = np.array(integrated_scores)
            return self.scores

        all_scores = dict()
        for ckpt_name in ckpt_names:
            if verbose: print("Computing inner products for", ckpt_name)
            
            score_dir = os.path.join(self.attribution_score_dir, ckpt_name, f"inner_{self.attr_hash}.json")
            if not os.path.exists(score_dir): self.save_inner_scores(ckpt_name=ckpt_name)
            with open(score_dir, "r") as f: scores = json.load(f)
            
            scores = np.array(scores)
            all_scores[ckpt_name] = scores
            
        if integrated and weighted: 
            assert weight is not None
            integrated_scores = None 
            raise NotImplementedError
        elif integrated and integration=="mean": 
            integrated_scores = np.mean(list(all_scores.values()), axis=0)
            self.scores = integrated_scores
        elif integrated and integration=="median": 
            integrated_scores = np.median(list(all_scores.values()), axis=0)
            self.scores = integrated_scores
        else: self.scores = all_scores
        return self.scores

    def save_datainf_scores(self, prompt_ids, gen_ids, attr_tokens_pos, attr_hash, ckpt_name):
        ckpt_dir =  os.path.join(self.model_save_dir, ckpt_name)
        score_dir = os.path.join(self.attribution_score_dir, ckpt_name)
        score_file = os.path.join(score_dir, f"score_{attr_hash}.json")
        if not score_dir: os.makedirs(score_dir)
        if os.path.exists(score_file): return

        # compute datainf score
        self.set_model(pretrained=True, pretrained_dir=ckpt_dir)
        self.model.eval()
        self.save_ckpt_gradients(ckpt_name=ckpt_name)
        attr_grad = self.get_attr_grad(prompt_ids, gen_ids, attr_tokens_pos)

        grad_dir = os.path.join(score_dir, "training_gradients")
        n_layers = len(attr_grad)
        n_train = len(self.train_dataset)
        tr_grad_norm = np.zeros([n_layers, n_train])
        
        for train_i in range(n_train):
            grad_i = torch.load(f"{grad_dir}/{train_i}.pt")
            for l in range(n_layers):
                tr_grad_norm[l, train_i] = (grad_i[l] * grad_i[l]).sum()

        d_l = np.array([grad.numel() for grad in attr_grad])
        lambdas = np.sum(tr_grad_norm, axis=-1) / (10 * n_train * d_l)

        rs = [torch.zeros_like(grad) for grad in attr_grad]
        for train_i in range(n_train):
            grad_i = torch.load(f"{grad_dir}/{train_i}.pt")
            for l in range(n_layers):
                c = (attr_grad[l] * grad_i[l]).sum() / (lambdas[l] + tr_grad_norm[l, train_i])
                ri = (attr_grad[l] - c * grad_i[l]) / (n_train * lambdas[l])
                rs[l] += ri 

        scores = np.zeros([n_train])
        for train_k in range(n_train):
            grad = torch.load(f"{grad_dir}/{train_k}.pt")
            for l in range(n_layers):
                scores[train_k] -= (rs[l] * grad[l]).sum()

        # save the scores to score_dir 
        with open(score_file, "w") as f: json.dump(scores.tolist(), f)

    def save_inner_scores(self, ckpt_name=None, ckpt_names=None):
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name] 
        else: ckpt_names = self.ckpt_names

        original_device = self.device
        for ckpt_name in ckpt_names:
            ckpt_dir =  os.path.join(self.model_save_dir, ckpt_name)
            score_dir = os.path.join(self.attribution_score_dir, ckpt_name)
            score_file = os.path.join(score_dir, f"inner_{self.attr_hash}.json")
            if not score_dir: os.makedirs(score_dir)
            if os.path.exists(score_file): continue

            # compute datainf score
            if self.device in ["auto", "cuda"]:
                self.device = "cuda:0"
                torch.cuda.set_device(self.device)
            self.set_model(pretrained=True, pretrained_dir=ckpt_dir)
            self.model.eval()
            self.save_ckpt_gradients(ckpt_name=ckpt_name)
            attr_grad = self.get_attr_grad()

            grad_dir = os.path.join(score_dir, "training_gradients")
            n_layers = len(attr_grad)
            n_train = len(self.train_dataset)
            inners = np.zeros([n_train])
            attr_grad_norm = torch.sum(torch.Tensor([(attr_grad[l]*attr_grad[l]).sum() for l in range(n_layers)])) ** 0.5
            for i in range(n_train):
                grad_i = torch.load(f"{grad_dir}/{i}.pt")
                grad_i_norm = torch.sum(torch.Tensor([(grad_i[l]*grad_i[l]).sum() for l in range(n_layers)])) ** 0.5
                inner = torch.sum(torch.Tensor([torch.sum(attr_grad[l] * grad_i[l]) for l in range(n_layers)]))
                inner /= (attr_grad_norm * grad_i_norm)
                inners[i] = inner.item()

            # save the scores to score_dir 
            with open(score_file, "w") as f: json.dump(inners.tolist(), f)

        if original_device != self.device:
            self.device = original_device
            if self.device not in ["auto", "cuda"]: torch.cuda.set_device(self.device)

    def get_topk_training_data(self, prompt_ids, gen_ids, attr_tokens_pos, k=3, return_scores=False, integration="mean", score_method="datainf", ckpt_num=-1):
        # if (hasattr(self, "scores") and self.scores is not None): 
        #     if type(self.scores) == dict:
        #         if ckpt_num >= 0: scores = {ckpt: self.scores[ckpt] for ckpt in self.ckpt_names[-ckpt_num:]}
        #         else: scores = self.scores
        #         if integration=="mean": scores = np.mean(list(scores.values()), axis=0)
        #         if integration=="median": scores = np.median(list(scores.values()), axis=0)
        #     else: scores = self.scores
        # else:
        if score_method=="datainf": scores = self.get_datainf_scores(prompt_ids, gen_ids, attr_tokens_pos, integrated=True, integration=integration, ckpt_num=ckpt_num)
        if score_method=="inner": scores = self.get_inner_scores(prompt_ids, gen_ids, attr_tokens_pos, integrated=True, integration=integration, ckpt_num=ckpt_num)

        topk_training_idx = np.argsort(-scores)[:k]
        topk_train_data = []
        for idx in topk_training_idx:
            topk_train_data.append(self.train_dataset[int(idx)])
        if return_scores: return topk_training_idx, topk_train_data, scores[topk_training_idx]
        return topk_training_idx, topk_train_data
    
    def tf_idf_topk_training_data(self, k, topk_data_text):
        # get TF from the topk training data
        tf_dict = dict()
        word_to_data_idx = dict()
        for data_i, text in enumerate(topk_data_text):
            for paragraph in text.split("\n"):
                for word in paragraph.split(" "):
                    word = word.strip(".,?!:;`'\"“”()[]\{\}<>-_=+*^&%$#@~|\\/\n\t\r\v\f").lower()
                    if word in tf_dict: tf_dict[word][data_i:] += 1
                    else: tf_dict[word] = np.array([0] * data_i + [1] * (k - data_i))
                    
                    if word not in word_to_data_idx: word_to_data_idx[word] = [data_i]
                    elif data_i not in word_to_data_idx[word]: word_to_data_idx[word].append(data_i)

        # get IDF based on the entire training data
        idf_dict = {word: 0 for word in tf_dict.keys()}
        for data in self.train_dataset:
            if "text" in data: text = data["text"]
            elif "input_ids" in data: text = self._ids_to_text(data["input_ids"])
            for word in set([word.strip(".,?!:;`'“”\"()[]\{\}<>-_=+*^&%$#@~|\\/\n\t\r\v\f").lower() for word in re.split("\s|\.|\n\;", text)]):
                if word in idf_dict: idf_dict[word] += 1

        for word in idf_dict.keys():
            idf_dict[word] = np.log(len(self.train_dataset) / (idf_dict[word] + 1))

        # get TF-IDF
        tfidf_dict = {word: tf_dict[word] * idf_dict[word] for word in tf_dict.keys()}
        return tfidf_dict, word_to_data_idx
        
    def attribute(self, prompt=None, generated_text=None, attr_tokens_pos=None, prompt_ids=None, generated_ids=None, pos_max_num=10, neg_max_num=10, integration="median", score_method="datainf", ckpt_names=None, ckpt_num=-1):
        assert prompt is not None or prompt_ids is not None
        assert generated_text is not None or generated_ids is not None
        
        device="cuda" if self.device=="auto" else self.device
        if prompt_ids is None: prompt_ids = self._text_to_ids(prompt).to(device)
        else: prompt_ids = torch.LongTensor(prompt_ids).to(device)
        if prompt is None: prompt = self._ids_to_text(prompt_ids)
        
        if generated_ids is None: generated_ids = self._text_to_ids(generated_text).to(device)
        
        if attr_tokens_pos is None: attr_tokens_pos = torch.arange(generated_ids.shape[1]).long().to(device)
        else: attr_tokens_pos = torch.LongTensor(attr_tokens_pos).to(device)
        
        vis_dir = os.path.join(self.project_dir, "visualization")

        base_html_code = open(os.path.join(vis_dir, "html/attribution.html"), "r").read()
        css_code = f"<style>{open(os.path.join(vis_dir, 'css/attribution.css'), 'r').read()}</style>"

        base_html_code = base_html_code.replace("<!--robot-icon-->", f"{open(os.path.join(vis_dir, 'icons/robot.svg'), 'r').read()}")
        base_html_code = base_html_code.replace("<!--up-solid-icon-->", f"{open(os.path.join(vis_dir, 'icons/up-solid.svg'), 'r').read()}")
        base_html_code = base_html_code.replace("<!--down-icon-->", f"{open(os.path.join(vis_dir, 'icons/down.svg'), 'r').read()}")

        styled_html_code = base_html_code.replace("<!--style-slot-->", css_code)
        prompt_html_code = f"<div class='prompt'>{prompt}</div>"

        generated_text_selected_mask = np.zeros(len(generated_ids[0]))
        generated_text_selected_mask[attr_tokens_pos.cpu().numpy()] = 1
        generated_text_html_code, generated_text_html_container_id = self.get_tokens_html_code(generated_ids[0], generated_text_selected_mask)
        generated_text_html_code = f"<div class='generated-text' id='{generated_text_html_container_id}'>{generated_text_html_code}</div>"
        prompt_added_html_code = styled_html_code.replace("<!--prompt-slot-->", prompt_html_code)
        generated_added_html_code = prompt_added_html_code.replace("<!--generated-text-slot-->", generated_text_html_code)

        indices, data, scores = self.get_topk_training_data(prompt_ids, generated_ids, attr_tokens_pos, k=len(self.train_dataset), return_scores=True, integration=integration, score_method=score_method, ckpt_num=ckpt_num)
        max_score_val = np.abs(scores).max()
        normalized_scores = scores / max_score_val
        scores = [float("{:.4f}".format(score)) for score in normalized_scores]
        
        # split train_idx, train_data, topk_scores into positive and negative
        positive_attribution = self.get_attribution_list(indices, data, scores, True, pos_max_num)
        negative_attribution = self.get_attribution_list(indices, data, scores, False, neg_max_num)
            
        # histogram
        n_bins = 10
        counts, bins = np.histogram(scores, bins=n_bins, range=(-1,1))
        for data_dict in positive_attribution:
            data_dict["score_histogram_bin"] = min(np.digitize(data_dict["score"], bins)-1, n_bins-1)
        for data_dict in negative_attribution:
            data_dict["score_histogram_bin"] = np.digitize(data_dict["score"], bins)-1

        scoreHistogramCounts = [[bins[i], bins[i+1], counts[i]] for i in range(len(counts))]

        # tf-idf
        pos_tf_idf, pos_word_to_data_idx = self.tf_idf_topk_training_data(pos_max_num, [d["text"] for d in positive_attribution])
        neg_tf_idf, neg_word_to_data_idx = self.tf_idf_topk_training_data(neg_max_num, [d["text"] for d in negative_attribution])
        
        # sort TF-IDF
        N = 10
        topN_word_indices_for_each_topk_pos = np.argsort(-np.array(list(pos_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(pos_tf_idf.keys())
        # topN_words_for_each_topk_pos = [[[word_indices[i], pos_tf_idf[word_indices[i]][j]] for i in topN_word_indices_for_each_topk_pos[j]] for j in range(pos_max_num)]
        topN_words_for_each_topk_pos = []
        for j in range(pos_max_num):
            topN_words_for_each_topk_pos.append([])
            for i in topN_word_indices_for_each_topk_pos[j]:
                data_idx = np.array(pos_word_to_data_idx[word_indices[i]])
                data_idx = data_idx[np.where(data_idx <= j)[0]].tolist()
                topN_words_for_each_topk_pos[j].append([word_indices[i], pos_tf_idf[word_indices[i]][j], data_idx])
        

        topN_word_indices_for_each_topk_neg = np.argsort(-np.array(list(neg_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(neg_tf_idf.keys())
        topN_words_for_each_topk_neg = [[[word_indices[i], neg_tf_idf[word_indices[i]][j]] for i in topN_word_indices_for_each_topk_neg[j]] for j in range(neg_max_num)]
        topN_words_for_each_topk_neg = []
        for j in range(neg_max_num):
            topN_words_for_each_topk_neg.append([])
            for i in topN_word_indices_for_each_topk_neg[j]:
                data_idx = np.array(neg_word_to_data_idx[word_indices[i]])
                data_idx = data_idx[np.where(data_idx <= j)[0]].tolist()
                topN_words_for_each_topk_neg[j].append([word_indices[i], neg_tf_idf[word_indices[i]][j], data_idx])
        

        # send all the positively/negatively attributed data and their scores through message
        iframe_id = f"iframe-{random.randint(0, 1000000)}"
        message_js = f"""
        (function() {{
            const event = new Event('attribute');
            event.positiveAttribution = {positive_attribution};
            event.negativeAttribution = {negative_attribution};
            event.positiveMaxNum = {pos_max_num};
            event.negativeMaxNum = {neg_max_num};
            event.posTfIdf = {topN_words_for_each_topk_pos};
            event.negTfIdf = {topN_words_for_each_topk_neg};
            event.scoreHistogramCounts = {list(scoreHistogramCounts)};
            event.scoreHistogramBins = {list(bins)};
            event.iframeId = "{iframe_id}";
            document.dispatchEvent(event);
        }}())
        """
        message_js = message_js.encode()
        messenger_js_base64 = base64.b64encode(message_js).decode("utf-8")
        message_js = f"""<script src='data:text/javascript;base64,{messenger_js_base64}'></script>"""
        html_msg_code = generated_added_html_code.replace("<!--message-slot-->", message_js)

        js_code_filename = os.path.join(vis_dir, "js/attribution.js")
        js_string = open(js_code_filename, "r").read()
        js_b = bytes(js_string, encoding="utf-8")
        js_base64 = base64.b64encode(js_b).decode("utf-8")
        html_msg_js_code = html_msg_code.replace("<!--js-slot-->", f"""<script data-notebookMode="true" data-package="{__name__}" src='data:text/javascript;base64,{js_base64}'></script>""")

        iframe = f"""
        <iframe
            srcdoc="{html.escape(html_msg_js_code)}" 
            id="{iframe_id}"
            frameBorder="0" 
            scrolling="auto"
            width="100%"
            height="1000px">
        </iframe>"""
        display_html(iframe, raw=True)

    def compare(self, prompt, edit_code=None, generated_text=None, user_provided_text=None, pos_max_num=10, neg_max_num=10, integration="median", score_method="datainf"):
        # user_provided can be given as either of text or html code
        device = "cuda" if self.device=="auto" else self.device

        if edit_code is not None and edit_code.startswith("<div>"): pass
        else:
            assert generated_text is not None
            assert user_provided_text is not None
            generated = generated_text.split(" ")
            user_provided = user_provided_text.split(" ")
            seqm = difflib.SequenceMatcher(None, generated, user_provided)
            edit_code = "<div>"
            for opcode, a0,a1,b0,b1 in seqm.get_opcodes():
                if opcode=="equal": edit_code += " ".join(seqm.a[a0:a1])
                elif opcode == "insert": edit_code += f'<span class="llm-attributor-added-text">{" ".join(seqm.b[b0:b1])}</span>'
                elif opcode == "delete": edit_code += f'<span class="llm-attributor-deleted-text">{" ".join(seqm.a[a0:a1])}</span>'
                elif opcode == "replace": edit_code += f'<span class="llm-attributor-added-text">{" ".join(seqm.b[b0:b1])}</span><span class="llm-attributor-deleted-text">{" ".join(seqm.a[a0:a1])}</span>'
            edit_code += "</div>"

        generated_ids_list, user_provided_ids_list = [], [] 
        added_flag_list, deleted_flag_list = [], []
        edit_code = edit_code.replace("<div>", "").replace("</div>", "")

        for chunk in edit_code.split('</span>'):
            chunk = chunk.lstrip(" ")
            if ('<span class="llm-attributor-added-text">' in chunk):
                eq_text, added_text = chunk.split('<span class="llm-attributor-added-text">')
                eq_ids = self._text_to_ids(eq_text)[0].cpu().numpy()
                added_ids = self._text_to_ids(added_text)[0].cpu().numpy()

                generated_ids_list = np.hstack([generated_ids_list, eq_ids])
                user_provided_ids_list = np.hstack([user_provided_ids_list, eq_ids, added_ids])
                deleted_flag_list = np.hstack([deleted_flag_list, [False]*len(eq_ids)])
                added_flag_list = np.hstack([added_flag_list, [False]*len(eq_ids), [True]*len(added_ids)])
            elif('<span class="llm-attributor-deleted-text">' in chunk):
                eq_text, deleted_text = chunk.split('<span class="llm-attributor-deleted-text">')
                eq_ids = self._text_to_ids(eq_text)[0].cpu().numpy()
                deleted_ids = self._text_to_ids(deleted_text)[0].cpu().numpy()
                
                generated_ids_list = np.hstack([generated_ids_list, eq_ids, deleted_ids])
                user_provided_ids_list = np.hstack([user_provided_ids_list, eq_ids])
                deleted_flag_list = np.hstack([deleted_flag_list, [False]*len(eq_ids), [True]*len(deleted_ids)])
                added_flag_list = np.hstack([added_flag_list, [False]*len(eq_ids)])
            else:
                eq_text = chunk
                eq_ids = self._text_to_ids(eq_text)[0].cpu().numpy()
                generated_ids_list = np.hstack([generated_ids_list, eq_ids])
                user_provided_ids_list = np.hstack([user_provided_ids_list, eq_ids])
                deleted_flag_list = np.hstack([deleted_flag_list, [False]*len(eq_ids)])
                added_flag_list = np.hstack([added_flag_list, [False]*len(eq_ids)])

        generated_ids = torch.LongTensor(generated_ids_list).unsqueeze(0).to(device)
        user_provided_ids = torch.LongTensor(user_provided_ids_list).unsqueeze(0).to(device)
        prompt_ids = self._text_to_ids(prompt).to(device)

        # start visualization
        vis_dir = os.path.join(self.project_dir, "visualization")
                
        base_html_code = open(os.path.join(vis_dir, "html/compare.html"), "r").read()
        css_code = f"<style>{open(os.path.join(vis_dir, 'css/compare.css'), 'r').read()}</style>"

        base_html_code = base_html_code.replace("<!--robot-icon-->", f"{open(os.path.join(vis_dir, 'icons/robot.svg'), 'r').read()}")
        base_html_code = base_html_code.replace("<!--user-icon-->", f"{open(os.path.join(vis_dir, 'icons/user.svg'), 'r').read()}")
        base_html_code = base_html_code.replace("<!--up-solid-icon-->", f"{open(os.path.join(vis_dir, 'icons/up-solid.svg'), 'r').read()}")
        base_html_code = base_html_code.replace("<!--down-icon-->", f"{open(os.path.join(vis_dir, 'icons/down.svg'), 'r').read()}")

        base_html_code = base_html_code.replace("<!--style-slot-->", css_code)
        prompt_html_code = f"<div class='prompt'>{self.prompt}</div>"

        generated_text_html_code, generated_text_html_container_id = self.get_tokens_html_code(generated_ids, deleted_flag_list)
        user_provided_text_html_code, user_provided_text_html_container_id = self.get_tokens_html_code(user_provided_ids, added_flag_list)
        # generated_text_html_code = f"<div class='generated-text' id='{generated_text_html_container_id}'>{generated_text_html_code}</div>"
        generated_text_html_code = f"<div class='generated-text'>{generated_text_html_code}</div>"
        user_provided_text_html_code = f"<div class='user-provided-text'>{user_provided_text_html_code}</div>"
        base_html_code = base_html_code.replace("<!--prompt-slot-->", prompt_html_code)
        base_html_code = base_html_code.replace("<!--generated-text-slot-->", generated_text_html_code)
        base_html_code = base_html_code.replace("<!--user-provided-text-slot-->", user_provided_text_html_code)

        # should do one for generated, one for compared
        gen_attr_tokens_pos = np.where(deleted_flag_list)[0]
        user_attr_tokens_pos = np.where(added_flag_list)[0]
        g_indices, g_data, g_scores = self.get_topk_training_data(prompt_ids, generated_ids, gen_attr_tokens_pos, k=len(self.train_dataset), return_scores=True, integration=integration, score_method=score_method)
        u_indices, u_data, u_scores = self.get_topk_training_data(prompt_ids, user_provided_ids, user_attr_tokens_pos, k=len(self.train_dataset), return_scores=True, integration=integration, score_method=score_method)

        max_g_score_val = np.abs(g_scores).max()
        max_u_score_val = np.abs(u_scores).max()
        max_score_val = np.max([max_g_score_val, max_u_score_val])
        # normalized_g_scores = g_scores / max_g_score_val
        normalized_g_scores = g_scores / max_score_val
        normalized_u_scores = u_scores / max_score_val
        g_scores = [float("{:.4f}".format(score)) for score in normalized_g_scores]
        u_scores = [float("{:.4f}".format(score)) for score in normalized_u_scores]

        # split train_idx, train_data, topk_scores into positive and negative
        g_positive_attribution = self.get_attribution_list(g_indices, g_data, g_scores, True, pos_max_num)
        g_negative_attribution = self.get_attribution_list(g_indices, g_data, g_scores, False, neg_max_num)
        u_positive_attribution = self.get_attribution_list(u_indices, u_data, u_scores, True, pos_max_num)
        u_negative_attribution = self.get_attribution_list(u_indices, u_data, u_scores, False, neg_max_num)
            
        # histogram
        n_bins = 10
        g_counts, g_bins = np.histogram(g_scores, bins=n_bins, range=(-1,1))
        u_counts, u_bins = np.histogram(u_scores, bins=n_bins, range=(-1,1))
        for data_dict in g_positive_attribution: data_dict["score_histogram_bin"] = min(np.digitize(data_dict["score"], g_bins)-1, n_bins-1)
        for data_dict in g_negative_attribution: data_dict["score_histogram_bin"] = np.digitize(data_dict["score"], g_bins)-1
        for data_dict in u_positive_attribution: data_dict["score_histogram_bin"] = min(np.digitize(data_dict["score"], u_bins)-1, n_bins-1)
        for data_dict in u_negative_attribution: data_dict["score_histogram_bin"] = np.digitize(data_dict["score"], u_bins)-1

        g_scoreHistogramCounts = [[g_bins[i], g_bins[i+1], g_counts[i]] for i in range(len(g_counts))]
        u_scoreHistogramCounts = [[u_bins[i], u_bins[i+1], u_counts[i]] for i in range(len(u_counts))]

        # tf-idf
        g_pos_tf_idf, g_pos_word_to_data_idx = self.tf_idf_topk_training_data(pos_max_num, [d["text"] for d in g_positive_attribution])
        g_neg_tf_idf, g_neg_word_to_data_idx = self.tf_idf_topk_training_data(neg_max_num, [d["text"] for d in g_negative_attribution])
        u_pos_tf_idf, u_pos_word_to_data_idx = self.tf_idf_topk_training_data(pos_max_num, [d["text"] for d in u_positive_attribution])
        u_neg_tf_idf, u_neg_word_to_data_idx = self.tf_idf_topk_training_data(neg_max_num, [d["text"] for d in u_negative_attribution])

        # sort TF-IDF
        N = 10
        g_topN_word_indices_for_each_topk_pos = np.argsort(-np.array(list(g_pos_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(g_pos_tf_idf.keys())
        # g_topN_words_for_each_topk_pos = [[[word_indices[i], g_pos_tf_idf[word_indices[i]][j]] for i in g_topN_word_indices_for_each_topk_pos[j]] for j in range(pos_max_num)]
        g_topN_words_for_each_topk_pos = []
        for j in range(pos_max_num):
            g_topN_words_for_each_topk_pos.append([])
            for i in g_topN_word_indices_for_each_topk_pos[j]:
                data_idx = np.array(g_pos_word_to_data_idx[word_indices[i]])
                data_idx = data_idx[np.where(data_idx <= j)[0]].tolist()
                g_topN_words_for_each_topk_pos[j].append([word_indices[i], g_pos_tf_idf[word_indices[i]][j], data_idx])

        g_topN_word_indices_for_each_topk_neg = np.argsort(-np.array(list(g_neg_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(g_neg_tf_idf.keys())
        # g_topN_words_for_each_topk_neg = [[[word_indices[i], g_neg_tf_idf[word_indices[i]][j]] for i in g_topN_word_indices_for_each_topk_neg[j]] for j in range(neg_max_num)]
        g_topN_words_for_each_topk_neg = []
        for j in range(neg_max_num):
            g_topN_words_for_each_topk_neg.append([])
            for i in g_topN_word_indices_for_each_topk_neg[j]:
                data_idx = np.array(g_neg_word_to_data_idx[word_indices[i]])
                data_idx = data_idx[np.where(data_idx <= j)[0]].tolist()
                g_topN_words_for_each_topk_neg[j].append([word_indices[i], g_neg_tf_idf[word_indices[i]][j], data_idx])

        u_topN_word_indices_for_each_topk_pos = np.argsort(-np.array(list(u_pos_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(u_pos_tf_idf.keys())
        # u_topN_words_for_each_topk_pos = [[[word_indices[i], u_pos_tf_idf[word_indices[i]][j]] for i in u_topN_word_indices_for_each_topk_pos[j]] for j in range(pos_max_num)]
        u_topN_words_for_each_topk_pos = []
        for j in range(pos_max_num):
            u_topN_words_for_each_topk_pos.append([])
            for i in u_topN_word_indices_for_each_topk_pos[j]:
                data_idx = np.array(u_pos_word_to_data_idx[word_indices[i]])
                data_idx = data_idx[np.where(data_idx <= j)[0]].tolist()
                u_topN_words_for_each_topk_pos[j].append([word_indices[i], u_pos_tf_idf[word_indices[i]][j], data_idx])

        u_topN_word_indices_for_each_topk_neg = np.argsort(-np.array(list(u_neg_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(u_neg_tf_idf.keys())
        # u_topN_words_for_each_topk_neg = [[[word_indices[i], u_neg_tf_idf[word_indices[i]][j]] for i in u_topN_word_indices_for_each_topk_neg[j]] for j in range(neg_max_num)]
        u_topN_words_for_each_topk_neg = []
        for j in range(neg_max_num):
            u_topN_words_for_each_topk_neg.append([])
            for i in u_topN_word_indices_for_each_topk_neg[j]:
                data_idx = np.array(u_neg_word_to_data_idx[word_indices[i]])
                data_idx = data_idx[np.where(data_idx <= j)[0]].tolist()
                u_topN_words_for_each_topk_neg[j].append([word_indices[i], u_neg_tf_idf[word_indices[i]][j], data_idx])

        # send all the positively/negatively attributed data and their scores through message
        iframe_id = f"iframe-{random.randint(0, 1000000)}"
        message_js = f"""
        (function() {{
            const event = new Event('compare');
            event.gPositiveAttribution = {g_positive_attribution};
            event.gNegativeAttribution = {g_negative_attribution};
            event.uPositiveAttribution = {u_positive_attribution};
            event.uNegativeAttribution = {u_negative_attribution};
            event.positiveMaxNum = {pos_max_num};
            event.negativeMaxNum = {neg_max_num};
            event.gPosTfIdf = {g_topN_words_for_each_topk_pos};
            event.gNegTfIdf = {g_topN_words_for_each_topk_neg};
            event.uPosTfIdf = {u_topN_words_for_each_topk_pos};
            event.uNegTfIdf = {u_topN_words_for_each_topk_neg};
            event.gScoreHistogramCounts = {list(g_scoreHistogramCounts)};
            event.gScoreHistogramBins = {list(g_bins)};
            event.uScoreHistogramCounts = {list(u_scoreHistogramCounts)};
            event.uScoreHistogramBins = {list(u_bins)};
            event.iframeId = "{iframe_id}";
            document.dispatchEvent(event);
        }}())
        """
        message_js = message_js.encode()
        messenger_js_base64 = base64.b64encode(message_js).decode("utf-8")
        message_js = f"""<script src='data:text/javascript;base64,{messenger_js_base64}'></script>"""
        base_html_code = base_html_code.replace("<!--message-slot-->", message_js)

        js_code_filename = os.path.join(vis_dir, "js/compare.js")
        js_string = open(js_code_filename, "r").read()
        js_b = bytes(js_string, encoding="utf-8")
        js_base64 = base64.b64encode(js_b).decode("utf-8")
        base_html_code = base_html_code.replace("<!--js-slot-->", f"""<script data-notebookMode="true" data-package="{__name__}" src='data:text/javascript;base64,{js_base64}'></script>""")

        iframe = f"""
        <iframe
            srcdoc="{html.escape(base_html_code)}"
            frameBorder="0"
            scrolling="no"
            width="100%"
            height="1500px"
            id={iframe_id}>
        </iframe>"""
        display_html(iframe, raw=True)

    def get_attribution_list(self, indices, data, scores, positive, max_num):
        if not positive: 
            indices = indices[::-1]
            data = data[::-1]
            scores = scores[::-1]

        attribution = []
        for i, (idx, d, score) in enumerate(zip(indices, data, scores)):
            if len(attribution) >= max_num : break
            if (positive and score <= 0) or (not positive and score >= 0): break

            previous_token_ids = self.get_previous_context(int(idx))
            next_token_ids = self.get_next_context(int(idx))
            attention_mask = [0] * len(previous_token_ids) + [1] * len(d["input_ids"]) + [0] * len(next_token_ids)
            token_ids = previous_token_ids + d["input_ids"] + next_token_ids
            bos_token_num, pad_token_num = 0, 0
            for token_id in token_ids:
                if token_id == self.tokenizer.bos_token_id: bos_token_num += 1
                else: break
            for token_id in token_ids[::-1]:
                if token_id == self.tokenizer.pad_token_id: pad_token_num += 1
                else: break
            token_ids = token_ids[bos_token_num:]
            if pad_token_num > 0: token_ids = token_ids[:-pad_token_num]
            text_html_code, tokens_container_id = self.get_tokens_html_code(token_ids, attention_mask)

            data_dict = {
                "text": self._ids_to_text(d["input_ids"]),
                "text_html_code": text_html_code,
                "tokens_container_id": tokens_container_id,
                "score": score,
                "data_index": idx,
            }
            for key in d:
                if key not in ["input_ids", "attention_mask", "text", "labels"]: data_dict[key] = d[key]

            attribution.append(data_dict)

        return attribution

    def get_previous_context(self, idx):
        eos_ids = [1,2,13]
        previous_idx = idx
        previous_token_ids = []
        sentence_complete_flag = False

        while previous_idx > 0:
            previous_idx -= 1
            previous_ids = self.train_dataset[previous_idx]["input_ids"]
            i = len(previous_ids) - 1
            while i >= 0:
                if previous_ids[i] in eos_ids: 
                    sentence_complete_flag = True
                    break
                previous_token_ids = [previous_ids[i]] + previous_token_ids
                i -= 1
            if sentence_complete_flag: break
        
        return previous_token_ids

    def get_next_context(self, idx):
        eos_ids = [1,2,13]
        next_idx = idx + 1
        next_token_ids = []
        sentence_complete_flag = False
        if self.train_dataset[idx]["input_ids"][-1] in eos_ids: sentence_complete_flag = True

        while next_idx < len(self.train_dataset) and not sentence_complete_flag:
            next_ids = self.train_dataset[next_idx]["input_ids"]
            i = 0
            while i < len(next_ids):
                if next_ids[i] in eos_ids: 
                    sentence_complete_flag = True
                    break
                next_token_ids.append(next_ids[i])
                i += 1
            if sentence_complete_flag: break
            next_idx += 1
        
        return next_token_ids

    def edit_text(self, generated_text):
        html_code_filename = os.path.join(self.project_dir, "visualization/html/text_edit.html")
        css_code_filename = os.path.join(self.project_dir, "visualization/css/text_edit.css")
        html_code = open(html_code_filename, "r").read()
        css_code = "<style>" + open(css_code_filename, "r").read() + "</style>"

        device = "cuda" if self.device=="auto" else self.device
        generated_ids = torch.LongTensor(self._text_to_ids(generated_text)).to(device)

        text_html_code, random_id = self.get_tokens_html_code(token_ids=generated_ids, attention_mask=np.ones(len(generated_ids[0])))
        # generate container for the generated text (should be split by word-unit not token-unit)
        random_int = random.randint(0, 1000000)
        random_id = f"tokens-container-{random_int}"
        text_html_code = f"<div class='words-container' id='{random_id}'>"

        w_idx = 0
        for p_idx, paragraph in enumerate(generated_text.split("\n")):
            for word in paragraph.split(" "):
                if "<" in word: word = word.replace("<", "&lt;")
                if ">" in word: word = word.replace(">", "&gt;")
                
                text_html_code += f"<div class='word' id='token-{random_int}-{w_idx}'>{word}</div>"
                w_idx += 1

            text_html_code += "<br>"
        text_html_code += '<div class="word dummy-word"></div></div>'
        html_code = html_code.replace("<!--text-slot-->", text_html_code)
        html_code = html_code.replace("<!--style-slot-->", css_code)

        js_code_filename = os.path.join(self.project_dir, "visualization/js/text_edit.js")
        js_string = open(js_code_filename, "r").read()
        js_b = bytes(js_string, encoding="utf-8")
        js_base64 = base64.b64encode(js_b).decode("utf-8")
        html_code = html_code.replace("<!--js-slot-->", f"""<script data-notebookMode="true" data-package="{__name__}" src='data:text/javascript;base64,{js_base64}'></script>""")
        
        iframe = f"""
        <iframe 
            srcdoc="{html.escape(html_code)}" 
            frameBorder="0" 
            width="100%"
            height="200px">
        """
        display_html(iframe, raw=True)
        pass
    
    def visualize_comparison(self, pos_max_num=10, neg_max_num=10, integration="median", score_method="datainf"):
        raise NotImplementedError
    
    

    def auto_to_cuda(self):
        self.origianl_device=self.device
        if self.device=="auto": self.device="cuda:0"

    def cuda_to_auto(self):
        if hasattr(self, "original_device") and self.original_device is not None: self.device = self.original_device






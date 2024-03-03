import os
from tqdm import tqdm 
import time 
import json 
import gc
import random

import pkgutil 
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

from IPython.display import display_html, HTML

import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer

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
            if len(self.ckpt_names) == 0: self.ckpt_names = None

        self.loaded_model_dir = None
        self.model = None

        # set tokenizer 
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if self.device != "auto": torch.cuda.set_device(self.device)

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


    def set_dataset(self, train_dataset, val_dataset=None, test_dataset=None, split_data_into_multiple_batches=False):
        self.train_dataset = self._tokenize_dataset(train_dataset, block_size=self.block_size, split_data_into_multiple_batches=split_data_into_multiple_batches)
        self.val_dataset = self._tokenize_dataset(val_dataset, block_size=self.block_size, split_data_into_multiple_batches=split_data_into_multiple_batches)
        self.test_dataset = self._tokenize_dataset(test_dataset, block_size=self.block_size, split_data_into_multiple_batches=split_data_into_multiple_batches)

    def _tokenize_dataset(self, dataset, corpus_key='text', block_size=None, split_data_into_multiple_batches=False):
        # TBU: check whether to truncate with block_size or split into multiple examples
        if dataset is None: return None 
        if type(dataset) in [dict, datasets.arrow_dataset.Dataset] and 'input_ids' in dataset: return dataset 

        if type(dataset) == list: 
            corpus_key = "text"
            dataset = {corpus_key: dataset}

        if type(dataset) == dict and corpus_key not in dataset: ### dataset structure: {"title": "text"}
            dataset = {"text": list(dataset.values()), "title": list(dataset.keys())}
        
        if type(dataset) == dict: dataset = datasets.Dataset.from_dict(dataset)

        if split_data_into_multiple_batches: 
            dataset = dataset.map(lambda data: self.tokenizer(data[corpus_key], max_length=None), remove_columns=[corpus_key])
            dataset = dataset.map(lambda data: self._split_tokens(data=data))

            final_dataset = {"input_ids": [], "attention_mask": []}
            if "title" in dataset.column_names: final_dataset["title"] = []
            
            for data in dataset:
                num_batches = len(data["input_ids"])
                for i in range(num_batches):
                    final_dataset["input_ids"].append(data["input_ids"][i])
                    final_dataset["attention_mask"].append(data["attention_mask"][i])
                    if "title" in final_dataset: final_dataset["title"].append(data["title"])

            dataset = datasets.Dataset.from_dict(final_dataset)

        else:
            dataset = dataset.map(lambda dataset: self.tokenizer(dataset[corpus_key], truncation=True, padding='max_length', max_length=block_size))

        dataset = dataset.add_column("labels", dataset["input_ids"])

        return dataset 
    
    def _split_tokens(self, data=None, tokens=None, block_size=None):
        assert data is not None or tokens is not None
        
        if tokens is None: tokens = data["input_ids"]
        attention_mask = data["attention_mask"] if data is not None and "attention_mask" in data else np.ones_like(tokens)
        title = data["title"] if data is not None and "title" in data else None
        
        if block_size is None: block_size = self.block_size
        if tokens[0] != self.tokenizer.bos_token_id: tokens = [self.tokenizer.bos_token_id] + tokens 
        if tokens[-1] != self.tokenizer.eos_token_id: tokens = tokens + [self.tokenizer.eos_token_id]

        total_length = len(tokens)
        if total_length >= block_size: total_length = (total_length // block_size) * block_size
        
        result = {
            "input_ids": [tokens[i:i+block_size] for i in range(0, total_length, block_size)],
            "attention_mask": [attention_mask[i:i+block_size] for i in range(0, total_length, block_size)],
        }
        if title is not None: result["title"] = title

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

        return peft_config

    def clear_model(self):
        if hasattr(self, "model") and self.model is not None: del self.model 
        gc.collect()
        torch.cuda.empty_cache()
    
    def check_model(self, eval_i=0, eval_prompt=None):
        # if eval_prompt is None, should input eval_i and compare the result with the groundtruth
        # if eval_prompt is given, generate with the input prompt
        self.model.eval()
        if eval_prompt is None and self.val_dataset is not None: eval_prompt = self.val_dataset[eval_i]['text']
        elif eval_prompt is None and self.test_dataset is not None: eval_prompt = self.test_dataset[eval_i]['text']
        elif eval_prompt is None: eval_prompt = self.train_dataset[eval_i]['text']
        model_input = self.tokenizer(eval_prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output = self.model.generate(**model_input, max_new_tokens=100)[0]
            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
            print("Generated:", decoded)
        
        if eval_prompt is None: print("Groundtruth:", self.val_dataset[eval_i]['answer'])

    def finetune(self, config=None, overwrite=False, bf16=False, logging_strategy="steps", logging_steps=10, save_strategy="epoch", optim="adamw_torch_fused", max_steps=-1, learning_rate=1e-4, num_train_epochs=5, gradient_accumulation_steps=2, per_device_train_batch_size=2, gradient_checkpointing=False):
        # TBU: Check if the train dataset is automatically shuffled
        if not hasattr(self, "model") or self.model is None: self.set_model(pretrained=False)
        if config is None and self.train_config is not None: config = self.train_config
        else:
            config = {
                'learning_rate': learning_rate,
                'num_train_epochs': num_train_epochs,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'per_device_train_batch_size': per_device_train_batch_size,
                'gradient_checkpointing': gradient_checkpointing,
            }
        
        profiler = nullcontext()
        training_args = TrainingArguments(
            output_dir=self.model_save_dir, 
            overwrite_output_dir=overwrite, 
            bf16=bf16, 
            logging_dir=os.path.join(self.model_save_dir, "logs"), 
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            optim=optim,
            max_steps=max_steps,
            **config)
        
        self.loaded_model_dir = None 

        # finetune
        with profiler:
            trainer = Trainer(model=self.model, args=training_args, train_dataset=self.train_dataset, data_collator=default_data_collator, callbacks=[])
            trainer.train()

        # model has been changed
        self.model.save_pretrained(self.model_save_dir)
        self.loaded_model_dir = self.model_save_dir
        ckpt_prefix = "checkpoint-"
        self.ckpt_names = [f for f in os.listdir(self.loaded_model_dir) if f.startswith(ckpt_prefix) and f[len(ckpt_prefix):].isdigit()]

    def generate(self, prompt=None, max_new_tokens=1000, return_decoded=True):
        # if not hasattr(self, "model") or self.model is None: self.set_model(pretrained=True, pretrained_dir=os.path.join(self.model_save_dir, "checkpoint-250"))  # TBU
        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**model_input, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)[0]
        if return_decoded: return self.tokenizer.decode(output, skip_special_tokens=True)
        else: return output
    
    def set_attr_text(self, prompt=None, prompt_ids=None, generated_text=None, generated_ids=None, entire_text=None, entire_text_ids=None, attr_tokens_pos=None, attr_attention_mask=None): 
        # prompt: input prompt without generated text (groundtruth answer)
        # attr_text: include both input prompt and generated text (groundtruth answer)
        assert prompt is not None or prompt_ids is not None or ((entire_text is not None or entire_text_ids is not None) and attr_tokens_pos is not None)

        self.prompt = prompt 
        self.generated_text = generated_text 
        
        if prompt_ids is None: prompt_ids = self._text_to_ids(prompt)
        if generated_ids is None: generated_ids = self._text_to_ids(generated_text)
        self.prompt_ids = torch.LongTensor(prompt_ids).to(self.device).reshape(1,-1)
        self.generated_ids = torch.LongTensor(generated_ids).to(self.device).reshape(1,-1)
        self.prompt_token_num = self.prompt_ids.shape[1]

        # if entire_text is None and entire_text_ids is None and generated_text is None and generated_ids is None:
        #     # generate attr_text from prompt
        #     entire_text_ids = self.generate(prompt=prompt, max_new_tokens=self.max_new_tokens, return_decoded=False)
        #     entire_text = self.tokenizer.decode(entire_text_ids[0], skip_special_tokens=True)
        # elif entire_text is None and entire_text_ids is None and (generated_text is not None or generated_ids is not None):
        #     pass

        if entire_text is None: entire_text = self.prompt + self.generated_text
        if entire_text_ids is None: entire_text_ids = self._text_to_ids(entire_text)
        
        self.entire_text_ids = torch.LongTensor(entire_text_ids).reshape(1,-1).to(self.device)
        self.entire_text_token_num = self.entire_text_ids.shape[1]
        self.attr_tokens_pos = torch.LongTensor(attr_tokens_pos).to(self.device) if attr_tokens_pos is not None else None
        if self.attr_tokens_pos is None: 
            self.attr_tokens_pos = torch.arange(self.prompt_token_num, self.entire_text_token_num).to(self.device)
            # code = self.select_attr_tokens_pos()
            code = None
        else: code = None
        self.attr_attention_mask = torch.Tensor(attr_attention_mask).to(self.device) if attr_attention_mask is not None else torch.ones_like(self.entire_text_ids).to(self.device)
        self.scores = None 
        
        return code

    def _text_to_ids(self, text):
        return torch.LongTensor(self.tokenizer(text, return_tensors="pt")["input_ids"])[:,1:]

    def _ids_to_text(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def set_attr_tokens_pos(self, attr_tokens_pos):
        if self.attr_tokens_pos is not None and torch.equal(self.attr_tokens_pos, torch.LongTensor(attr_tokens_pos).to(self.device)): return
        self.attr_tokens_pos = torch.LongTensor(attr_tokens_pos).to(self.device)
        self.attr_grad = None
        self.scores = None

    def get_tokens_html_code(self, token_ids, attention_mask=None):
        random_int = random.randint(0, 1000000)
        random_id = f"tokens-container-{random_int}"
        text_html_code = f"<div class='tokens-container' id='{random_id}'>"
        nobr_closed = True

        if attention_mask is None: attention_mask = np.ones(len(token_ids))

        for i, token_id in enumerate(token_ids):
            token_decoded = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            class_name = "token attended-token" if attention_mask[i] == 1 else "token unattended-token"
            if token_decoded=="<0x0A>": 
                class_name += " line-break-token"
                if nobr_closed : text_html_code += f"<div class='{class_name}' id='token-{random_int}-{i}'>&#182</div><br>"
                else: text_html_code += f"<div class='{class_name}' id='token-{random_int}-{i}'>&#182</div></nobr><br>"
                nobr_closed = True
                continue

            if "<" in token_decoded: token_decoded = token_decoded.replace("<", "&lt;")
            if ">" in token_decoded: token_decoded = token_decoded.replace(">", "&gt;")

            if "▁" == token_decoded:
                class_name += " space-token"
                token_decoded = "&nbsp;"
                html_code = f"<div class='{class_name}' id='token-{random_int}-{i}'>{token_decoded}</div>"
                if not nobr_closed: html_code = html_code + "</nobr>"
            elif "▁" == token_decoded[0]: 
                if i>0: class_name += " left-space-token"
                token_decoded = token_decoded[1:]
                html_code = f"<nobr><div class='{class_name}' id='token-{random_int}-{i}'>{token_decoded}</div>"
                nobr_closed = False 
            else:
                html_code = f"<div class='{class_name}' id='token-{random_int}-{i}'>{token_decoded}</div>"
            
            text_html_code += html_code
        
        if not nobr_closed: text_html_code += "</nobr>"
        text_html_code += "</div>"

        return text_html_code, random_id 

    def generate_html_for_str(self, token_ids, attention_mask=None):
        if type(token_ids) == torch.Tensor: token_ids = token_ids.detach().cpu().numpy()
        if token_ids.ndim==2: token_ids = token_ids.reshape(-1)
        if attention_mask is None: attention_mask = np.ones_like(token_ids)

        html_code_filename = os.path.join(self.project_dir, "visualization/html/training_data_string.html")
        css_code_filename = os.path.join(self.project_dir, "visualization/css/styles.css")
        
        html_code = open(html_code_filename, "r").read()
        css_code = "<style>" + open(css_code_filename, "r").read() + "</style>"

        text_html_code, random_id = self.get_tokens_html_code(token_ids, attention_mask)
        html_code = html_code.replace("<!--tokens-slot-->", text_html_code)
        html_code = html_code.replace("<!--style-slot-->", css_code)
        
        return html_code, random_id

    def select_attr_tokens_pos(self):
        generated_mask = np.ones([self.entire_text_token_num])
        generated_mask[:self.prompt_token_num] = 0
        html_code, tokens_container_id = self.generate_html_for_str(token_ids=self.entire_text_ids, attention_mask=generated_mask)
        
        random_int_id = tokens_container_id.split("-")[-1]
        button_code = f"""<button class="copy-selected-token-idx-button" id="copy-selected-token-idx-button-{random_int_id}" type="button">Copy Selected Token Indices</button>
        <div class="selected-token-indices" id="selected-token-indices-{random_int_id}"></div>"""
        html_code = html_code.replace("<!--button-slot-->", button_code)

        message_js = f"""
        (function() {{
            const event = new Event('selectAttrTokensPos');
            event.token_container_id = '{tokens_container_id}';
            event.prompt_token_num = {self.prompt_token_num};
            event.total_token_num = {self.entire_text_token_num};
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
            width="100%">
        """
        display_html(iframe, raw=True)

    def set_attr_grad(self):
        logsoftmax = nn.LogSoftmax(dim=-1)
        self.model.eval()
        self.model.zero_grad()
        out = self.model.base_model(self.entire_text_ids, self.attr_attention_mask)
        attr_logits = out.logits 
        attr_logprobs = logsoftmax(attr_logits) 
        attr_logprobs = attr_logprobs[0, self.attr_tokens_pos-1, self.entire_text_ids[0, self.attr_tokens_pos]]
        attr_logprob = attr_logprobs.sum() 
        self.attr_grad = torch.autograd.grad(attr_logprob, [param for param in self.model.parameters() if param.requires_grad])
        self.model.zero_grad()
        return attr_logprob.item()  # may be removed later

    def preprocess(self):
        raise NotImplementedError
    
    def save_ckpt_gradients(self, ckpt_names=None, ckpt_name=None, verbose=False):
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name] 
        else: ckpt_names = self.ckpt_names
        
        # save gradient values for the model parameters for each training data for each checkpointed model
        for ckpt in ckpt_names:
            if verbose: print("Saving gradients for", ckpt)

            ckpt_dir = os.path.join(self.model_save_dir, ckpt)

            # create grad_dir if it doesn't exists
            grad_dir = f"{ckpt_dir}/training_grads_post"
            if not os.path.exists(grad_dir):
                os.makedirs(grad_dir)
            
            # check if the gradients are already computed
            grad_files = [filename for filename in os.listdir(grad_dir) if filename.endswith(".pt") or filename.endswith(".pth")]
            grad_computed = (len(grad_files) == len(self.train_dataset))
            if grad_computed: continue

            # compute gradient and save
            self.set_model(pretrained=True, pretrained_dir=ckpt_dir)
            self.model.eval()
            
            for i, data in enumerate(tqdm(self.train_dataset)):
                # get the Delta_theta when we update the model with "data"
                input_ids = torch.LongTensor(data["input_ids"]).unsqueeze(0).to(self.device)
                attention_mask = torch.LongTensor(data["attention_mask"]).unsqueeze(0).to(self.device)
                labels = torch.LongTensor(data["labels"]).unsqueeze(0).to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                grad_loss = torch.autograd.grad(loss, [param for param in self.model.parameters() if param.requires_grad])
                torch.save(grad_loss, f"{grad_dir}/{i}.pt")
            
            self.clear_model()

    def get_datainf_scores(self, ckpt_name=None, ckpt_names=None, integrated=True, integration="mean", weighted=False, weight=None, verbose=False, override=True):
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name] 
        else: ckpt_names = self.ckpt_names

        all_scores = dict()

        for ckpt_name in ckpt_names:
            if verbose: print("Computing datainf scores for", ckpt_name)
            
            score_dir = os.path.join(self.model_save_dir, ckpt_name, "datainf.json")
            
            if override or not os.path.exists(score_dir): self.save_datainf_scores(ckpt_name=ckpt_name, override=override)
            with open(score_dir, "r") as f: scores = json.load(f)
            
            scores = np.array(scores)
            all_scores[ckpt_name] = scores
            # all_scores.append(scores)
            
        # all_scores = np.array(all_scores)
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

    def save_datainf_scores(self, ckpt_name=None, ckpt_names=None, override=True):
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name] 
        else: ckpt_names = self.ckpt_names

        for ckpt_name in ckpt_names:
            ckpt_dir =  os.path.join(self.model_save_dir, ckpt_name)
            score_dir = os.path.join(ckpt_dir, "datainf.json")
            if not override and os.path.exists(score_dir): 
                continue

            # compute datainf score
            self.set_model(pretrained=True, pretrained_dir=ckpt_dir)
            self.model.eval()
            self.save_ckpt_gradients(ckpt_name=ckpt_name)
            self.set_attr_grad()

            grad_dir = os.path.join(ckpt_dir, "training_grads_post")
            n_layers = len(self.attr_grad)
            n_train = len(self.train_dataset)
            tr_grad_norm = np.zeros([n_layers, n_train])
            
            for train_i in range(n_train):
                grad_i = torch.load(f"{grad_dir}/{train_i}.pt")
                for l in range(n_layers):
                    tr_grad_norm[l, train_i] = (grad_i[l] * grad_i[l]).sum()

            d_l = np.array([grad.numel() for grad in self.attr_grad])
            lambdas = np.sum(tr_grad_norm, axis=-1) / (10 * n_train * d_l)

            rs = [torch.zeros_like(grad) for grad in self.attr_grad]
            for train_i in range(n_train):
                grad_i = torch.load(f"{grad_dir}/{train_i}.pt")
                for l in range(n_layers):
                    c = (self.attr_grad[l] * grad_i[l]).sum() / (lambdas[l] + tr_grad_norm[l, train_i])
                    ri = (self.attr_grad[l] - c * grad_i[l]) / (n_train * lambdas[l])
                    rs[l] += ri 

            scores = np.zeros([n_train])
            for train_k in range(n_train):
                grad = torch.load(f"{grad_dir}/{train_k}.pt")
                for l in range(n_layers):
                    scores[train_k] -= (rs[l] * grad[l]).sum()

            # save the scores to score_dir 
            with open(score_dir, "w") as f: json.dump(scores.tolist(), f)

    def get_topk_training_data(self, k=3, return_scores=False, override=True, integration="mean"):
        if self.scores is None: scores = self.get_datainf_scores(integrated=True, integration=integration, override=override)
        else: scores = self.scores 

        topk_training_idx = np.argsort(-scores)[:k]
        topk_train_data = []
        for idx in topk_training_idx:
            topk_train_data.append(self.train_dataset[int(idx)])
        if return_scores: return topk_training_idx, topk_train_data, scores[topk_training_idx]
        return topk_training_idx, topk_train_data
    
    def tf_idf_topk_training_data(self, k, topk_data_text):
        # get TF from the topk training data
        tf_dict = dict()
        for data_i, text in enumerate(topk_data_text):
            for word in text.split(" "):
                word = word.strip(".,?!:;`'\"()[]\{\}<>-_=+*^&%$#@~|\\/\n\t\r\v\f").lower()
                if word in tf_dict: tf_dict[word][data_i:] += 1
                else: tf_dict[word] = np.array([0] * data_i + [1] * (k - data_i))

        # get IDF based on the entire training data
        idf_dict = {word: 0 for word in tf_dict.keys()}
        for data in self.train_dataset:
            if "text" in data: text = data["text"]
            elif "input_ids" in data: text = self._ids_to_text(data["input_ids"])
            for word in set([word.strip(".,?!:;`'\"()[]\{\}<>-_=+*^&%$#@~|\\/\n\t\r\v\f").lower() for word in re.split("\s|\.|\n\;", text)]):
                if word in idf_dict: idf_dict[word] += 1

        for word in idf_dict.keys():
            idf_dict[word] = np.log(len(self.train_dataset) / (idf_dict[word] + 1))

        # get TF-IDF
        tfidf_dict = {word: tf_dict[word] * idf_dict[word] for word in tf_dict.keys()}
        return tfidf_dict
        

    def visualize_attributed_training_data(self, pos_max_num=10, neg_max_num=10):
        vis_dir = os.path.join(self.project_dir, "visualization")

        base_html_code = open(os.path.join(vis_dir, "html/attribution.html"), "r").read()
        css_code = f"<style>{open(os.path.join(vis_dir, 'css/attribution.css'), 'r').read()}</style>"

        base_html_code = base_html_code.replace("<!--bird-in-hat-icon-->", f"{open(os.path.join(vis_dir, 'icons/bird-in-hat.svg'), 'r').read()}")
        base_html_code = base_html_code.replace("<!--up-icon-->", f"{open(os.path.join(vis_dir, 'icons/up.svg'), 'r').read()}")
        base_html_code = base_html_code.replace("<!--down-icon-->", f"{open(os.path.join(vis_dir, 'icons/down.svg'), 'r').read()}")

        styled_html_code = base_html_code.replace("<!--style-slot-->", css_code)
        prompt_html_code = f"<div class='prompt'>{self.prompt}</div>"
        # generated_text_html_code = f"<div class='generated-text'>{self.generated_text}</div>"
        generated_text_selected_indices = (self.attr_tokens_pos - self.prompt_token_num).cpu().numpy()
        generated_text_selected_mask = np.zeros(len(self.generated_ids[0]))
        generated_text_selected_mask[generated_text_selected_indices] = 1
        generated_text_html_code, generated_text_html_container_id = self.get_tokens_html_code(self.generated_ids[0], generated_text_selected_mask)
        generated_text_html_code = f"<div class='generated-text' id='{generated_text_html_container_id}'>{generated_text_html_code}</div>"
        prompt_added_html_code = styled_html_code.replace("<!--prompt-slot-->", prompt_html_code)
        generated_added_html_code = prompt_added_html_code.replace("<!--generated-text-slot-->", generated_text_html_code)

        indices, data, scores = self.get_topk_training_data(k=len(self.train_dataset), override=False, return_scores=True, integration="median")  # TODO: update override
        scores = [float("{:.2e}".format(score)) for score in scores]
        
        # split train_idx, train_data, topk_scores into positive and negative
        positive_attribution, negative_attribution = [], []
        for i, (idx, d, score) in enumerate(zip(indices, data, scores)):
            if len(positive_attribution) >= pos_max_num : break
            if score <= 0: break

            previous_token_ids = self.get_previous_context(int(idx))
            next_token_ids = self.get_next_context(int(idx))
            attention_mask = [0] * len(previous_token_ids) + [1] * len(d["input_ids"]) + [0] * len(next_token_ids)
            token_ids = previous_token_ids + d["input_ids"] + next_token_ids
            text_html_code, tokens_container_id = self.get_tokens_html_code(token_ids, attention_mask)
            
            data_dict = {
                "text": self._ids_to_text(d["input_ids"]),
                "text_html_code": text_html_code,
                "tokens_container_id": tokens_container_id,
                "title": d["title"],
                "score": score,
                "data_index": idx,
            }
            
            if score > 0 and len(positive_attribution) < pos_max_num: positive_attribution.append(data_dict)
            
        indices = indices[::-1]
        data = data[::-1]
        scores = scores[::-1]

        for i, (idx, d, score) in enumerate(zip(indices, data, scores)):
            if len(negative_attribution) >= neg_max_num : break
            if score >= 0: break

            previous_token_ids = self.get_previous_context(int(idx))
            next_token_ids = self.get_next_context(int(idx))
            attention_mask = [0] * len(previous_token_ids) + [1] * len(d["input_ids"]) + [0] * len(next_token_ids)
            token_ids = previous_token_ids + d["input_ids"] + next_token_ids
            text_html_code, tokens_container_id = self.get_tokens_html_code(token_ids, attention_mask)
            
            data_dict = {
                "text": self._ids_to_text(d["input_ids"]),
                "text_html_code": text_html_code,
                "tokens_container_id": tokens_container_id,
                "title": d["title"],
                "score": score,
                "data_index": idx,
            }
            
            if score < 0 and len(negative_attribution) < neg_max_num: negative_attribution.append(data_dict)

        # histogram
        n_bins = 12
        counts, bins = np.histogram(scores, bins=n_bins)
        for data_dict in positive_attribution:
            data_dict["score_histogram_bin"] = min(np.digitize(data_dict["score"], bins)-1, n_bins-1)
        for data_dict in negative_attribution:
            data_dict["score_histogram_bin"] = np.digitize(data_dict["score"], bins)-1

        scoreHistogramCounts = [[bins[i], bins[i+1], counts[i]] for i in range(len(counts))]

        # tf-idf
        pos_tf_idf = self.tf_idf_topk_training_data(pos_max_num, [d["text"] for d in positive_attribution])
        neg_tf_idf = self.tf_idf_topk_training_data(neg_max_num, [d["text"] for d in negative_attribution])
        
        # sort TF-IDF
        N = 10
        topN_word_indices_for_each_topk_pos = np.argsort(-np.array(list(pos_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(pos_tf_idf.keys())
        topN_words_for_each_topk_pos = [[[word_indices[i], pos_tf_idf[word_indices[i]][j]] for i in topN_word_indices_for_each_topk_pos[j]] for j in range(pos_max_num)]

        topN_word_indices_for_each_topk_neg = np.argsort(-np.array(list(neg_tf_idf.values())), axis=0)[:N,:].T
        word_indices = list(neg_tf_idf.keys())
        topN_words_for_each_topk_neg = [[[word_indices[i], neg_tf_idf[word_indices[i]][j]] for i in topN_word_indices_for_each_topk_neg[j]] for j in range(neg_max_num)]


        # send all the positively/negatively attributed data and their scores through message
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
            frameBorder="0" 
            width="100%"
            height="2000px">
        </iframe>"""
        display_html(iframe, raw=True)

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
    
    def compute_token_importance(self, data=None, ckpt_name=None, ckpt_names=None):
        print("Computing token importance...")
        if ckpt_names is not None: ckpt_names = ckpt_names
        elif ckpt_name is not None: ckpt_names = [ckpt_name]
        else: ckpt_names = self.ckpt_names 

        focused_tokens = torch.LongTensor(data["input_ids"]).reshape(1,-1)
        focused_labels = torch.LongTensor(data["labels"]).unsqueeze(0).to(self.device)
        focused_attention_mask = torch.LongTensor(data["attention_mask"]).unsqueeze(0).to(self.device)

        position_ids = None 
        inputs_embeds = None

        seq_length = focused_tokens.shape[1]
        past_key_values_length = 0

        if position_ids is None:
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else: position_ids = position_ids.view(-1, seq_length).long()

        token_importance_for_each_ckpt = []
        for ckpt_name in ckpt_names:
            ckpt_dir = os.path.join(self.model_save_dir, ckpt_name)
            self.set_model(pretrained=True, pretrained_dir=ckpt_dir)
            self.model.eval()
            self.model.zero_grad()

            self.set_attr_grad()
            self.model.zero_grad()
            
            inputs_embeds = self.model.base_model.model.model.embed_tokens(focused_tokens)
            inputs_embeds.requires_grad = True
            self.model.zero_grad()

            out = self.model(inputs_embeds=inputs_embeds, attention_mask=focused_attention_mask, labels=focused_labels)
            loss = out.loss
            grad_loss = torch.autograd.grad(loss, [param for param in self.model.parameters() if param.requires_grad], create_graph=True)
            self.model.zero_grad()

            inner = 0
            for g1, g2 in zip(self.attr_grad, grad_loss):
                inner += (g1*g2).sum()
            embedding_grad = torch.autograd.grad(inner, inputs_embeds)[0]  # 1 x num_tokens x embedding_dim
            embedding_grad_norm = torch.sqrt(torch.sum(embedding_grad ** 2, dim=-1))[0]
            token_importance_for_each_ckpt.append(embedding_grad_norm.detach().cpu().numpy())

        token_importance = np.mean(token_importance_for_each_ckpt, axis=0)
        return token_importance

    def visualize_token_importance(self, data=None, ckpt_name=None, ckpt_names=None):
        def colorize_tokens(tokens, weights, attention_mask=None):
            if type(tokens) == torch.Tensor: tokens = tokens.detach().cpu().numpy()
            if tokens.ndim==2: tokens = tokens.reshape(-1)
            if type(weights) == torch.Tensor: weights = weights.detach().cpu().numpy()
            if attention_mask is None: attention_mask = np.ones_like(tokens)
            if attention_mask.ndim==2: attention_mask = attention_mask.reshape(-1)
            assert attention_mask.shape[0] == tokens.shape[0]
            
            cmap = matplotlib.colormaps.get_cmap('Reds')
            template = '<span style="color: black; background-color: {}; display: inline-block">{}</span>'
            colored_string = ''
            
            for token, weight, masked in zip(tokens, weights, attention_mask):
                if not masked: continue
                color = matplotlib.colors.rgb2hex(cmap(weight)[:3]) + "80"
                token_decoded = self.tokenizer.convert_ids_to_tokens([token])[0]
                if token_decoded=="<0x0A>": 
                    colored_string += "<br>"
                    continue
                if "▁" in token_decoded: token_decoded = token_decoded.replace("▁", "&nbsp;")
                if "<" in token_decoded: token_decoded = token_decoded.replace("<", "&lt;")
                if ">" in token_decoded: token_decoded = token_decoded.replace(">", "&gt;")
                colored_string += template.format(color, token_decoded)
            
            return colored_string

        ignore_token_ids = [1,2,13]
        focused_tokens = torch.LongTensor(data["input_ids"]).reshape(1,-1)
        focused_attention_mask = torch.LongTensor(data["attention_mask"]).unsqueeze(0).to(self.device)
        not_ignored = torch.prod(torch.vstack([focused_tokens != ignore_token for ignore_token in ignore_token_ids]), dim=0)
        token_importance = self.compute_token_importance(data, ckpt_name=ckpt_name, ckpt_names=ckpt_names)
        scores = torch.from_numpy(token_importance) * not_ignored
        scores = scores / torch.max(scores)
        code = colorize_tokens(focused_tokens, scores, focused_attention_mask[0])
        return code
    








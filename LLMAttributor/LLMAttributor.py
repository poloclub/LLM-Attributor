import os
from tqdm import tqdm 
import time 
import json 
import gc

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, default_data_collator
import datasets 
from contextlib import nullcontext
from peft import PeftConfig, PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from IPython.display import display, HTML

import matplotlib

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
    
    def set_attr_prompt(self, prompt=None, prompt_ids=None, generated_text=None, generated_ids=None, attr_text=None, attr_text_ids=None, attr_tokens_pos=None, attr_attention_mask=None): 
        # prompt: input prompt without generated text (groundtruth answer)
        # attr_text: include both input prompt and generated text (groundtruth answer)
        assert prompt is not None or prompt_ids is not None or ((attr_text is not None or attr_text_ids is not None) and attr_tokens_pos is not None)
        
        if prompt_ids is None: prompt_ids = self._text_to_ids(prompt)
        prompt_ids = torch.LongTensor(prompt_ids).to(self.device).reshape(1,-1)
        self.attr_prompt_token_len = prompt_ids.shape[1]

        if attr_text is None and attr_text_ids is None and generated_text is None and generated_ids is None:
            # generate attr_text from prompt
            attr_text_ids = self.generate(prompt=prompt, max_new_tokens=self.max_new_tokens, return_decoded=False)
            attr_text = self.tokenizer.decode(attr_text_ids[0], skip_special_tokens=True)
        elif attr_text is None and attr_text_ids is None and (generated_text is not None or generated_ids is not None):
            pass

        if attr_text_ids is None: attr_text_ids = self._text_to_ids(attr_text)
        self.attr_text_ids = torch.LongTensor(attr_text_ids).reshape(1,-1).to(self.device)
        attr_text_token_len = self.attr_text_ids.shape[1]
        self.attr_tokens_pos = torch.LongTensor(attr_tokens_pos).to(self.device) if attr_tokens_pos is not None else None
        if self.attr_tokens_pos is None: 
            self.attr_tokens_pos = torch.arange(self.attr_prompt_token_len, attr_text_token_len).to(self.device)
            code = self.select_attr_tokens_pos()
        else: code = None
        self.attr_attention_mask = torch.Tensor(attr_attention_mask).to(self.device) if attr_attention_mask is not None else torch.ones_like(self.attr_text_ids).to(self.device)
        self.scores = None 
        
        return code

    def _text_to_ids(self, text):
        return torch.LongTensor(self.tokenizer(text, return_tensors="pt")["input_ids"])

    def _ids_to_text(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def set_attr_tokens_pos(self, attr_tokens_pos):
        if self.attr_tokens_pos is not None and torch.equal(self.attr_tokens_pos, torch.LongTensor(attr_tokens_pos).to(self.device)): return
        self.attr_tokens_pos = torch.LongTensor(attr_tokens_pos).to(self.device)
        self.attr_grad = None
        self.scores = None
    
    def select_attr_tokens_pos(self):
        def generate_html_for_str(tokens=None, prompt_len=None):
            if type(tokens) == torch.Tensor: tokens = tokens.detach().cpu().numpy()
            if tokens.ndim==2: tokens = tokens.reshape(-1)
            assert tokens is not None

            html_code = "<div>"
            
            for i, token in enumerate(tokens): 
                token_decoded = self.tokenizer.convert_ids_to_tokens([token])[0]
                if token_decoded=="<0x0A>": 
                    html_code += "<br>"
                    continue
                if "▁" in token_decoded: token_decoded = token_decoded.replace("▁", "&nbsp;")
                if "<" in token_decoded: token_decoded = token_decoded.replace("<", "&lt;")
                if ">" in token_decoded: token_decoded = token_decoded.replace(">", "&gt;")

                text_color = "#000000"
                cursor = "pointer"
                if prompt_len is not None and i < prompt_len: 
                    text_color = "#808080"
                    cursor = "auto"
                html_code += f"<div style='color: {text_color}; display: inline-block; cursor: {cursor}; user-select: none; user-drag: none; -webkit-user-drag: none; -moz-user-select: none; -webkit-user-select: none; -ms-user-select: none;' id='token-{i}' onmousedown='mousedown_token(this, {prompt_len})' onmouseenter='mouseenter_token(this, {prompt_len})' onmouseout='mouseout_token(this)'>{token_decoded}</div>"

            html_code += "</div>"
            return html_code 
        
        total_token_num = self.attr_text_ids.shape[1]
        highlight_color = "#ff0000"
        javascript_code = f"""
        <script type="text/Javascript">
            window.startToken = -1;
            window.beingDragged = false;
            window.selecting = false;
            
            document.addEventListener('mouseup', (e) => {{
                if (window.beingDragged) {{
                    for (let i={self.attr_prompt_token_len}; i<{total_token_num}; i++) {{
                        if (document.getElementById(`token-${{i}}`) == null) continue;
                        document.getElementById(`token-${{i}}`).selected = document.getElementById(`token-${{i}}`).newSelected;
                        document.getElementById(`token-${{i}}`).style.color = document.getElementById(`token-${{i}}`).selected?"{highlight_color}":"black";
                    }}
                }}
                
                window.startToken = -1;
                window.beingDragged = false;
                window.selecting = false;
                
            }})
            
            function mousedown_token(token, prompt_len) {{
                // if mouse being clicked, from start to this one, change to red
                let clickedTokenIdx = Number(token.id.split("-")[1]);
                
                if (clickedTokenIdx >= prompt_len) {{
                    window.beingDragged = true;
                    
                    if (token.newSelected) {{token.newSelected = false; window.selecting=false;}}
                    else {{token.newSelected = true; window.selecting=true;}}

                    window.startToken = clickedTokenIdx;
                    if (window.selecting) token.style.color = "{highlight_color}";
                    else token.style.color = "#000000";
                }}
            }}
            
            function mouseenter_token(token, prompt_len) {{
                if ((Number(token.id.split("-")[1])) >= prompt_len) token.style.backgroundColor = "{highlight_color}80"; //highlight this one's background always
                if (window.beingDragged) {{
                    let enteredTokenIdx = Number(token.id.split("-")[1]);
                    let start = Math.min(enteredTokenIdx, window.startToken);
                    let end = Math.max(enteredTokenIdx, window.startToken);
                    for (let i=prompt_len; i<{total_token_num}; i++) {{
                        if (document.getElementById(`token-${{i}}`) == null) continue;
                        if ((i>=start)&&(i<=end)) {{
                            document.getElementById(`token-${{i}}`).newSelected = window.selecting;
                            document.getElementById(`token-${{i}}`).style.color = window.selecting?"{highlight_color}":"black";
                        }}
                        else {{
                            document.getElementById(`token-${{i}}`).newSelected = document.getElementById(`token-${{i}}`).selected;
                            document.getElementById(`token-${{i}}`).style.color = document.getElementById(`token-${{i}}`).newSelected?"{highlight_color}":"black";
                        }}
                    }}
                }}
            }}
            
            function mouseout_token(token) {{
                token.style.backgroundColor = "#00000000";
            }}
            
            function showHighlightedTokenIndices() {{
                let highlightedTokenIndices = [];
                for (let i=0; i<{total_token_num}; i++) {{
                    if (document.getElementById(`token-${{i}}`) == null) {{
                        if (i == 0) continue; // if the only token or first, skip
                        else if (i == {total_token_num} - 1) {{
                            if (highlightedTokenIndices.includes(i-1)) highlightedTokenIndices.push(i)
                        }}
                        else {{
                            if (highlightedTokenIndices.includes(i-1) && (document.getElementById(`token-${{i+1}}`)!=null) && (document.getElementById(`token-${{i+1}}`).selected)) highlightedTokenIndices.push(i)
                        }}
                    }}
                    else if (document.getElementById(`token-${{i}}`).selected) {{
                        highlightedTokenIndices.push(i);
                    }}
                }}

                let highlightedTokenIndicesStr = highlightedTokenIndices.toString();
                document.getElementById("highlighted-token-indices").innerHTML = highlightedTokenIndicesStr;
                navigator.clipboard.writeText("["+highlightedTokenIndicesStr+"]");
            }}
        </script>
        """

        html_code = generate_html_for_str(tokens=self.attr_text_ids, prompt_len=self.attr_prompt_token_len)
        html_code += """
        <button onclick="showHighlightedTokenIndices()" style="margin-top: 5px; font-size: 14px;">Copy Highlighted Token Indices</button>
        <div id="highlighted-token-indices" style="display: inline-block; padding-left: 3px; font-size: 14px;"></div>
        """
        
        return html_code + javascript_code
    
    def set_attr_grad(self):
        logsoftmax = nn.LogSoftmax(dim=-1)
        self.model.eval()
        self.model.zero_grad()
        out = self.model.base_model(self.attr_text_ids, self.attr_attention_mask)
        attr_logits = out.logits 
        attr_logprobs = logsoftmax(attr_logits) 
        attr_logprobs = attr_logprobs[0, self.attr_tokens_pos-1, self.attr_text_ids[0, self.attr_tokens_pos]]
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

    def get_datainf_scores(self, ckpt_name=None, ckpt_names=None, integrated=True, weighted=False, weight=None, verbose=False, override=True):
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
        elif integrated: 
            # integrated_scores = np.mean(np.abs(list(all_scores.values())), axis=0)  
            integrated_scores = np.mean(list(all_scores.values()), axis=0)  
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

    def get_topk_training_data(self, k=10, return_scores=False):
        if self.scores is None: scores = self.get_datainf_scores(integrated=True)
        topk_training_idx = np.argsort(-scores)[:k]
        topk_train_data = []
        for idx in topk_training_idx:
            topk_train_data.append(self.train_dataset[int(idx)])
        if return_scores: return topk_training_idx, topk_train_data, scores[topk_training_idx]
        return topk_training_idx, topk_train_data
    
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
    








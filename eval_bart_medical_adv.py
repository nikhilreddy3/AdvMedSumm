import pandas as pd
import numpy as np
import torch
import argparse
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader
import nltk
import inspect
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    logging,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartForConditionalGeneration
)

from datasets import load_dataset
import evaluate
from questeval.questeval_metric import QuestEval
from summac.model_summac import SummaCZS

import os
os.environ["WANDB_PROJECT"]="adv_lang_summ_evaluation"

set_seed(41)

ATTACK_EPS = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

metric = evaluate.load("rouge")

summary_data = []

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


class myDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=1024) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], 
                                   max_length=self.max_len, 
                                   padding="max_length", 
                                   truncation=True)
        label = self.tokenizer(text_target =self.labels[idx], 
                                   max_length=self.max_len, 
                                   padding="max_length", 
                                   truncation=True)
        
        label["input_ids"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in label['input_ids']]

        encoding['labels'] = label["input_ids"]
        return encoding
    
    def __len__(self):
        return len(self.texts)

class MyClassModel(BartForConditionalGeneration):
    def __init__(self, config, epsilon=ATTACK_EPS):
        super(MyClassModel, self).__init__(config)
        self.encoder = self.model.get_encoder()
        self.decoder = self.model.get_decoder()
        self.epsilon = epsilon
        #self.feat_min = clip_min
        #self.feat_max = clip_max

    def emb_perturb(self, x,y,
                    decoder_input_ids, 
                    decoder_attention_mask, 
                    attention_mask,
                    decoder_inputs_embeds,
                    use_cache,
                    return_dict
                    ):
        #x = x.detach().clone().to(device)
        x["last_hidden_state"] = x["last_hidden_state"].detach().clone().to(device)
        y = y.detach().clone().to(device) 

        x["last_hidden_state"].requires_grad = True
        #decoder_outputs = self.forward_decoder(x)
        decoder_outputs = self.forward_decoder(decoder_input_ids=decoder_input_ids, 
                                               decoder_attention_mask=decoder_attention_mask, 
                                               encoder_outputs=x,
                                               attention_mask=attention_mask,
                                               decoder_inputs_embeds=decoder_inputs_embeds,
                                               use_cache=use_cache,
                                               return_dict=return_dict
                                               )

        lm_logits = self.get_lm_head_outs(decoder_outputs=decoder_outputs)

        criteron = CrossEntropyLoss()
        loss = criteron(lm_logits.view(-1, self.config.vocab_size), y.view(-1))
        grads = torch.autograd.grad(loss, x.last_hidden_state, retain_graph=False, create_graph=False)[0]
        sign = grads.sign()
                        
        x_adv = x["last_hidden_state"] + self.epsilon*sign
        x_adv = x_adv.detach()
        return x_adv


    def forward_encoder(self, input_ids, attention_mask,
                        head_mask: Optional[torch.Tensor] = None,
                        inputs_embeds: Optional[torch.FloatTensor] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        return_dict: Optional[bool] = None,
                        ):
        #encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return encoder_outputs

    def forward_decoder(self, decoder_input_ids, decoder_attention_mask, encoder_outputs,
                        attention_mask: Optional[torch.Tensor] = None,
                        decoder_head_mask: Optional[torch.Tensor] = None,
                        cross_attn_head_mask: Optional[torch.Tensor] = None,
                        past_key_values: Optional[List[torch.FloatTensor]] = None,
                        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                        use_cache: Optional[bool] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        return_dict: Optional[bool] = None,                        
                        ):

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_lm_head_outs(self, decoder_outputs):
        lm_logits = self.lm_head(decoder_outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        return lm_logits        
    
    def get_lm_loss(self, lm_logits, labels):
        labels = labels.to(lm_logits.device)
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))        
        return masked_lm_loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
    # ) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                #logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
                pass
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        if encoder_outputs is None:
            encoder_outputs = self.forward_encoder(input_ids=input_ids, 
                                                attention_mask=attention_mask, 
                                                return_dict=return_dict)

        if self.training:
            encoder_outputs["last_hidden_state"] = self.emb_perturb(encoder_outputs, labels,
                                               decoder_input_ids=decoder_input_ids, 
                                                decoder_attention_mask=decoder_attention_mask, 
                                                #encoder_outputs=encoder_outputs,
                                                attention_mask=attention_mask,
                                                decoder_inputs_embeds=decoder_inputs_embeds,
                                                use_cache=use_cache,
                                                return_dict=return_dict)

        decoder_outputs = self.forward_decoder(decoder_input_ids=decoder_input_ids, 
                                               decoder_attention_mask=decoder_attention_mask, 
                                               encoder_outputs=encoder_outputs,
                                               attention_mask=attention_mask,
                                               decoder_inputs_embeds=decoder_inputs_embeds,
                                               use_cache=use_cache,
                                               return_dict=return_dict
                                               )

        lm_logits = self.get_lm_head_outs(decoder_outputs=decoder_outputs)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = self.get_lm_loss(lm_logits, labels)


        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # return lm_logits
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def get_hf_model(model_ckpt):
    config = AutoConfig.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, config = config)
    # model = MyClassModel.from_pretrained(model_ckpt, config = config)
    return model

def eval_hqs_model(model, val_data, tokenizer, batch_size=16, base_dir='./'):
    traing_args = Seq2SeqTrainingArguments(
        output_dir = f'{base_dir}/models/',
        per_device_eval_batch_size = batch_size,   
        report_to="wandb",
        predict_with_generate = True,
        generation_max_length = 62,
        generation_num_beams = 6,
        # save_safetensors=True
    )

    result = defaultdict(float)
    
    trainer = Seq2SeqTrainer(model=model,
                             args=traing_args,
                             )

    val_result = trainer.predict(val_data)
    
    preds = np.where(val_result.predictions != -100, val_result.predictions, tokenizer.pad_token_id)
    generated_text = tokenizer.batch_decode(preds, skip_special_tokens=True)

    val_pred_data = {
    'texts': val_data.texts,
    'labels': val_data.labels,
    'generated_texts': generated_text
    }
    
    # Create a DataFrame from the data
    df = pd.DataFrame(val_pred_data)
    
    # Save the DataFrame as a CSV file
    df.to_csv('hqs_val_pred_adv_data.csv', index=False)
    
    questeval = QuestEval(no_cuda=False)
    score = questeval.corpus_questeval(hypothesis=generated_text, sources=val_data.texts)
    result["val_questeval"] = score

    model = SummaCZS(granularity="document", model_name="vitc", device="cuda")

    final_score = []
    for source_document, summary in zip(val_data.texts, generated_text):
        score = model.score([source_document], [summary])
        final_score.append(score["scores"][0])

    print(final_score)
    result["val_summacc"] = sum(final_score) / len(final_score)

    return result
    
def eval_rrs_model(model, val_data, indiana_data, tokenizer, batch_size=16, base_dir='./'):
    traing_args = Seq2SeqTrainingArguments(
        output_dir = f'{base_dir}/models/',
        per_device_eval_batch_size = batch_size,   
        report_to="wandb",
        predict_with_generate = True,
        generation_max_length = 62,
        generation_num_beams = 6,
        # save_safetensors=True
    )

    result = defaultdict(float)
    
    trainer = Seq2SeqTrainer(model=model,
                             args=traing_args,
                             )

    val_result = trainer.predict(val_data)
    
    preds = np.where(val_result.predictions != -100, val_result.predictions, tokenizer.pad_token_id)
    generated_text = tokenizer.batch_decode(preds, skip_special_tokens=True)

    val_pred_data = {
    'texts': val_data.texts,
    'labels': val_data.labels,
    'generated_texts': generated_text
    }
    
    # Create a DataFrame from the data
    df = pd.DataFrame(val_pred_data)
    
    # Save the DataFrame as a CSV file
    df.to_csv('rrs_val_pred_adv_data.csv', index=False)
    
    questeval = QuestEval(no_cuda=False)
    score = questeval.corpus_questeval(hypothesis=generated_text, sources=val_data.texts)
    result["val_questeval"] = score

    model = SummaCZS(granularity="document", model_name="vitc", device="cuda")

    final_score = []
    for source_document, summary in zip(val_data.texts, generated_text):
        score = model.score([source_document], [summary])
        final_score.append(score["scores"][0])

    print(final_score)
    result["val_summacc"] = sum(final_score) / len(final_score)

    indiana_result = trainer.predict(indiana_data)

    preds = np.where(indiana_result.predictions != -100, indiana_result.predictions, tokenizer.pad_token_id)
    generated_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    indiana_pred_data = {
    'texts': val_data.texts,
    'labels': val_data.labels,
    'generated_texts': generated_text
    }
    
    # Create a DataFrame from the data
    df = pd.DataFrame(indiana_pred_data)
    
    # Save the DataFrame as a CSV file
    df.to_csv('indiana_pred_adv_data.csv', index=False)

    questeval = QuestEval(no_cuda=False)
    score = questeval.corpus_questeval(hypothesis=generated_text, sources=indiana_data.texts)
    result["indiana_questeval"] = score

    final_score = []
    for source_document, summary in zip(indiana_data.texts, generated_text):
        score = model.score([source_document], [summary])
        final_score.append(score["scores"][0])

    print(final_score)
    result["indiana_summacc"] = sum(final_score) / len(final_score)

    return result


ckpt_list = ["1145","2288"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adv perturbations for bias and OOD')
    parser.add_argument('-m', '--model', default='facebook/bart-base', type = str)
    parser.add_argument('-x', '--dataset', default='rrs', type = str)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-l', '--max_len', default=1024, type=int)
    parser.add_argument('-v', '--verbose_setting', default=1, type=int)
    parser.add_argument('-d', '--base_dir', default='./', type = str)
    parser.add_argument('-c', '--model_dir', default='models/'', type = str)

    args = parser.parse_args()

    if args.verbose_setting:
        logging.set_verbosity_debug()
        
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if args.dataset == "rrs":
        def read_rrsdataset(filename):
                with open(filename, 'r') as file:
                    data = json.load(file)
                
                df = pd.DataFrame(data)
                df = df.rename(columns={'study_id': 'Study ID', 'subject_id': 'Subject ID',
                                        'findings': 'Findings', 'impression': 'Impression',
                                        'background': 'Background'})
                print(df.head())
                df['Text'] = df['Findings'] + ' ' + df['Background']
                df['Summary'] = df['Impression']
                df.drop(['Findings', 'Impression', 'Background'], axis=1, inplace=True)
                return df['Text'].tolist(), df['Summary'].tolist()
            
        indiana_text, indiana_summary =  read_rrsdataset("indiana_dev.json")
        val_text, val_summary =  read_rrsdataset("dev.json")
        
        val_data = myDataset(val_text,
                       val_summary,
                       tokenizer, args.max_len
                       )
        indiana_data = myDataset(indiana_text,
                               indiana_summary,
                               tokenizer, args.max_len
                               )
    elif args.dataset == "hqs":
        def read_hqsdataset(filename):
            # Read the Excel file
            data = pd.read_excel(filename)
            text = data['NLM Question'].tolist()
                
            summaries = data['Summary'].tolist()
        
            return text, summaries
            
        val_text, val_summary =  read_hqsdataset("MEDIQA2021-Task1-QuestionSummarization-ValidationSet.xlsx")
    
        val_data = myDataset(val_text,
                               val_summary,
                               tokenizer, args.max_len
                               )
        
        
    for ckpt in ckpt_list:
        print("*********************************Checkpoint: "+ckpt)
        model_ckpt = f"{args.model_dir}checkpoint-{ckpt}"
        model = get_hf_model(model_ckpt)
        if args.dataset == "rrs":
            eval_res = eval_rrs_model(model, val_data, indiana_data, tokenizer = tokenizer, batch_size=args.batch_size, base_dir='./')
        elif args.dataset == "hqs":
            eval_res = eval_hqs_model(model, val_data,tokenizer = tokenizer, batch_size=args.batch_size, base_dir='./')

        with open(f"{ckpt}_val_adv+"+args.dataset+".json", 'w') as f:
            json.dump(eval_res, f)

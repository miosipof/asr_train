from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainerCallback, Seq2SeqTrainer, TrainerCallback
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset
from dataclasses import dataclass
import evaluate
from typing import Any, Dict, List, Union
import torch
from peft import get_peft_model, PeftConfig, PeftModel
import time
import os
import logging
log = logging.getLogger(__name__)

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

class Training:
    training_args: Any
    
    def __init__(self, pretrained_model_id, finetuned_model_id, training_args, language="en", task="transcribe"):
        self.training_args = training_args
        self.pretrained_model_id = pretrained_model_id
        self.finetuned_model_id = finetuned_model_id
        self.language = language
        self.task = task
        self.normalizer = BasicTextNormalizer()

        self.prepare_model()
    

    def prepare_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.pretrained_model_id)
        self.processor = WhisperProcessor.from_pretrained(self.pretrained_model_id, language=self.language, task=self.task)
        
        self.model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

        return self.model

    def apply_lora(self,lora_config):
        self.lora_config = lora_config
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.generation_config.language = self.language
        log.info(f"Using LoRA with parameters: {self.model.print_trainable_parameters()}")

        return self.model

    def compute_metrics(self,pred):
        
        metric = evaluate.load("wer")

        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        pred_str = [self.normalizer(pred) for pred in pred_str]
        label_str = [self.normalizer(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
        label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]
        
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)    
        return {"wer": wer}


    def train(self,ds):

        trainer = Seq2SeqTrainer(
            args=self.training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["val"],
            data_collator=DataCollator(processor=self.processor),
            tokenizer=self.processor.feature_extractor,
            callbacks=[
                ShuffleCallback()],
            model=self.model,
            compute_metrics=self.compute_metrics
        )  

        start=time.time()
        trainer.train()
        end=time.time()

        print(f"Training completed! \nTime elapsed: {end-start}. \nFine-tuned model name: {self.finetuned_model_id}")


        kwargs = {
            "finetuned_from": self.pretrained_model_id,
            "language": self.language,
        }
        
        try:
            trainer.push_to_hub(token = os.environ["HF_TOKEN"], **kwargs)
            print("The model has been uploaded to hub")
        except Exception as e:
            print(e)

    def eval(self,ds):
        
        if self.lora_config:
            self.peft_config = PeftConfig.from_pretrained(self.finetuned_model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.peft_config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(self.model, self.finetuned_model_id)
            self.model.config.use_cache = True
            self.processor = WhisperProcessor.from_pretrained(self.pretrained_model_id, language=self.language, task=self.task)
        
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(self.finetuned_model_id)
            self.processor = WhisperProcessor.from_pretrained(self.finetuned_model_id, language=self.language, task=self.task)

        
        for ds_name, ds_split in ds.items():
            if ds_name == "test":
                eval_ds = ds["test"]
            
        if eval_ds:
            print("Using Test split")
        else:
            eval_ds = ds["val"]
            print("Using Validation split")
            
        trainer = Seq2SeqTrainer(
            args=self.training_args,
            train_dataset=ds["train"],
            eval_dataset=eval_ds,
            data_collator=DataCollator(processor=self.processor),
            tokenizer=self.processor.feature_extractor,
            callbacks=[
                ShuffleCallback()],
            model=self.model,
            compute_metrics=self.compute_metrics
        )  

        result = trainer.evaluate()

        return result
    
# trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)


@dataclass
class DataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
        
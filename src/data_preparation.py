from datasets import load_dataset, Audio, load_from_disk, IterableDatasetDict, concatenate_datasets

from transformers import WhisperProcessor, GenerationConfig
from huggingface_hub import login
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import numpy as np
import librosa

from typing import Any, Dict, List, Union
import os
import logging

log = logging.getLogger(__name__)

class Preprocessor:
    processor: Any  
    pretrained_model_id: str
    sampling_rate: int
    max_input_length: float
    dataset_name: str
    text_column_name: str
    language: str
    task: str
    
    def __init__(self, dataset_name, text_column_name, pretrained_model_id = "openai/whisper-small", language="en", task="transcribe"):

        try:
            login(token=os.environ['HF_TOKEN'])
            log.info("HuggingFace login successful")
        except Exception as e:
            log.error(f"Error during HuggingFace login: {str(e)}")
            raise        
        
        self.pretrained_model_id = pretrained_model_id
        self.sampling_rate = 16000
        self.max_input_length = 30.0
        self.dataset_name = dataset_name
        self.text_column_name = text_column_name
        self.language = language
        self.task = task

        self.load_processor()

        log.info(f"Preprocessor initialized for {dataset_name} dataset, {pretrained_model_id} model")
    
    def load(self, test_size=0.1):
        hf_dataset = load_dataset(self.dataset_name, streaming=False)
        log.info(f"Dataset loaded from {self.dataset_name}")
        raw_datasets = IterableDatasetDict()

        if "train" in hf_dataset:
            raw_datasets["train"] = hf_dataset["train"]
        else:
            raw_datasets["train"] = hf_dataset
        
        train_val_split = raw_datasets["train"].train_test_split(test_size=test_size)
        raw_datasets["train"] = train_val_split["train"]
        raw_datasets["val"] = train_val_split["test"]

        len_train = len(raw_datasets["train"])
        len_val = len(raw_datasets["val"])
        
        log.info(f"Generated Train split {len_train}, Validation split {len_val}")
        
        if "test" in hf_dataset:
            raw_datasets["test"] = hf_dataset["test"]
            len_test = len(raw_datasets["test"])
            log.info(f"Generated Test split({len_test})")

        else:
            log.info("Test datasen not specified")
            
        for ds in raw_datasets:
            raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

        log.info(f"Dataset casted to audio with sampling rate = {self.sampling_rate}")

        return raw_datasets


    def load_processor(self):   
        self.processor = WhisperProcessor.from_pretrained(self.pretrained_model_id, language=self.language, task=self.task)
        self.generation_config = GenerationConfig.from_pretrained(self.pretrained_model_id)
        self.normalizer = BasicTextNormalizer()
        
    def prepare_dataset(self,batch):

        do_lower_case = True
        do_remove_punctuation = True
            
        audio = batch["audio"]
            
        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
            
        # optional pre-processing steps
        transcription = batch[self.text_column_name]
        if do_lower_case:
            transcription = transcription.lower()
        if do_remove_punctuation:
            transcription = self.normalizer(transcription).strip()
            
        # encode target text to label ids
        batch["labels"] = self.processor.tokenizer(transcription).input_ids
            
        return batch

    def save_hf_datasets(self,dataset_dict,prefix):
        for ds_name, iterable_ds in dataset_dict.items():
            iterable_ds.save_to_disk("assets/"+ds_name+"_"+prefix+".hf")

    def load_hf_datasets(self,prefix):

        dataset_dict = {}
                
        dataset_dict["train"] = load_from_disk("assets/"+"train_"+prefix+".hf")
        dataset_dict["val"] = load_from_disk("assets/"+"val_"+prefix+".hf")
        dataset_dict["test"] = load_from_disk("assets/"+"test_"+prefix+".hf")

        return dataset_dict



class Augmentation:

    def __init__(self, dataset_to_augment, time_scaling_factor=50.0, sigma=0.05):
        self.dataset_to_augment = dataset_to_augment
        self.ds_length = len(self.dataset_to_augment)
        self.time_scaling_factor = np.abs(time_scaling_factor+1e-7)
        self.sigma = np.abs(sigma+1e-7)


    def sample_concat(self, sample):

        rand_idx = np.random.randint(0,self.ds_length)
        rand_sample = self.dataset_to_augment[rand_idx]

        # Time scaling
        time_scaling = 1+(np.random.randint(-1,1))*np.random.random()/self.time_scaling_factor
        rand_sample["audio"]["array"] = librosa.effects.time_stretch(y=rand_sample["audio"]["array"],
                                                                     rate=time_scaling)

        # Gaussian noise
        gaussian_noise = np.random.normal(0, self.sigma, len(rand_sample["audio"]["array"]))
        rand_sample["audio"]["array"] = rand_sample["audio"]["array"] + gaussian_noise

        # Combining new samples
        new_sample = sample
        new_sample["audio"]["path"] = ''
        new_sample["audio"]["array"] = np.concatenate([rand_sample["audio"]["array"], sample["audio"]["array"]])
        new_sample["audio"]["sampling_rate"] = sample["audio"]["sampling_rate"]
        new_sample["transcription"] = rand_sample["transcription"] + " " + sample["transcription"]

        return new_sample

    def augment(self):
        augmented_dataset = self.dataset_to_augment.map(self.sample_concat)
        self.dataset_to_augment = concatenate_datasets([augmented_dataset,self.dataset_to_augment])

        return self.dataset_to_augment

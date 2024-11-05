import os
import logging
from peft import LoraConfig
from src.data_preparation import Preprocessor
from src.model_train import Training
from transformers import Seq2SeqTrainingArguments
import torch

log = logging.getLogger(__name__)

pretrained_model_id = "openai/whisper-small"
language = "it"
task = "trascribe"

torch.cuda.empty_cache()

# Chech if CUDA is available
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.version.cuda)

# Initialize Preprocessor
data_processing = Preprocessor(dataset_name="miosipof/asr_en",
                          text_column_name = "transcription",
                          pretrained_model_id = pretrained_model_id)

# Load dataset from HuggingFace
ds = data_processing.load()


# Extract audio features via Log-Mel spectrogram and Whisper Encoder
ds = ds.map(data_processing.prepare_dataset, remove_columns=list(next(iter(ds.values())).features)).with_format("torch")

# Save processed datasets to disk
data_processing.save_hf_datasets(dataset_dict=ds, prefix="ASR")

# Load processed datasets from disk
ds = data_processing.load_hf_datasets(prefix="ASR")

print(ds)

# Choose the name for the fine-tuned model
finetuned_model_id = "miosipof/asr_temp"

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=finetuned_model_id,  # change to a repo name of your choice
    optim="adafactor",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    weight_decay=2e-5,
    warmup_steps=50,
    num_train_epochs=3,
    eval_strategy="steps",
    fp16=True,
    predict_with_generate=True,
    per_device_eval_batch_size=8,
    generation_max_length=448,
    logging_steps=50,
    # max_steps = 100,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

# Set Low-Rank Adadptation parameters
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.01, 
    bias="none")

# Apply LoRA
model_trainer.apply_lora(lora_config)

# Initialize Training object
model_trainer = Training(
    pretrained_model_id, 
    finetuned_model_id,
    training_args, 
    language, task)

# Start training loop
model_trainer.train(ds)

# Start evaluation loop
result = model_trainer.eval(ds)

print(result)
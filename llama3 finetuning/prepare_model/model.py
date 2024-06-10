import torch

from accelerate import Accelerator
from torch.utils.checkpoint import checkpoint_sequential
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig

from prepare_model import cfg


def load_model(model_id=cfg['path']['base_model']):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        use_flash_attention_2=False, # set to True if you're using A100
        device_map={"":0}
    )
    return model


def apply_gradient_checkpointing(model):
    model.gradient_checkpointing_enable()
    model.use_cache = False


def create_peft_config(model):
    peft_config = LoraConfig(
        task_type=TaskType.CASUAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


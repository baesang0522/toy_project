import torch.profiler

from transformers import TrainingArguments, Trainer

from common.common_functions import load_config
from prepare_model.model import load_model, apply_gradient_checkpointing, create_peft_config
from prepare_dataset.prepare_dataset import (load_data_to_dataset_format, split_train_test, formatting_prompts_func,
                                             tokenize_function)

cfg = load_config('llama3 finetuning/config/config.yaml')


def training_args_setting(config=cfg):
    training_args = TrainingArguments(
        output_dir=config['path']['save_path'],
        evaluation_strategy=config['param']['evaluation_strategy'],
        eval_steps=config['param']['eval_steps'],
        learning_rate=config['param']['learning_rate'],
        num_train_epochs=config['param']['num_train_epochs'],
        weight_decay=config['param']['weight_decay'],
        logging_steps=config['param']['logging_steps'],
        per_device_train_batch_size=config['param']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['param']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['param']['gradient_accumulation_steps'],
        fp16=config['param']['fp16'],
    )
    return training_args


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")


def fine_tuning_process():
    dataset = load_data_to_dataset_format()
    train_dataset, eval_dataset = split_train_test(dataset)

    print("[Prepare Dataset]::: Now formating data to Llama3 prompt...")
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    print("[Prepare Dataset]::: Formatting finished.")

    print("[Prepare Dataset]::: Tokenizing dataset...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    print("[Prepare Dataset]::: Tokenizing finished.")

    print("[Loading Model]::: Load base model from Huggingface and apply Lora...")
    model = load_model()
    model, _ = create_peft_config(model)
    print("[Loading Model]::: Loading base model finished.")

    trainer = Trainer(
        model=model,
        args=training_args_setting(),
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=1
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        print("[Finetuning Base Model]::: Finetuning process start...")
        trainer.train()
        prof.step()
        print("[Finetuning Base Model]::: Model Finetuned.")


if __name__ == '__main__':
    fine_tuning_process()






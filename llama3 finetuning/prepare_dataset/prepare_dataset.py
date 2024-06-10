import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict

from prepare_dataset import cfg


def load_data_to_dataset_format(data_path=cfg['path']['data_path']):
    try:
        data = pd.read_parquet(data_path)
        dataset = Dataset.from_pandas(data)
        return dataset

    except FileNotFoundError:
        print("Check your data type and use the appropriate Pandas data loading functions.")

    except Exception as e:
        print(e)


def split_train_test(dataset):
    split_dataset = dataset.train_test_split(test_size=cfg['params']['test_size'])
    dataset_dict = DatasetDict(
        {
            "train": split_dataset["train"],
            "test": split_dataset["test"]
        }
    )
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    return train_dataset, eval_dataset


def formatting_prompts_func(examples):
    llama3_prompt = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>\n{}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n{}<|eot_id|><|end_of_text|>
    """

    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = llama3_prompt.format(instruction, input, output)
        texts.append(text)
    return {"text": texts}


def tokenize_function(examples, tokenizer, max_seq_length=cfg['params']['max_seq_length']):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
    return tokenized_inputs




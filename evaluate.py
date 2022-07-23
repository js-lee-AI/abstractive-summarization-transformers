from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mda.utils import ConfigModule, get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from setproctitle import setproctitle
import torch
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tokenize_function(example):
    tokenized_output = tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=512
    )
    return tokenized_output


parser = argparse.ArgumentParser(description="name of the configure file")
parser.add_argument(
    "--cfg_name", '-c',
    type=str,
    default="xsum_finetuning_50k_noaug.yaml",
    help="Name of the configure file",
)
cfg = parser.parse_args()

config_module = ConfigModule()
args = config_module.get_args(cfg_name=cfg.cfg_name)
setproctitle(args.setproc)
checkpoint_dir = args.finetune_xsum.train_params.output_checkpoint_dir
args = args.finetune_xsum.test_params
logger = get_logger(args.eval_result_log_path)
logger.info(f"Your checkpoint dir: {checkpoint_dir}/last_checkpoint")

metric = load_metric("rouge")

xsum_files = {
    "test": args.test_file,
}

raw_datasets = load_dataset("csv", data_files=xsum_files, delimiter="\t")

raw_datasets["test"] = raw_datasets["test"].rename_column("document", "text")
raw_datasets["test"] = raw_datasets["test"].rename_column("summary", "abs")

tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_dir}/last_checkpoint")
# tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint_dir}/12000")
model = AutoModelForSeq2SeqLM.from_pretrained(
    f"{checkpoint_dir}/last_checkpoint"
).cuda()
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     f"{checkpoint_dir}/12000"
# ).cuda()
tokenized_dataset = raw_datasets["test"].map(tokenize_function, batched=True)
abs_dataset = tokenized_dataset["abs"]

if "id" in tokenized_dataset.column_names:
    tokenized_dataset.remove_columns_(
        ["id"]
    )

if "Unnamed: 0" in tokenized_dataset.column_names:
    tokenized_dataset.remove_columns_(
        ["Unnamed: 0"]
    )

if "org_index" in tokenized_dataset.column_names:
    tokenized_dataset.remove_columns_(
        ["org_index"]
    )

tokenized_dataset.remove_columns_(
    ["text", "abs"]
)

tokenized_dataset.set_format("torch")
logger.info(tokenized_dataset)

test_dataloader = DataLoader(
    tokenized_dataset["input_ids"],
    shuffle=False,
    batch_size=args.per_device_test_batch_size,
)

generated_text_list = []
for idx, batch in tqdm(
    enumerate(test_dataloader), desc="evaluate...", total=len(test_dataloader)
):
    model.eval()
    with torch.no_grad():
        generated_tokens = model.generate(batch.cuda())
        generated_text = tokenizer.batch_decode(
            generated_tokens,
            num_beams=args.num_beams,
            max_length=512,
            eos_token_id=tokenizer.eos_token_id,
            skip_special_tokens=True,
        )
        generated_text_list.extend(generated_text)

metric = load_metric("rouge")
print(f"len(generated_text): {len(generated_text_list)}")
print(f"len(abs_dataset): {len(abs_dataset)}")
print(f"len(abs_dataset[: len(generated_text)]): {len(abs_dataset[: len(generated_text_list)])}")
result = metric.compute(
    predictions=generated_text_list, references=abs_dataset[: len(generated_text_list)]
)

print(result)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
result = {k: round(v, 4) for k, v in result.items()}
logger.info(result)

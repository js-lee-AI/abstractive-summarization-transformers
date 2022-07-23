from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import wandb
import math

from torch.utils.data import DataLoader
import torch

from torch.optim import AdamW
from transformers import get_scheduler
from datasets import load_dataset

from tqdm import tqdm
from setproctitle import setproctitle
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_logger(output_log_path=None):
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if output_log_path is not None:
        file_handler = logging.FileHandler(output_log_path)
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    if output_log_path is not None:
        logger.addHandler(file_handler)

    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def tokenize_function(example):
    tokenized_output = tokenizer(example["text"], max_length=512, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["abs"], max_length=512, truncation=True)

    tokenized_output["labels"] = labels["input_ids"]
    return tokenized_output


parser = argparse.ArgumentParser(description="name of the configure file")
parser.add_argument(
    "--cfg_name", '-c',
    type=str,
    default="xsum_finetuning_50k_noaug.yaml",
    help="Name of the configure file",
)
args = parser.parse_args()
set_seed(args.finetune_xsum.seed)

args = args.finetune_xsum.train_params
logger = get_logger(args.train_log_path)
logger.info(f"Configure Arguments - {args}")

xsum_files = {
    "train": args.train_file,
    "valid": args.validation_file,
}

raw_datasets = load_dataset("csv", data_files=xsum_files, delimiter="\t")

raw_datasets["train"] = raw_datasets["train"].rename_column("document", "text")
raw_datasets["train"] = raw_datasets["train"].rename_column("summary", "abs")
raw_datasets["valid"] = raw_datasets["valid"].rename_column("document", "text")
raw_datasets["valid"] = raw_datasets["valid"].rename_column("summary", "abs")

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)

if "id" in tokenized_dataset["train"].column_names:
    tokenized_dataset["train"].remove_columns_(
        ["id"]
    )

if "Unnamed: 0" in tokenized_dataset["train"].column_names:
    tokenized_dataset["train"].remove_columns_(
        ["Unnamed: 0"]
    )

if "org_index" in tokenized_dataset["train"].column_names:
    tokenized_dataset["train"].remove_columns_(
        ["org_index"]
    )

tokenized_dataset["train"].remove_columns_(
    ["text", "abs"]
)
tokenized_dataset["valid"].remove_columns_(
    ["text", "abs"]
)

tokenized_dataset["train"].set_format("torch")
tokenized_dataset["valid"].set_format("torch")

logger.info(tokenized_dataset["train"])
logger.info(tokenized_dataset["valid"])

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).cuda()
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=args.per_device_train_batch_size,
    collate_fn=data_collator,
)

dev_dataloader = DataLoader(
    tokenized_dataset["valid"],
    shuffle=True,
    batch_size=args.per_device_eval_batch_size,
    collate_fn=data_collator,
)

wandb.init(
    name=f"{args.wandb_name}",
    project="xsum",
)

model = torch.nn.DataParallel(model, output_device=1)
CURRENT_STEPS = 0

set_seed(seed=42)

optimizer = AdamW(model.parameters(), lr=args.learning_rate)
num_train_epochs = args.num_train_epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=num_training_steps,
)
progress_bar = tqdm(range(num_training_steps))

wandb_generation_table = []
for epoch in range(num_train_epochs):
    if args.max_train_steps is not None:
        if CURRENT_STEPS % args.max_train_steps == 0:
            break

    for batch in train_dataloader:
        if args.max_train_steps is not None:
            if CURRENT_STEPS % args.max_train_steps == 0:
                break

        model.train()
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss.mean()
        loss.backward()
        ppl = math.exp(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        wandb.log(
            data={
                f"train/loss": loss.item(),
                f"train/ppl": ppl,
                "lr": get_lr(optimizer),
            },
            step=CURRENT_STEPS,
        )

        if CURRENT_STEPS % args.per_eval_steps == 0:
            logger.info("START VALIDATION..")
            model.eval()
            with torch.no_grad():
                for eval_batch in tqdm(dev_dataloader):
                    eval_batch = {k: v.cuda() for k, v in eval_batch.items()}
                    eval_loss = model(
                        input_ids=eval_batch["input_ids"],
                        attention_mask=eval_batch["attention_mask"],
                        labels=eval_batch["labels"],
                    ).loss.mean()
                    eval_ppl = math.exp(eval_loss.item())
                    wandb.log(
                        data={
                            f"valid/loss": eval_loss.item(),
                            f"valid/ppl": eval_ppl,
                        },
                        step=CURRENT_STEPS,
                    )
            logger.info(
                f"step {CURRENT_STEPS} | valid_loss {eval_loss.item()} | valid_ppl {eval_ppl}"
            )

        CURRENT_STEPS += 1
        progress_bar.update(1)
        progress_bar.set_description(
            "step %d | loss %.04f  ppl %.02f  lr %10.2e "
            % (CURRENT_STEPS, loss, ppl, lr_scheduler.get_last_lr()[0])
        )

        if CURRENT_STEPS % args.per_save_checkpoint_steps == 0:
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(
                f"./{args.output_checkpoint_dir}/{CURRENT_STEPS}"
            )
            tokenizer.save_pretrained(f"./{args.output_checkpoint_dir}/{CURRENT_STEPS}")


model_to_save = (
    model.module if hasattr(model, "module") else model
)  # Take care of distributed/parallel training
model_to_save.save_pretrained(f"./{args.output_checkpoint_dir}/last_checkpoint")
tokenizer.save_pretrained(f"./{args.output_checkpoint_dir}/last_checkpoint")

import torch
import argparse, os, json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Expect BIO tagging with columns: tokens (space-joined), tags (space-joined)
def load_bio_csv(path):
    df = pd.read_csv(path)
    tokens = [x.split() for x in df['tokens']]
    tags = [x.split() for x in df['tags']]
    return tokens, tags

def encode_examples(batch, tokenizer, tag2id):
    tokenized = tokenizer(
        batch["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True
    )

    all_labels = []
    for i in range(len(batch["tokens"])):  # loop over sentences
        word_ids = tokenized.word_ids(batch_index=i)
        labels = []
        tags = batch["tags"][i]

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # ignored token
            elif word_id < len(tags):  # valid mapping
                labels.append(tag2id[tags[word_id]])
            else:
                labels.append(-100)  # safety check
        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', default='distilbert-base-cased')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=False)
    ap.add_argument('--out_dir', default='checkpoints/ner')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_tokens, train_tags = load_bio_csv(args.train_csv)
    if args.val_csv:
        val_tokens, val_tags = load_bio_csv(args.val_csv)
    else:
        split_idx = int(len(train_tokens)*0.9)
        val_tokens, val_tags = train_tokens[split_idx:], train_tags[split_idx:]
        train_tokens, train_tags = train_tokens[:split_idx], train_tags[:split_idx]

    # Build tag map from data
    uniq = sorted({t for seq in train_tags+val_tags for t in seq})
    tag2id = {t:i for i,t in enumerate(uniq)}
    id2tag = {i:t for t,i in tag2id.items()}
    with open(os.path.join(args.out_dir, 'label_map.json'), 'w') as f:
        json.dump({'tag2id': tag2id, 'id2tag': id2tag}, f)

    ds_train = Dataset.from_dict({'tokens': train_tokens, 'tags': train_tags})
    ds_val = Dataset.from_dict({'tokens': val_tokens, 'tags': val_tags})

    def _encode(batch):
        return encode_examples(batch, tokenizer, tag2id)

    ds_train = ds_train.map(_encode, batched=True)
    ds_val = ds_val.map(_encode, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(tag2id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=5e-5,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    model.config.id2label = {0: "O", 1: "B-ANIMAL"}
    model.config.label2id = {"O": 0, "B-ANIMAL": 1}
    model.save_pretrained(args.out_dir)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

if __name__ == '__main__':
    main()

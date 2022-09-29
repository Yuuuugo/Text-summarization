import torch
from datasets import Dataset
import pandas as pd
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        max_text_len: int = 1024,
        max_summary_len: int = 256,
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        row = self.data.iloc[index]

        text = row["article"]
        encoded_text = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.max_text_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        summary = row["summary"]
        encoded_summary = self.tokenizer.batch_encode_plus(
            summary,
            max_length=self.max_text_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        labels = encoded_summary["input_ids"]
        labels[labels == 0] = -100

        return {
            "text": text,
            "summary": summary,
            "text_input_ids": encoded_text["input_ids"],
            "text_attention_mask": encoded_text["attention_mask"],
            "summary_input_ids": labels,
            "summary_attention_mask": encoded_summary["attention_mask"],
        }


class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        max_text_len: int = 1024,
        max_summary_len: int = 256,
    ):
        super().__init__()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    def setup(self):
        self.train_dataset = SummaryDataset(
            data=self.train_df,
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            max_summary_len=self.max_summary_len,
        )

        self.val_dataset = SummaryDataset(
            data=self.val_df,
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            max_summary_len=self.max_summary_len,
        )
        self.test_dataset = SummaryDataset(
            data=self.test_df,
            tokenizer=self.tokenizer,
            max_text_len=self.max_text_len,
            max_summary_len=self.max_summary_len,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataloader,
            batch_size=self.batch_size,
            shuffle=False,
            num_worker=2,
        )


if __name__ == "__main__":
    train_df = pd.read_csv("cnn_dailymail/train.csv")
    test_df = pd.read_csv("cnn_dailymail/test.csv")
    val_df = pd.read_csv("cnn_dailymail/validation.csv")

    dataset = SummaryDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=T5Tokenizer,
    )

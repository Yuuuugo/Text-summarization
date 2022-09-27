import gradio as gr
import torch
from transformers import T5Tokenizer


def summarize(text):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = torch.load("model.pt")
    predictions = []
    tokenized_sentence = tokenizer.batch_encode_plus(
        [text], max_length=512, pad_to_max_length=True, return_tensors="pt"
    )
    generated_ids = model.generate(
        input_ids=tokenized_sentence["input_ids"],
        attention_mask=tokenized_sentence["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )
    preds = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in generated_ids
    ]
    predictions.extend(preds)
    return predictions


gr.Interface(fn=summarize, inputs=["text"], outputs=["text"]).launch()

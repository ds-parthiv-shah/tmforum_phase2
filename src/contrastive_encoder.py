# import torch
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
# model = AutoModel.from_pretrained('facebook/contriever')

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
red = classifier("i have been waiting for huggingface cource for my whole life")
print(red)
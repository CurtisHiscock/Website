from transformers import AutoTokenizer, AutoModel
import os

model_name = "sentence-transformers/all-MiniLM-L6-v2"
save_path = "./all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
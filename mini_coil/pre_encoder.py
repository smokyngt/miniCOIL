from typing import List

import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os
import onnxruntime as ort

from mini_coil.settings import DATA_DIR


def download_and_save_onnx(model_repository, model_save_path):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_repository)
    tokenizer = AutoTokenizer.from_pretrained(model_repository)

    texts = [
        "Hello, this is a test.",
        "This is another test, a bit longer than the first one.",
    ]

    # Prepare dummy input for model export
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Prepare the model for exporting
    model.eval()

    # Export the model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']),
            f=model_save_path,
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=[
                'last_hidden_state',
                'embedding'
            ],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
                'embedding': {0: 'batch_size'}
            },
            opset_version=14
        )


class PreEncoder():

    def __init__(self, model_repository: str, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_repository)
        self.session = ort.InferenceSession(model_path)

    def encode(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True)
        # {
        #     'input_ids': array(
        #         [
        #             [101, 7592, 1010, 2023, 2003, 1037, 3231, 1012, 102],
        #             [101, 2023, 2003, 2178, 3231, 1012, 102, 0, 0]
        #         ]
        #     ),
        #     'token_type_ids': array(
        #         [
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0]
        #         ]
        #     ),
        #     'attention_mask': array(
        #         [
        #             [1, 1, 1, 1, 1, 1, 1, 1, 1],
        #             [1, 1, 1, 1, 1, 1, 1, 0, 0]
        #         ]
        #     )
        # }

        outputs = self.session.run(None, {**inputs})

        # (batch_size, sequence_length)
        input_ids = inputs['input_ids']

        # (batch_size, sequence_length, embedding_size)
        token_embeddings = outputs[0]

        # (batch_size, embedding_size)
        text_embeddings = outputs[1]

        # (batch_size)
        number_of_tokens = np.sum(inputs['attention_mask'], axis=1)

        return {
            'number_of_tokens': number_of_tokens,
            'token_ids': input_ids,
            'token_embeddings': token_embeddings,
            'text_embeddings': text_embeddings
        }


if __name__ == "__main__":
    # Specify the Hugging Face repository and the local path for saving the ONNX model
    model_repository = "sentence-transformers/all-MiniLM-L6-v2"
    model_save_path = os.path.join(DATA_DIR, "all_miniLM_L6_v2.onnx")

    download_and_save_onnx(model_repository, model_save_path)

    pre_encoder = PreEncoder(model_repository, model_save_path)

    texts = [
        "Hello, this is a test.",
        "This is another test, a bit longer than the first one.",
    ]

    result = pre_encoder.encode(texts)

    print(result["text_embeddings"][0][:10])

    print(result["number_of_tokens"])

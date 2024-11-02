import os
import time
from typing import List

import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import onnxruntime as ort

from mini_coil.settings import DATA_DIR


def cosine_similarity(rows_a: np.ndarray, rows_b: np.ndarray):
    """
    Compute a matrix of cosine distances between two sets of vectors.
    """
    # Normalize the vectors
    rows_a = rows_a / np.linalg.norm(rows_a, axis=1, keepdims=True)
    rows_b = rows_b / np.linalg.norm(rows_b, axis=1, keepdims=True)

    # Compute the cosine similarity
    return np.dot(rows_a, rows_b.T)


def download_and_save_onnx(model_repository, model_save_path):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_repository, trust_remote_code=True)
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


class PreEncoder:

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


def check_similarity():
    from sentence_transformers import SentenceTransformer

    # model_repository = "Alibaba-NLP/gte-large-en-v1.5"
    model_repository = "mixedbread-ai/mxbai-embed-large-v1"

    model = SentenceTransformer(model_repository, trust_remote_code=True, device="cpu")

    text_a = "The bat flew out of the cave."
    text_b = "He is a baseball player. He knows how to swing a bat."
    text_c = "A bat can use echolocation to navigate in the dark."
    text_d = "It was just a cricket bat."
    text_e = "And guess who the orphans have at bat!"
    text_f = "Eric Byrnes, never with an at bat in Yankee Stadium and they don't get much bigger than this one."

    texts = [text_a, text_b, text_c, text_d, text_e, text_f]

    time_start = time.time()

    embeddings = model.encode(texts)

    print("Time taken to encode:", time.time() - time_start)

    original_matrix = cosine_similarity(embeddings, embeddings)

    print("original similarity matrix\n", original_matrix)

    texts = [
        "java developer intern",
        "coffee from java island",
        "java programming language",
        "java is located in indonesia",
    ]

    embeddings = model.encode(texts)

    original_matrix = cosine_similarity(embeddings, embeddings)

    print("original similarity matrix\n", original_matrix)


if __name__ == "__main__":
    # check_similarity()
    #
    # exit(0)

    # Specify the Hugging Face repository and the local path for saving the ONNX model
    model_repository = "jinaai/jina-embeddings-v2-base-en"
    model_save_path = os.path.join(DATA_DIR, "jina-embeddings-v2-base-en.onnx")

    download_and_save_onnx(model_repository, model_save_path)

    pre_encoder = PreEncoder(model_repository, model_save_path)

    text_a = "The bat flew out of the cave."
    text_b = "He is a baseball player. He knows how to swing a bat."
    text_c = "A bat can use echolocation to navigate in the dark."
    text_d = "It was just a cricket bat."
    text_e = "And guess who the orphans have at bat!"
    text_f = "Eric Byrnes, never with an at bat in Yankee Stadium and they don't get much bigger than this one."

    texts = [text_a, text_b, text_c, text_d, text_e, text_f]

    text_embeddings = pre_encoder.encode(texts)["text_embeddings"]

    original_matrix = cosine_similarity(text_embeddings, text_embeddings)

    print("original similarity matrix\n", original_matrix)

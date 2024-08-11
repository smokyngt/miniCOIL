import os

import lightning as L
from ipdb import launch_ipdb_on_exception

from mini_coil.data_pipeline.vocab_resolver import VocabResolver
from mini_coil.model.decoder import Decoder
from mini_coil.model.encoder import Encoder
from mini_coil.settings import DATA_DIR
from mini_coil.training.coil_module import MiniCoil
from mini_coil.training.data_loader import PreEncodedLoader


def get_encoder(vocab_size):
    return Encoder(
        input_dim=384,
        intermediate_dim=128,
        output_dim=4,
        vocab_size=vocab_size,
    )


def get_decoder(vocab_size):
    return Decoder(
        input_dim=4,
        output_dim=384,
        vocab_size=vocab_size,
    )


def get_model(vocab_size):
    return MiniCoil(
        encoder=get_encoder(vocab_size),
        decoder=get_decoder(vocab_size),
    )


def main():
    batch_size = 64
    model_repository = "sentence-transformers/all-MiniLM-L6-v2"

    path = os.path.join(DATA_DIR, "test")
    train_loader = PreEncodedLoader(path, batch_size)

    test_vocab_path = os.path.join(DATA_DIR, "test", "vocab.txt")

    vocab_resolver = VocabResolver(model_repository)
    vocab_resolver.load_vocab(test_vocab_path)

    mini_coil = get_model(vocab_resolver.vocab_size())

    trainer = L.Trainer(max_epochs=1000)

    # catch with ipdb
    with launch_ipdb_on_exception():
        trainer.fit(
            model=mini_coil,
            train_dataloaders=train_loader
        )


if __name__ == "__main__":
    main()

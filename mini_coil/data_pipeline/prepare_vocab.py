import argparse
from typing import List
import tqdm

from nltk import WordNetLemmatizer


def read_source_vocab(path) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    import nltk

    nltk.download('wordnet')

    source_vocab = read_source_vocab(args.input_file)

    # Remove stopwords and words with less than 3 characters
    # As vocab is sorted by frequency, define stopwords as first 100 words if the word is less than 5

    target_vocab = [word for idx, word in enumerate(source_vocab) if len(word) > (5 if idx < 100 else 2)]

    lemmatizer = WordNetLemmatizer()

    seen_normalized_words = set()

    with open(args.output_file, "w") as f:
        for word in tqdm.tqdm(target_vocab):
            normalized_word = lemmatizer.lemmatize(word)
            if normalized_word in seen_normalized_words:
                continue
            seen_normalized_words.add(normalized_word)
            f.write(f"{normalized_word}\n")


if __name__ == "__main__":
    main()

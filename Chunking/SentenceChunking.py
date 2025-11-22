# sentence_chunking.py
import nltk
from nltk.tokenize import sent_tokenize


# Download required NLTK data (run once)
# nltk.download('punkt')

def sentence_based_chunking(text, sentences_per_chunk=5, overlap_sentences=1):
    """
    Split text into chunks based on sentence boundaries
    """
    sentences = sent_tokenize(text)
    chunks = []

    i = 0
    while i < len(sentences):
        chunk_end = min(i + sentences_per_chunk, len(sentences))
        chunk = ' '.join(sentences[i:chunk_end])
        chunks.append(chunk)

        i += sentences_per_chunk - overlap_sentences

    return chunks


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    filename = "Best Russian Short Stories.txt"

    try:
        text = read_file(filename)
        chunks = sentence_based_chunking(text, sentences_per_chunk=3, overlap_sentences=1)

        print(f"Total sentences: {len(sent_tokenize(text))}")
        print(f"Total chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk)

    except FileNotFoundError:
        print(f"File {filename} not found.")


if __name__ == "__main__":
    main()
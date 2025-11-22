# fixed_size_chunking.py
import re


def fixed_size_chunking(text, chunk_size=1000, overlap=0):
    """
    Split text into chunks of fixed size with optional overlap
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk.strip()) > 0:  # Only add non-empty chunks
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def read_file(filename):
    """Read text from file"""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    # Example usage
    filename = "Best Russian Short Stories.txt"

    try:
        text = read_file(filename)
        chunks = fixed_size_chunking(text, chunk_size=500, overlap=50)

        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    except FileNotFoundError:
        print(f"File {filename} not found. Please create a sample text file.")


if __name__ == "__main__":
    main()
# paragraph_chunking.py

def paragraph_based_chunking(text, paragraphs_per_chunk=3, overlap_paragraphs=1):
    """
    Split text into chunks based on paragraph boundaries
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []

    i = 0
    while i < len(paragraphs):
        chunk_end = min(i + paragraphs_per_chunk, len(paragraphs))
        chunk = '\n\n'.join(paragraphs[i:chunk_end])
        chunks.append(chunk)

        i += paragraphs_per_chunk - overlap_paragraphs

    return chunks


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    filename = "Best Russian Short Stories.txt"

    try:
        text = read_file(filename)
        chunks = paragraph_based_chunking(text, paragraphs_per_chunk=2, overlap_paragraphs=1)

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        print(f"Total paragraphs: {len(paragraphs)}")
        print(f"Total chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    except FileNotFoundError:
        print(f"File {filename} not found.")


if __name__ == "__main__":
    main()
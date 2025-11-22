# semantic_chunking.py
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


def recursive_character_chunking(text, chunk_size=1000, chunk_overlap=200):
    """
    Use recursive character splitting to keep related text together
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    return chunks


def character_chunking(text, chunk_size=1000, chunk_overlap=200):
    """
    Simple character-based chunking with LangChain
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separator="\n"
    )

    chunks = text_splitter.split_text(text)
    return chunks


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    filename = "Best Russian Short Stories.txt"

    try:
        text = read_file(filename)

        print("=== Recursive Character Chunking ===")
        recursive_chunks = recursive_character_chunking(text, 500, 100)
        print(f"Chunks: {len(recursive_chunks)}")

        print("\n=== Character Chunking ===")
        char_chunks = character_chunking(text, 500, 100)
        print(f"Chunks: {len(char_chunks)}")

        for i, chunk in enumerate(recursive_chunks[:3], 1):
            print(f"\n--- Recursive Chunk {i} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    except FileNotFoundError:
        print(f"File {filename} not found.")
    except ImportError:
        print("Please install langchain: pip install langchain")


if __name__ == "__main__":
    main()
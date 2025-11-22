# content_aware_chunking.py
import re


def heading_based_chunking(text):
    """
    Split text based on headings (Markdown-style or uppercase headings)
    """
    # Pattern for markdown headings (#, ##, ###) and uppercase headings
    heading_pattern = r'(?:\n|^)(#{1,3}\s+.+?|\n[A-Z][A-Z\s]{10,}:\n|\n[A-Z][A-Z\s]+\n)'

    chunks = []
    sections = re.split(heading_pattern, text)
    headings = re.findall(heading_pattern, text)

    # Combine headings with their content
    for i, section in enumerate(sections):
        if section.strip():
            if i == 0 and not section.startswith('#'):
                # First section without heading
                chunks.append(("Introduction", section))
            elif i > 0 and i - 1 < len(headings):
                chunks.append((headings[i - 1].strip(), section))

    return chunks


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def main():
    filename = "Best Russian Short Stories.txt"

    try:
        text = read_file(filename)
        chunks = heading_based_chunking(text)

        print(f"Total sections: {len(chunks)}")

        for i, (heading, content) in enumerate(chunks, 1):
            print(f"\n--- Section {i}: {heading[:50]} ---")
            print(f"Content length: {len(content)} chars")
            print(content[:200] + "..." if len(content) > 200 else content)

    except FileNotFoundError:
        print(f"File {filename} not found.")


if __name__ == "__main__":
    main()
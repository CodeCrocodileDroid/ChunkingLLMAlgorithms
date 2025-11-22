import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class TextSearcher:
    def __init__(self, file_path: str):
        """
        Initialize the text searcher with a file path.

        Args:
            file_path (str): Path to the text file to search
        """
        self.file_path = file_path
        self.sentences = self._load_and_split_text()
        self.model = None
        self.embeddings = None

    def _load_and_split_text(self) -> List[str]:
        """
        Load the text file and split it into sentences.

        Returns:
            List[str]: List of sentences from the text file
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Split text into sentences (simple approach)
            sentences = re.split(r'[.!?]+', text)
            # Remove empty strings and strip whitespace
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences

        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return []
        except Exception as e:
            print(f"Error loading file: {e}")
            return []

    def initialize_semantic_search(self):
        """
        Initialize the semantic search model and compute embeddings.
        This needs to be called before using semantic search.
        """
        try:
            print("Loading semantic search model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Computing embeddings...")
            self.embeddings = self.model.encode(self.sentences)
            print("Semantic search ready!")
        except Exception as e:
            print(f"Error initializing semantic search: {e}")

    def regular_search(self, query: str, case_sensitive: bool = False) -> List[Tuple[str, int]]:
        """
        Perform regular text search (keyword matching).

        Args:
            query (str): Search query
            case_sensitive (bool): Whether search should be case sensitive

        Returns:
            List[Tuple[str, int]]: List of (sentence, position) tuples
        """
        results = []

        if not case_sensitive:
            query = query.lower()

        for i, sentence in enumerate(self.sentences):
            search_text = sentence if case_sensitive else sentence.lower()
            if query in search_text:
                results.append((sentence, i))

        return results

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, int, float]]:
        """
        Perform semantic search using sentence embeddings.

        Args:
            query (str): Search query
            top_k (int): Number of top results to return

        Returns:
            List[Tuple[str, int, float]]: List of (sentence, position, similarity_score) tuples
        """
        if self.model is None or self.embeddings is None:
            print("Semantic search not initialized. Call initialize_semantic_search() first.")
            return []

        # Encode the query
        query_embedding = self.model.encode([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                results.append((self.sentences[idx], idx, similarities[idx]))

        return results

    def display_results(self, results: List, search_type: str):
        """
        Display search results in a formatted way.

        Args:
            results (List): List of search results
            search_type (str): Type of search ('regular' or 'semantic')
        """
        if not results:
            print(f"No results found for {search_type} search.")
            return

        print(f"\n{'=' * 60}")
        print(f"{search_type.upper()} SEARCH RESULTS")
        print(f"{'=' * 60}")

        for i, result in enumerate(results, 1):
            if search_type == 'regular':
                sentence, position = result
                print(f"{i}. Position {position}: {sentence}")
            else:  # semantic
                sentence, position, score = result
                print(f"{i}. [Score: {score:.3f}] Position {position}: {sentence}")
            print()


def main():
    # Configuration
    FILE_PATH = "Best Russian Short Stories.txt"  # Change this to your file path

    # Create a sample file if it doesn't exist
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            pass
    except FileNotFoundError:
        print("Creating sample text file...")
        sample_text = """
        Artificial intelligence is transforming the technology landscape.
        Machine learning algorithms can recognize patterns in large datasets.
        Natural language processing helps computers understand human language.
        Deep learning uses neural networks with multiple layers.
        Computer vision enables machines to interpret visual information.
        Robotics combines AI with mechanical engineering.
        Data science involves extracting insights from complex data.
        Cloud computing provides scalable resources for AI applications.
        The Internet of Things connects physical devices to the digital world.
        Blockchain technology offers secure and transparent transactions.
        """
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"Sample file created at: {FILE_PATH}")

    # Initialize searcher
    searcher = TextSearcher(FILE_PATH)

    if not searcher.sentences:
        print("No text found to search.")
        return

    print(f"Loaded {len(searcher.sentences)} sentences from the file.")

    # Initialize semantic search
    searcher.initialize_semantic_search()

    while True:
        print("\n" + "=" * 50)
        print("SEARCH OPTIONS")
        print("=" * 50)
        print("1. Regular Search (Keyword Matching)")
        print("2. Semantic Search (Meaning-based)")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '3':
            print("Goodbye!")
            break

        if choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue

        query = input("Enter your search query: ").strip()

        if not query:
            print("Query cannot be empty.")
            continue

        if choice == '1':
            # Regular search
            case_sensitive = input("Case sensitive? (y/n): ").strip().lower() == 'y'
            results = searcher.regular_search(query, case_sensitive)
            searcher.display_results(results, "regular")

        elif choice == '2':
            # Semantic search
            try:
                top_k = int(input("Number of results to show (default 5): ") or "5")
            except ValueError:
                top_k = 5

            results = searcher.semantic_search(query, top_k)
            searcher.display_results(results, "semantic")


if __name__ == "__main__":
    main()
import re
import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict


class VectorTextSearcher:
    def __init__(self, file_path: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the text searcher with ChromaDB vector database.

        Args:
            file_path (str): Path to the text file to search
            persist_directory (str): Directory to store the vector database
        """
        self.file_path = file_path
        self.persist_directory = persist_directory
        self.sentences = []
        self.model = None
        self.client = None
        self.collection = None
        self.is_initialized = False

    def _load_and_split_text(self) -> List[str]:
        """
        Load the text file and split it into sentences.

        Returns:
            List[str]: List of sentences from the text file
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Split text into sentences (improved approach)
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

    def _needs_reindexing(self) -> bool:
        """
        Check if we need to reindex the database (file modified or first run).

        Returns:
            bool: True if reindexing is needed
        """
        # Check if database directory exists
        if not os.path.exists(self.persist_directory):
            return True

        # Check if collection exists and has documents
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection("text_embeddings")
            if collection.count() == 0:
                return True
        except:
            return True

        return False

    def initialize(self):
        """
        Initialize the system - load text and setup vector database.
        Only computes embeddings on first run or if file changes.
        """
        print("Initializing text search system...")

        # Load and split text
        self.sentences = self._load_and_split_text()
        if not self.sentences:
            print("No text found to process.")
            return False

        print(f"Loaded {len(self.sentences)} sentences from: {self.file_path}")

        # Initialize model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="text_embeddings",
                metadata={"description": "Text embeddings for semantic search"}
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return False

        # Check if we need to compute embeddings
        if self._needs_reindexing():
            print("Computing and storing embeddings (this may take a moment)...")
            success = self._store_embeddings()
            if success:
                print("✓ Embeddings computed and stored successfully!")
            else:
                print("✗ Failed to store embeddings")
                return False
        else:
            print("✓ Using existing embeddings from vector database")

        self.is_initialized = True
        print("System ready for searches!")
        return True

    def _store_embeddings(self) -> bool:
        """
        Compute embeddings and store them in ChromaDB.

        Returns:
            bool: True if successful
        """
        try:
            # Clear existing collection
            try:
                self.client.delete_collection("text_embeddings")
            except:
                pass

            self.collection = self.client.get_or_create_collection(
                name="text_embeddings",
                metadata={"description": "Text embeddings for semantic search"}
            )

            # Compute embeddings in batches to handle large files
            batch_size = 32
            all_embeddings = []

            for i in range(0, len(self.sentences), batch_size):
                batch = self.sentences[i:i + batch_size]
                batch_embeddings = self.model.encode(batch).tolist()
                all_embeddings.extend(batch_embeddings)

                # Show progress
                progress = min(i + batch_size, len(self.sentences))
                print(f"  Processing: {progress}/{len(self.sentences)} sentences")

            # Prepare and store all documents
            documents = []
            metadatas = []
            ids = []

            for i, (sentence, embedding) in enumerate(zip(self.sentences, all_embeddings)):
                documents.append(sentence)
                metadatas.append({
                    "position": i,
                    "length": len(sentence),
                    "source_file": self.file_path
                })
                ids.append(f"doc_{i}")

            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=all_embeddings
            )

            return True

        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False

    def regular_search(self, query: str, case_sensitive: bool = False) -> List[Tuple[str, int]]:
        """
        Perform regular text search (keyword matching).

        Args:
            query (str): Search query
            case_sensitive (bool): Whether search should be case sensitive

        Returns:
            List[Tuple[str, int]]: List of (sentence, position) tuples
        """
        if not self.is_initialized:
            print("System not initialized. Call initialize() first.")
            return []

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
        Perform semantic search using ChromaDB vector database.

        Args:
            query (str): Search query
            top_k (int): Number of top results to return

        Returns:
            List[Tuple[str, int, float]]: List of (sentence, position, similarity_score) tuples
        """
        if not self.is_initialized:
            print("System not initialized. Call initialize() first.")
            return []

        if self.collection is None:
            print("Vector database not available.")
            return []

        try:
            # Query the vector database
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, len(self.sentences)),
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                ):
                    # Convert distance to similarity score (cosine similarity)
                    similarity = 1 - distance
                    position = metadata['position']
                    search_results.append((doc, position, similarity))

            return search_results

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def get_database_info(self) -> Dict:
        """
        Get information about the vector database.

        Returns:
            Dict: Database information
        """
        if not self.is_initialized or self.collection is None:
            return {"status": "not_initialized"}

        try:
            return {
                "status": "ready",
                "document_count": self.collection.count(),
                "collection_name": "text_embeddings",
                "source_file": self.file_path,
                "sentences_loaded": len(self.sentences)
            }
        except:
            return {"status": "error"}

    def display_results(self, results: List, search_type: str, query: str = ""):
        """
        Display search results in a formatted way.

        Args:
            results (List): List of search results
            search_type (str): Type of search ('regular' or 'semantic')
            query (str): The original search query
        """
        if not results:
            print(f"\n❌ No results found for: '{query}'")
            return

        print(f"\n{'=' * 70}")
        print(f"🔍 {search_type.upper()} SEARCH RESULTS: '{query}'")
        print(f"{'=' * 70}")

        for i, result in enumerate(results, 1):
            if search_type == 'regular':
                sentence, position = result
                print(f"{i}. 📍 Position {position}")
                print(f"   {sentence}")
            else:  # semantic
                sentence, position, score = result
                print(f"{i}. 📍 Position {position} | 🔥 Score: {score:.3f}")
                print(f"   {sentence}")
            print()


def create_sample_file(file_path: str):
    """
    Create a sample text file if it doesn't exist.
    """
    sample_text = """
Artificial Intelligence and Modern Technology

Artificial intelligence is transforming the technology landscape across various industries. 
Machine learning algorithms can recognize complex patterns in large datasets efficiently. 
Natural language processing helps computers understand and interpret human language accurately. 
Deep learning uses neural networks with multiple layers to solve complex problems. 
Computer vision enables machines to interpret and analyze visual information from the world. 
Robotics combines AI with mechanical engineering to create autonomous systems. 
Data science involves extracting valuable insights from complex and structured data. 
Cloud computing provides scalable resources and infrastructure for AI applications. 
The Internet of Things connects physical devices to the digital world seamlessly. 
Blockchain technology offers secure, transparent and decentralized transactions. 
Quantum computing promises to solve complex computational problems much faster. 
Cybersecurity protects computer systems and data from digital attacks and breaches. 
Virtual reality creates immersive digital environments for entertainment and training. 
Augmented reality overlays digital information onto the real world environment. 
Autonomous vehicles use advanced sensors and AI to navigate without human input. 
Edge computing processes data closer to the source reducing latency significantly. 
5G technology enables faster wireless communication and connectivity worldwide. 
Biotechnology uses living systems to develop products and technologies for healthcare. 
Renewable energy technologies provide sustainable power solutions for the future. 
Smart cities use technology to improve urban infrastructure and services efficiently.
    """

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"✓ Created sample file: {file_path}")
        return True
    except Exception as e:
        print(f"✗ Error creating sample file: {e}")
        return False


def main():
    """
    Main function to run the text search application.
    """
    # Configuration
    FILE_PATH = "Best Russian Short Stories.txt"  # Change this to your text file path
    PERSIST_DIR = "./chroma_db"  # Directory for vector database

    print("🚀 Text Search System with Vector Database")
    print("=" * 50)

    # Check if text file exists, create sample if not
    if not os.path.exists(FILE_PATH):
        print(f"Text file not found: {FILE_PATH}")
        create_sample = input("Create a sample text file? (y/n): ").strip().lower()
        if create_sample == 'y':
            if not create_sample_file(FILE_PATH):
                return
        else:
            print("Please provide a valid text file path.")
            return

    # Initialize the search system
    searcher = VectorTextSearcher(FILE_PATH, PERSIST_DIR)

    if not searcher.initialize():
        print("Failed to initialize the search system.")
        return

    # Show system information
    info = searcher.get_database_info()
    print(f"\n📊 System Information:")
    print(f"   • Documents in database: {info.get('document_count', 0)}")
    print(f"   • Sentences loaded: {info.get('sentences_loaded', 0)}")
    print(f"   • Source file: {info.get('source_file', 'N/A')}")

    # Main interaction loop
    while True:
        print("\n" + "=" * 50)
        print("🔍 SEARCH OPTIONS")
        print("=" * 50)
        print("1. 🔤 Regular Search (Keyword Matching)")
        print("2. 🧠 Semantic Search (AI-Powered)")
        print("3. 📊 System Status")
        print("4. 🚪 Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '4':
            print("\n👋 Thank you for using the Text Search System!")
            break

        if choice == '3':
            info = searcher.get_database_info()
            print(f"\n📊 System Status:")
            for key, value in info.items():
                print(f"   • {key.replace('_', ' ').title()}: {value}")
            continue

        if choice not in ['1', '2']:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            continue

        query = input("\nEnter your search query: ").strip()

        if not query:
            print("❌ Query cannot be empty.")
            continue

        if choice == '1':
            # Regular search
            case_sensitive = input("Case sensitive search? (y/n): ").strip().lower() == 'y'
            results = searcher.regular_search(query, case_sensitive)
            searcher.display_results(results, "regular", query)

        elif choice == '2':
            # Semantic search
            try:
                top_k = int(input("Number of results to show (default 5): ") or "5")
                top_k = max(1, min(top_k, 20))  # Limit between 1 and 20
            except ValueError:
                top_k = 5
                print("Using default: 5 results")

            print(f"\n🔍 Performing semantic search for: '{query}'...")
            results = searcher.semantic_search(query, top_k)
            searcher.display_results(results, "semantic", query)


if __name__ == "__main__":
    main()
import re
import os
import wx
import wx.lib.mixins.listctrl as listmix
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import threading
import time


class VectorTextSearcher:
    def __init__(self, file_path: str = "", persist_directory: str = "./chroma_db"):
        """
        Initialize the text searcher with ChromaDB vector database.
        """
        self.file_path = file_path
        self.persist_directory = persist_directory
        self.sentences = []
        self.model = None
        self.client = None
        self.collection = None
        self.is_initialized = False

    def set_file_path(self, file_path: str):
        """Set the file path and reset initialization"""
        self.file_path = file_path
        self.is_initialized = False
        self.sentences = []

    def _load_and_split_text(self) -> List[str]:
        """
        Load the text file and split it into sentences.
        """
        if not self.file_path or not os.path.exists(self.file_path):
            return []

        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Split text into sentences
            sentences = re.split(r'[.!?]+', text)
            # Remove empty strings and strip whitespace
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences

        except Exception as e:
            print(f"Error loading file: {e}")
            return []

    def _needs_reindexing(self) -> bool:
        """
        Check if we need to reindex the database.
        """
        if not os.path.exists(self.persist_directory):
            return True

        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection("text_embeddings")
            if collection.count() == 0:
                return True
        except:
            return True

        return False

    def initialize(self, progress_callback=None) -> bool:
        """
        Initialize the system with progress callback for GUI.
        """
        if not self.file_path:
            return False

        # Load and split text
        self.sentences = self._load_and_split_text()
        if not self.sentences:
            return False

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
            success = self._store_embeddings_batched(progress_callback)
            if not success:
                return False

        self.is_initialized = True
        return True

    def _store_embeddings_batched(self, progress_callback=None) -> bool:
        """
        Compute embeddings and store them in ChromaDB with progress updates.
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

            # Batch sizes
            embedding_batch_size = 32
            db_batch_size = 100

            all_documents = []
            all_metadatas = []
            all_ids = []
            all_embeddings = []

            total_sentences = len(self.sentences)

            # Process embeddings in batches
            for i in range(0, total_sentences, embedding_batch_size):
                batch_sentences = self.sentences[i:i + embedding_batch_size]

                # Compute embeddings for this batch
                batch_embeddings = self.model.encode(batch_sentences).tolist()
                all_embeddings.extend(batch_embeddings)

                # Prepare documents for this batch
                for j, sentence in enumerate(batch_sentences):
                    absolute_index = i + j
                    all_documents.append(sentence)
                    all_metadatas.append({
                        "position": absolute_index,
                        "length": len(sentence),
                        "source_file": self.file_path
                    })
                    all_ids.append(f"doc_{absolute_index}")

                # Update progress
                if progress_callback:
                    progress = min(i + embedding_batch_size, total_sentences)
                    progress_callback(progress, total_sentences, "Processing sentences")

            # Store in database in smaller batches
            for i in range(0, len(all_documents), db_batch_size):
                end_idx = min(i + db_batch_size, len(all_documents))

                batch_documents = all_documents[i:end_idx]
                batch_metadatas = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                batch_embeddings = all_embeddings[i:end_idx]

                self.collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )

                if progress_callback:
                    progress_callback(end_idx, len(all_documents), "Storing embeddings")

            return True

        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False

    def regular_search(self, query: str, case_sensitive: bool = False) -> List[Tuple[str, int]]:
        """
        Perform regular text search.
        """
        if not self.is_initialized:
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
        Perform semantic search.
        """
        if not self.is_initialized or self.collection is None:
            return []

        try:
            top_k = min(top_k, 50)

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            search_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                ):
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
        """
        if not self.is_initialized or self.collection is None:
            return {"status": "not_initialized"}

        try:
            return {
                "status": "ready",
                "document_count": self.collection.count(),
                "source_file": self.file_path,
                "sentences_loaded": len(self.sentences)
            }
        except:
            return {"status": "error"}


class ResultsListCtrl(wx.ListCtrl, listmix.ListCtrlAutoWidthMixin):
    def __init__(self, parent, id=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.LC_REPORT):
        wx.ListCtrl.__init__(self, parent, id, pos, size, style)
        listmix.ListCtrlAutoWidthMixin.__init__(self)

        self.setup_columns()

    def setup_columns(self):
        """Setup list control columns"""
        self.InsertColumn(0, "Rank", width=50)
        self.InsertColumn(1, "Position", width=70)
        self.InsertColumn(2, "Score", width=70)
        self.InsertColumn(3, "Text", width=500)

    def update_results(self, results, search_type="semantic"):
        """Update the list with search results"""
        self.DeleteAllItems()

        for i, result in enumerate(results, 1):
            if search_type == "regular":
                sentence, position = result
                score = "N/A"
            else:
                sentence, position, score = result
                score = f"{score:.3f}"

            index = self.InsertItem(self.GetItemCount(), str(i))
            self.SetItem(index, 1, str(position))
            self.SetItem(index, 2, score)
            self.SetItem(index, 3, sentence)


class TextSearchFrame(wx.Frame):
    def __init__(self, parent, title="Text Search with Vector Database"):
        super(TextSearchFrame, self).__init__(parent, title=title, size=(1000, 700))

        self.searcher = VectorTextSearcher()
        self.current_file = ""

        self.init_ui()
        self.Centre()
        self.Show()

    def init_ui(self):
        """Initialize the user interface"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(panel, label="Text Search with Vector Database")
        title_font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        # File selection section
        file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.file_label = wx.StaticText(panel, label="No file selected")
        self.file_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        load_btn = wx.Button(panel, label="Load Text File...")
        load_btn.Bind(wx.EVT_BUTTON, self.on_load_file)

        file_sizer.Add(self.file_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        file_sizer.Add(load_btn, 0, wx.EXPAND)

        main_sizer.Add(file_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Progress bar
        self.progress_bar = wx.Gauge(panel, range=100, size=(-1, 20))
        self.progress_label = wx.StaticText(panel, label="Ready")
        main_sizer.Add(self.progress_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        main_sizer.Add(self.progress_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Status bar
        self.status_bar = wx.StatusBar(panel)
        self.status_bar.SetStatusText("Please load a text file to begin")
        main_sizer.Add(self.status_bar, 0, wx.EXPAND)

        # Search controls
        search_sizer = wx.GridBagSizer(5, 5)

        # Query input
        search_sizer.Add(wx.StaticText(panel, label="Search Query:"), pos=(0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.search_text = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER)
        self.search_text.Bind(wx.EVT_TEXT_ENTER, self.on_search)
        search_sizer.Add(self.search_text, pos=(0, 1), span=(1, 2), flag=wx.EXPAND)

        # Search type
        search_sizer.Add(wx.StaticText(panel, label="Search Type:"), pos=(1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.search_type = wx.Choice(panel, choices=["Semantic Search", "Regular Search"])
        self.search_type.SetSelection(0)
        search_sizer.Add(self.search_type, pos=(1, 1), flag=wx.EXPAND)

        # Case sensitive (for regular search)
        self.case_sensitive = wx.CheckBox(panel, label="Case Sensitive")
        search_sizer.Add(self.case_sensitive, pos=(1, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        # Number of results
        search_sizer.Add(wx.StaticText(panel, label="Results:"), pos=(2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.num_results = wx.SpinCtrl(panel, value="5", min=1, max=50)
        search_sizer.Add(self.num_results, pos=(2, 1), flag=wx.EXPAND)

        # Search button
        search_btn = wx.Button(panel, label="Search")
        search_btn.Bind(wx.EVT_BUTTON, self.on_search)
        search_sizer.Add(search_btn, pos=(2, 2), flag=wx.EXPAND)

        search_sizer.AddGrowableCol(1)
        main_sizer.Add(search_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Results
        results_label = wx.StaticText(panel, label="Search Results:")
        results_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(results_label, 0, wx.ALL, 5)

        self.results_list = ResultsListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        main_sizer.Add(self.results_list, 1, wx.EXPAND | wx.ALL, 10)

        # Database info
        info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.info_label = wx.StaticText(panel, label="Database: Not initialized")
        info_sizer.Add(self.info_label, 1, wx.ALIGN_CENTER_VERTICAL)

        refresh_btn = wx.Button(panel, label="Refresh Info")
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh_info)
        info_sizer.Add(refresh_btn, 0, wx.LEFT, 10)

        main_sizer.Add(info_sizer, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main_sizer)

        # Bind events
        self.search_type.Bind(wx.EVT_CHOICE, self.on_search_type_change)

    def on_load_file(self, event):
        """Handle file loading"""
        with wx.FileDialog(self, "Open text file",
                           wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*") as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            filepath = fileDialog.GetPath()
            self.load_file(filepath)

    def load_file(self, filepath):
        """Load and process the selected file"""
        self.current_file = filepath
        self.file_label.SetLabel(f"File: {os.path.basename(filepath)}")
        self.status_bar.SetStatusText(f"Loading file: {filepath}")

        # Update UI
        self.progress_bar.SetValue(0)
        self.progress_label.SetLabel("Initializing...")
        self.search_text.Disable()

        # Start processing in background thread
        self.searcher.set_file_path(filepath)
        thread = threading.Thread(target=self.process_file)
        thread.daemon = True
        thread.start()

    def process_file(self):
        """Process file in background thread"""

        def progress_callback(current, total, message):
            wx.CallAfter(self.update_progress, current, total, message)

        success = self.searcher.initialize(progress_callback)

        wx.CallAfter(self.on_processing_complete, success)

    def update_progress(self, current, total, message):
        """Update progress bar from background thread"""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.SetValue(percent)
            self.progress_label.SetLabel(f"{message}: {current}/{total} ({percent}%)")

    def on_processing_complete(self, success):
        """Called when file processing is complete"""
        if success:
            self.progress_bar.SetValue(100)
            self.progress_label.SetLabel("Ready for searches")
            self.status_bar.SetStatusText(f"Ready: {len(self.searcher.sentences)} sentences loaded")
            self.search_text.Enable()
            self.search_text.SetFocus()
            self.update_database_info()
        else:
            self.progress_label.SetLabel("Failed to process file")
            self.status_bar.SetStatusText("Error processing file")
            wx.MessageBox("Failed to process the selected file. Please try another file.", "Error",
                          wx.OK | wx.ICON_ERROR)

    def on_search(self, event):
        """Handle search button click"""
        if not self.searcher.is_initialized:
            wx.MessageBox("Please load and process a text file first.", "Not Ready", wx.OK | wx.ICON_WARNING)
            return

        query = self.search_text.GetValue().strip()
        if not query:
            wx.MessageBox("Please enter a search query.", "Empty Query", wx.OK | wx.ICON_WARNING)
            return

        # Perform search
        search_type = self.search_type.GetSelection()  # 0 = semantic, 1 = regular
        top_k = self.num_results.GetValue()

        if search_type == 0:  # Semantic search
            results = self.searcher.semantic_search(query, top_k)
            search_type_str = "semantic"
        else:  # Regular search
            case_sensitive = self.case_sensitive.GetValue()
            results = self.searcher.regular_search(query, case_sensitive)
            search_type_str = "regular"

        # Update results
        self.results_list.update_results(results, search_type_str)
        self.status_bar.SetStatusText(f"Found {len(results)} results for: '{query}'")

    def on_search_type_change(self, event):
        """Handle search type change"""
        search_type = self.search_type.GetSelection()
        # Show/hide case sensitive checkbox based on search type
        self.case_sensitive.Show(search_type == 1)  # Show only for regular search
        self.GetSizer().Layout()

    def on_refresh_info(self, event):
        """Refresh database information"""
        self.update_database_info()

    def update_database_info(self):
        """Update database information display"""
        info = self.searcher.get_database_info()
        if info.get("status") == "ready":
            self.info_label.SetLabel(
                f"Database: {info.get('document_count', 0)} documents, "
                f"{info.get('sentences_loaded', 0)} sentences"
            )
        else:
            self.info_label.SetLabel("Database: Not initialized")


class TextSearchApp(wx.App):
    def OnInit(self):
        self.frame = TextSearchFrame(None)
        self.SetTopWindow(self.frame)
        return True


def main():
    app = TextSearchApp()
    app.MainLoop()


if __name__ == "__main__":
    main()
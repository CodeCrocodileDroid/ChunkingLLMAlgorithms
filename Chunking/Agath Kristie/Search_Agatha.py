import re
import os
import wx
import wx.lib.mixins.listctrl as listmix
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import threading
import time


class NovelSearcher:
    def __init__(self, persist_directory: str = "./agatha_christie_db"):
        """
        Initialize the novel searcher for Agatha Christie books.
        """
        self.persist_directory = persist_directory
        self.novels = {}  # {filename: {sentences: [], title: str, author: str}}
        self.model = None
        self.client = None
        self.collection = None
        self.is_initialized = False

    def _extract_metadata(self, filename, content):
        """Extract metadata from novel content"""
        metadata = {
            "filename": filename,
            "title": os.path.splitext(filename)[0],
            "author": "Agatha Christie",
            "sentences": []
        }

        # Try to extract title from content (first line or common patterns)
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                if not line.startswith(('CHAPTER', 'Chapter', 'CHAPTER', 'PART', 'Part')):
                    metadata["title"] = line
                    break

        return metadata

    def load_novels_from_folder(self, folder_path: str) -> Dict:
        """
        Load all text novels from a folder.
        """
        novels = {}
        supported_extensions = {'.txt', '.pdf'}  # Add PDF support later if needed

        if not os.path.exists(folder_path):
            return novels

        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath) and any(filename.lower().endswith(ext) for ext in supported_extensions):
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()

                    # Split into sentences
                    sentences = re.split(r'[.!?]+', content)
                    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

                    metadata = self._extract_metadata(filename, content)
                    metadata["sentences"] = sentences
                    metadata["filepath"] = filepath
                    metadata["sentence_count"] = len(sentences)

                    novels[filename] = metadata
                    print(f"Loaded {filename}: {len(sentences)} sentences")

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        return novels

    def _needs_reindexing(self, current_novels: Dict) -> bool:
        """
        Check if we need to reindex the database.
        """
        if not os.path.exists(self.persist_directory):
            return True

        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection("christie_novels")
            if collection.count() == 0:
                return True

            # Check if novel count matches
            existing_count = collection.count()
            current_count = sum(len(novel["sentences"]) for novel in current_novels.values())

            # Allow some tolerance for minor changes
            return abs(existing_count - current_count) > 10

        except:
            return True

    def initialize(self, novels: Dict, progress_callback=None) -> bool:
        """
        Initialize the system with Agatha Christie novels.
        """
        if not novels:
            return False

        self.novels = novels

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
                name="christie_novels",
                metadata={"description": "Agatha Christie novels embeddings"}
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            return False

        # Check if we need to compute embeddings
        if self._needs_reindexing(novels):
            success = self._store_embeddings_batched(progress_callback)
            if not success:
                return False

        self.is_initialized = True
        return True

    def _store_embeddings_batched(self, progress_callback=None) -> bool:
        """
        Compute embeddings and store them in ChromaDB for all novels.
        """
        try:
            # Clear existing collection
            try:
                self.client.delete_collection("christie_novels")
            except:
                pass

            self.collection = self.client.get_or_create_collection(
                name="christie_novels",
                metadata={"description": "Agatha Christie novels embeddings"}
            )

            # Prepare all data
            all_documents = []
            all_metadatas = []
            all_ids = []
            all_embeddings = []

            total_sentences = sum(len(novel["sentences"]) for novel in self.novels.values())
            processed_sentences = 0

            # Process each novel
            for filename, novel_data in self.novels.items():
                sentences = novel_data["sentences"]
                title = novel_data["title"]

                # Process in batches
                embedding_batch_size = 32

                for i in range(0, len(sentences), embedding_batch_size):
                    batch_sentences = sentences[i:i + embedding_batch_size]

                    # Compute embeddings for this batch
                    batch_embeddings = self.model.encode(batch_sentences).tolist()

                    # Prepare documents for this batch
                    for j, sentence in enumerate(batch_sentences):
                        absolute_index = i + j
                        all_documents.append(sentence)
                        all_metadatas.append({
                            "filename": filename,
                            "title": title,
                            "author": "Agatha Christie",
                            "position": absolute_index,
                            "novel_position": f"{filename}_{absolute_index}",
                            "sentence_length": len(sentence)
                        })
                        all_ids.append(f"{filename}_{absolute_index}")
                        all_embeddings.append(batch_embeddings[j])

                    processed_sentences += len(batch_sentences)

                    # Update progress
                    if progress_callback:
                        progress_callback(processed_sentences, total_sentences,
                                          f"Processing: {title}")

            # Store in database in batches
            db_batch_size = 100
            print(f"Storing {len(all_documents)} embeddings in database...")

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

    def semantic_search(self, query: str, top_k: int = 10, novel_filter: str = None) -> List[
        Tuple[str, str, int, float]]:
        """
        Perform semantic search across all novels.
        Returns: (sentence, novel_title, position, similarity_score)
        """
        if not self.is_initialized or self.collection is None:
            return []

        try:
            top_k = min(top_k, 50)

            # Build filter if novel is specified
            where_filter = None
            if novel_filter and novel_filter != "All Novels":
                where_filter = {"title": novel_filter}

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
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
                    novel_title = metadata.get('title', 'Unknown')
                    position = metadata.get('position', 0)
                    search_results.append((doc, novel_title, position, similarity))

            return search_results

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def regular_search(self, query: str, case_sensitive: bool = False, novel_filter: str = None) -> List[
        Tuple[str, str, int]]:
        """
        Perform regular text search across novels.
        """
        if not self.is_initialized:
            return []

        results = []

        if not case_sensitive:
            query = query.lower()

        for filename, novel_data in self.novels.items():
            novel_title = novel_data["title"]

            # Skip if novel filter is specified and doesn't match
            if novel_filter and novel_filter != "All Novels" and novel_title != novel_filter:
                continue

            for i, sentence in enumerate(novel_data["sentences"]):
                search_text = sentence if case_sensitive else sentence.lower()
                if query in search_text:
                    results.append((sentence, novel_title, i))

        return results

    def get_novel_titles(self) -> List[str]:
        """Get list of all novel titles"""
        return ["All Novels"] + sorted([novel["title"] for novel in self.novels.values()])

    def get_database_info(self) -> Dict:
        """Get database information"""
        if not self.is_initialized or self.collection is None:
            return {"status": "not_initialized"}

        try:
            total_sentences = sum(len(novel["sentences"]) for novel in self.novels.values())
            return {
                "status": "ready",
                "novel_count": len(self.novels),
                "sentence_count": total_sentences,
                "document_count": self.collection.count()
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
        self.InsertColumn(1, "Novel", width=150)
        self.InsertColumn(2, "Position", width=80)
        self.InsertColumn(3, "Score", width=80)
        self.InsertColumn(4, "Text", width=600)

    def update_results(self, results, search_type="semantic"):
        """Update the list with search results"""
        self.DeleteAllItems()

        for i, result in enumerate(results, 1):
            if search_type == "regular":
                sentence, novel_title, position = result
                score = "N/A"
            else:
                sentence, novel_title, position, score = result
                score = f"{score:.3f}"

            index = self.InsertItem(self.GetItemCount(), str(i))
            self.SetItem(index, 1, novel_title)
            self.SetItem(index, 2, str(position))
            self.SetItem(index, 3, score)
            self.SetItem(index, 4, sentence)


class AgathaChristieSearchFrame(wx.Frame):
    def __init__(self, parent, title="Agatha Christie Novel Search"):
        super(AgathaChristieSearchFrame, self).__init__(parent, title=title, size=(1200, 800))

        self.searcher = NovelSearcher()
        self.novels_folder = ""

        self.init_ui()
        self.Centre()
        self.Show()

    def init_ui(self):
        """Initialize the user interface"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Title with Agatha Christie theme
        title = wx.StaticText(panel, label="🔍 Agatha Christie Novel Search")
        title_font = wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 10)

        subtitle = wx.StaticText(panel, label="Search through the complete works of Agatha Christie")
        subtitle_font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_ITALIC, wx.FONTWEIGHT_NORMAL)
        subtitle.SetFont(subtitle_font)
        main_sizer.Add(subtitle, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        # Folder selection section
        folder_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.folder_label = wx.StaticText(panel, label="No folder selected")
        self.folder_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        load_btn = wx.Button(panel, label="Select Novels Folder...")
        load_btn.Bind(wx.EVT_BUTTON, self.on_select_folder)

        folder_sizer.Add(self.folder_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        folder_sizer.Add(load_btn, 0, wx.EXPAND)

        main_sizer.Add(folder_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Progress bar
        self.progress_bar = wx.Gauge(panel, range=100, size=(-1, 20))
        self.progress_label = wx.StaticText(panel, label="Ready to load Agatha Christie novels")
        main_sizer.Add(self.progress_bar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        main_sizer.Add(self.progress_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # Status bar
        self.status_bar = wx.StatusBar(panel)
        self.status_bar.SetStatusText("Please select a folder containing Agatha Christie novels in text format")
        main_sizer.Add(self.status_bar, 0, wx.EXPAND)

        # Search controls
        search_sizer = wx.GridBagSizer(5, 5)

        # Query input
        search_sizer.Add(wx.StaticText(panel, label="Search Query:"), pos=(0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.search_text = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER, size=(300, -1))
        self.search_text.Bind(wx.EVT_TEXT_ENTER, self.on_search)
        search_sizer.Add(self.search_text, pos=(0, 1), flag=wx.EXPAND)

        # Novel filter
        search_sizer.Add(wx.StaticText(panel, label="Filter Novel:"), pos=(0, 2),
                         flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)
        self.novel_filter = wx.Choice(panel, choices=["All Novels"])
        self.novel_filter.SetSelection(0)
        search_sizer.Add(self.novel_filter, pos=(0, 3), flag=wx.EXPAND)

        # Search type
        search_sizer.Add(wx.StaticText(panel, label="Search Type:"), pos=(1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.search_type = wx.Choice(panel, choices=["Semantic Search", "Regular Search"])
        self.search_type.SetSelection(0)
        search_sizer.Add(self.search_type, pos=(1, 1), flag=wx.EXPAND)

        # Case sensitive (for regular search)
        self.case_sensitive = wx.CheckBox(panel, label="Case Sensitive")
        search_sizer.Add(self.case_sensitive, pos=(1, 2), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)

        # Number of results
        search_sizer.Add(wx.StaticText(panel, label="Results:"), pos=(1, 3), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
                         border=10)
        self.num_results = wx.SpinCtrl(panel, value="10", min=1, max=50)
        search_sizer.Add(self.num_results, pos=(1, 4), flag=wx.EXPAND)

        # Search button
        search_btn = wx.Button(panel, label="Search Christie Novels")
        search_btn.Bind(wx.EVT_BUTTON, self.on_search)
        search_sizer.Add(search_btn, pos=(2, 0), span=(1, 5), flag=wx.EXPAND | wx.TOP, border=5)

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
        self.info_label = wx.StaticText(panel, label="No novels loaded")

        stats_btn = wx.Button(panel, label="Show Statistics")
        stats_btn.Bind(wx.EVT_BUTTON, self.on_show_stats)

        info_sizer.Add(self.info_label, 1, wx.ALIGN_CENTER_VERTICAL)
        info_sizer.Add(stats_btn, 0, wx.LEFT, 10)

        main_sizer.Add(info_sizer, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main_sizer)

        # Bind events
        self.search_type.Bind(wx.EVT_CHOICE, self.on_search_type_change)

    def on_select_folder(self, event):
        """Handle folder selection"""
        with wx.DirDialog(self, "Select folder containing Agatha Christie novels") as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return

            folder_path = dirDialog.GetPath()
            self.load_novels_folder(folder_path)

    def load_novels_folder(self, folder_path):
        """Load and process novels from the selected folder"""
        self.novels_folder = folder_path
        self.folder_label.SetLabel(f"Folder: {os.path.basename(folder_path)}")
        self.status_bar.SetStatusText(f"Loading novels from: {folder_path}")

        # Update UI
        self.progress_bar.SetValue(0)
        self.progress_label.SetLabel("Scanning folder for novels...")
        self.search_text.Disable()

        # Start processing in background thread
        thread = threading.Thread(target=self.process_novels_folder, args=(folder_path,))
        thread.daemon = True
        thread.start()

    def process_novels_folder(self, folder_path):
        """Process novels folder in background thread"""

        def progress_callback(current, total, message):
            wx.CallAfter(self.update_progress, current, total, message)

        # Load novels
        novels = self.searcher.load_novels_from_folder(folder_path)

        if not novels:
            wx.CallAfter(self.on_processing_complete, False, "No novels found in folder")
            return

        # Initialize searcher
        success = self.searcher.initialize(novels, progress_callback)
        wx.CallAfter(self.on_processing_complete, success, f"Found {len(novels)} novels")

    def update_progress(self, current, total, message):
        """Update progress bar from background thread"""
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.SetValue(percent)
            self.progress_label.SetLabel(f"{message} - {current}/{total} ({percent}%)")

    def on_processing_complete(self, success, message):
        """Called when novel processing is complete"""
        if success:
            self.progress_bar.SetValue(100)
            self.progress_label.SetLabel("Ready to search Agatha Christie novels!")
            self.status_bar.SetStatusText(message)
            self.search_text.Enable()
            self.search_text.SetFocus()

            # Update novel filter dropdown
            novel_titles = self.searcher.get_novel_titles()
            self.novel_filter.SetItems(novel_titles)
            self.novel_filter.SetSelection(0)

            self.update_database_info()
        else:
            self.progress_label.SetLabel("Failed to process novels")
            self.status_bar.SetStatusText("Error processing novels")
            wx.MessageBox(f"Failed to process novels: {message}", "Error", wx.OK | wx.ICON_ERROR)

    def on_search(self, event):
        """Handle search button click"""
        if not self.searcher.is_initialized:
            wx.MessageBox("Please load Agatha Christie novels first.", "Not Ready", wx.OK | wx.ICON_WARNING)
            return

        query = self.search_text.GetValue().strip()
        if not query:
            wx.MessageBox("Please enter a search query.", "Empty Query", wx.OK | wx.ICON_WARNING)
            return

        # Get search parameters
        search_type = self.search_type.GetSelection()  # 0 = semantic, 1 = regular
        top_k = self.num_results.GetValue()
        novel_filter = self.novel_filter.GetStringSelection()

        # Perform search
        if search_type == 0:  # Semantic search
            results = self.searcher.semantic_search(query, top_k, novel_filter)
            search_type_str = "semantic"
        else:  # Regular search
            case_sensitive = self.case_sensitive.GetValue()
            results = self.searcher.regular_search(query, case_sensitive, novel_filter)
            search_type_str = "regular"

        # Update results
        self.results_list.update_results(results, search_type_str)

        # Update status
        filter_text = f" in '{novel_filter}'" if novel_filter != "All Novels" else ""
        self.status_bar.SetStatusText(f"Found {len(results)} results for '{query}'{filter_text}")

    def on_search_type_change(self, event):
        """Handle search type change"""
        search_type = self.search_type.GetSelection()
        # Show/hide case sensitive checkbox based on search type
        self.case_sensitive.Show(search_type == 1)  # Show only for regular search
        self.GetSizer().Layout()

    def on_show_stats(self, event):
        """Show database statistics"""
        info = self.searcher.get_database_info()

        if info.get("status") == "ready":
            stats_msg = f"""Agatha Christie Collection Statistics:

• Novels loaded: {info.get('novel_count', 0)}
• Total sentences: {info.get('sentence_count', 0):,}
• Documents in database: {info.get('document_count', 0):,}

Available Novels:
"""
            for title in sorted(self.searcher.novels.keys()):
                novel = self.searcher.novels[title]
                stats_msg += f"  • {novel['title']} ({len(novel['sentences'])} sentences)\n"

            wx.MessageBox(stats_msg, "Collection Statistics", wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox("No novels loaded or database not initialized.", "Statistics", wx.OK | wx.ICON_WARNING)

    def update_database_info(self):
        """Update database information display"""
        info = self.searcher.get_database_info()
        if info.get("status") == "ready":
            self.info_label.SetLabel(
                f"Agatha Christie Collection: {info.get('novel_count', 0)} novels, "
                f"{info.get('sentence_count', 0):,} sentences"
            )
        else:
            self.info_label.SetLabel("No novels loaded")


class AgathaChristieApp(wx.App):
    def OnInit(self):
        self.frame = AgathaChristieSearchFrame(None)
        self.SetTopWindow(self.frame)
        return True


def main():
    app = AgathaChristieApp()
    app.MainLoop()


if __name__ == "__main__":
    main()
import re
import os
import wx
import wx.lib.mixins.listctrl as listmix
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import threading
import time


class SmartNovelSearcher:
    def __init__(self, persist_directory: str = "./christie_db"):
        self.persist_directory = persist_directory
        self.novels = {}
        self.model = None
        self.client = None
        self.collection = None
        self.is_initialized = False

    def load_novels_from_folder(self, folder_path: str) -> Dict:
        """Load novels from folder but don't process if DB exists"""
        novels = {}

        if not os.path.exists(folder_path):
            return novels

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()

                    # Just get basic info, don't process sentences if DB exists
                    novels[filename] = {
                        "title": os.path.splitext(filename)[0],
                        "filepath": filepath,
                        "content": content  # Store content for later processing if needed
                    }

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        return novels

    def _database_exists(self) -> bool:
        """Check if database already exists and is valid"""
        try:
            if not os.path.exists(self.persist_directory):
                return False

            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection("christie_novels")

            # Check if collection has documents
            if collection.count() > 0:
                print(f"✓ Found existing database with {collection.count()} documents")
                return True
            else:
                print("✗ Database exists but is empty")
                return False

        except Exception as e:
            print(f"✗ Database check failed: {e}")
            return False

    def _needs_reindexing(self, current_novels: Dict) -> bool:
        """Check if we need to reindex based on folder contents"""
        if not self._database_exists():
            return True

        # If DB exists and is valid, we don't need to reindex
        return False

    def initialize_from_existing_db(self) -> bool:
        """Initialize using existing database without processing files"""
        try:
            print("Initializing from existing database...")

            # Load model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            # Connect to existing DB
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_collection("christie_novels")

            # Get novel list from database metadata
            try:
                # Sample query to get metadata
                sample_results = self.collection.get(limit=1, include=["metadatas"])
                if sample_results['metadatas']:
                    # We can't easily get all unique novels from ChromaDB without querying everything
                    # For now, we'll mark as initialized and novels will be loaded on demand
                    pass
            except:
                pass

            self.is_initialized = True
            print("✓ Successfully loaded existing database")
            return True

        except Exception as e:
            print(f"✗ Failed to initialize from existing DB: {e}")
            return False

    def initialize_with_novels(self, novels: Dict, progress_callback=None) -> bool:
        """Initialize by processing novels and creating new database"""
        if not novels:
            return False

        self.novels = novels

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            return False

        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Clear any existing collection
            try:
                self.client.delete_collection("christie_novels")
            except:
                pass

            self.collection = self.client.create_collection(name="christie_novels")
        except Exception as e:
            return False

        # Process novels and create embeddings
        success = self._store_embeddings_batched(progress_callback)
        if success:
            self.is_initialized = True
        return success

    def _store_embeddings_batched(self, progress_callback=None) -> bool:
        """Process novels and store embeddings in database"""
        try:
            all_documents = []
            all_metadatas = []
            all_ids = []

            total_sentences = 0
            processed_novels = {}

            # First pass: count total sentences and process novel info
            for filename, novel_data in self.novels.items():
                content = novel_data["content"]
                sentences = re.split(r'[.!?]+', content)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

                processed_novels[filename] = {
                    "title": novel_data["title"],
                    "sentences": sentences,
                    "filepath": novel_data["filepath"]
                }
                total_sentences += len(sentences)

            self.novels = processed_novels
            processed = 0

            # Second pass: create embeddings
            for filename, novel_data in self.novels.items():
                sentences = novel_data["sentences"]
                title = novel_data["title"]

                # Process in batches
                for i in range(0, len(sentences), 16):
                    batch = sentences[i:i + 16]
                    embeddings = self.model.encode(batch).tolist()

                    for j, (sentence, embedding) in enumerate(zip(batch, embeddings)):
                        all_documents.append(sentence)
                        all_metadatas.append({
                            "filename": filename,
                            "title": title,
                            "position": i + j
                        })
                        all_ids.append(f"{filename}_{i + j}")

                    processed += len(batch)
                    if progress_callback:
                        progress_callback(processed, total_sentences, f"Processing {title}")

            # Store in batches
            for i in range(0, len(all_documents), 50):
                end_idx = min(i + 50, len(all_documents))
                batch_docs = all_documents[i:end_idx]
                batch_meta = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]

                # Get embeddings for this batch
                batch_embeddings = self.model.encode(batch_docs).tolist()

                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )

                if progress_callback and i % 500 == 0:
                    progress_callback(i, len(all_documents), "Storing embeddings")

            print(f"✓ Created new database with {len(all_documents)} documents")
            return True

        except Exception as e:
            print(f"✗ Error creating database: {e}")
            return False

    def semantic_search(self, query: str, top_k: int = 10, novel_filter: str = None):
        if not self.is_initialized:
            return []

        try:
            where_filter = {"title": novel_filter} if novel_filter and novel_filter != "All" else None

            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, 20),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            search_results = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0],
                                                   results['distances'][0]):
                    search_results.append((
                        doc,
                        metadata.get('title', 'Unknown'),
                        metadata.get('position', 0),
                        1 - distance
                    ))

            return search_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_novel_titles(self):
        """Get novel titles from current loaded novels or scan DB"""
        if self.novels:
            return ["All"] + sorted([novel["title"] for novel in self.novels.values()])
        else:
            # Try to get titles from database
            try:
                # Get a sample of documents to extract titles
                sample = self.collection.get(limit=1000, include=["metadatas"])
                titles = set()
                for metadata in sample['metadatas']:
                    titles.add(metadata.get('title', 'Unknown'))
                return ["All"] + sorted(list(titles))
            except:
                return ["All"]

    def get_stats(self):
        if not self.is_initialized:
            return {}
        try:
            return {
                "novels": len(self.get_novel_titles()) - 1,  # Subtract "All"
                "documents": self.collection.count()
            }
        except:
            return {}


class CompactResultsList(wx.ListCtrl):
    def __init__(self, parent):
        super().__init__(parent, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        self.setup_columns()

    def setup_columns(self):
        self.InsertColumn(0, "#", width=30)
        self.InsertColumn(1, "Novel", width=120)
        self.InsertColumn(2, "Text", width=450)

    def update_results(self, results, search_type="semantic"):
        self.DeleteAllItems()
        for i, result in enumerate(results, 1):
            if search_type == "regular":
                sentence, novel, pos = result
                score = ""
            else:
                sentence, novel, pos, score = result
                score = f" ({score:.2f})"

            index = self.InsertItem(self.GetItemCount(), str(i))
            self.SetItem(index, 1, novel)
            display_text = sentence[:200] + "..." if len(sentence) > 200 else sentence
            self.SetItem(index, 2, display_text)


class SmartChristieSearch(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Agatha Christie Search", size=(800, 600))
        self.searcher = SmartNovelSearcher()
        self.current_folder = ""
        self.init_ui()
        self.Centre()
        self.Show()

        # Auto-detect existing database on startup
        self.auto_detect_database()

    def init_ui(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Title
        title = wx.StaticText(panel, label="Agatha Christie Novel Search")
        title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        # Database status
        self.db_status = wx.StaticText(panel, label="Database: Not loaded")
        sizer.Add(self.db_status, 0, wx.ALL, 3)

        # File controls
        file_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.file_label = wx.StaticText(panel, label="No folder selected", size=(300, -1))
        load_btn = wx.Button(panel, label="Open Folder...")
        load_btn.Bind(wx.EVT_BUTTON, self.on_load_folder)

        file_sizer.Add(self.file_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        file_sizer.Add(load_btn, 0, wx.EXPAND)
        sizer.Add(file_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Progress
        self.progress = wx.Gauge(panel, range=100, size=(-1, 15))
        self.progress_label = wx.StaticText(panel, label="Ready")
        sizer.Add(self.progress, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        sizer.Add(self.progress_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        # Search controls
        search_sizer = wx.GridBagSizer(3, 3)

        search_sizer.Add(wx.StaticText(panel, label="Search:"), pos=(0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.search_text = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER)
        self.search_text.Bind(wx.EVT_TEXT_ENTER, self.on_search)
        search_sizer.Add(self.search_text, pos=(0, 1), flag=wx.EXPAND)

        search_sizer.Add(wx.StaticText(panel, label="Novel:"), pos=(0, 2), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
                         border=5)
        self.novel_choice = wx.Choice(panel, choices=["All"])
        self.novel_choice.SetSelection(0)
        search_sizer.Add(self.novel_choice, pos=(0, 3), flag=wx.EXPAND)

        search_sizer.Add(wx.StaticText(panel, label="Type:"), pos=(1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        self.type_choice = wx.Choice(panel, choices=["Semantic", "Keyword"])
        self.type_choice.SetSelection(0)
        search_sizer.Add(self.type_choice, pos=(1, 1), flag=wx.EXPAND)

        self.case_check = wx.CheckBox(panel, label="Case")
        search_sizer.Add(self.case_check, pos=(1, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        search_sizer.Add(wx.StaticText(panel, label="Results:"), pos=(1, 3), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT,
                         border=5)
        self.results_spin = wx.SpinCtrl(panel, value="10", min=1, max=20, size=(60, -1))
        search_sizer.Add(self.results_spin, pos=(1, 4), flag=wx.EXPAND)

        search_btn = wx.Button(panel, label="Search")
        search_btn.Bind(wx.EVT_BUTTON, self.on_search)
        search_sizer.Add(search_btn, pos=(2, 0), span=(1, 5), flag=wx.EXPAND | wx.TOP, border=3)

        search_sizer.AddGrowableCol(1)
        sizer.Add(search_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Results
        sizer.Add(wx.StaticText(panel, label="Results:"), 0, wx.ALL, 3)
        self.results_list = CompactResultsList(panel)
        sizer.Add(self.results_list, 1, wx.EXPAND | wx.ALL, 5)

        # Status
        self.status = wx.StatusBar(panel)
        self.status.SetStatusText("Ready")
        sizer.Add(self.status, 0, wx.EXPAND)

        panel.SetSizer(sizer)

        # Bind events
        self.type_choice.Bind(wx.EVT_CHOICE, self.on_type_change)
        self.on_type_change(None)

    def auto_detect_database(self):
        """Automatically detect and load existing database on startup"""
        if self.searcher._database_exists():
            success = self.searcher.initialize_from_existing_db()
            if success:
                self.db_status.SetLabel("Database: Loaded existing")
                self.status.SetStatusText("Loaded existing database - ready to search!")
                self.search_text.Enable()

                # Update novel list from database
                novels = self.searcher.get_novel_titles()
                self.novel_choice.SetItems(novels)
                self.novel_choice.SetSelection(0)

                # Show stats
                stats = self.searcher.get_stats()
                if stats:
                    self.db_status.SetLabel(f"Database: {stats['novels']} novels, {stats['documents']} passages")
            else:
                self.db_status.SetLabel("Database: Found but failed to load")
        else:
            self.db_status.SetLabel("Database: Not found - load a folder to create")

    def on_type_change(self, event):
        self.case_check.Show(self.type_choice.GetSelection() == 1)
        self.Layout()

    def on_load_folder(self, event):
        with wx.DirDialog(self, "Select folder with Christie novels") as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.load_folder(dlg.GetPath())

    def load_folder(self, path):
        self.current_folder = path
        self.file_label.SetLabel(os.path.basename(path))
        self.status.SetStatusText("Checking folder...")
        self.progress.SetValue(0)
        self.search_text.Disable()

        thread = threading.Thread(target=self.process_folder, args=(path,))
        thread.daemon = True
        thread.start()

    def process_folder(self, path):
        def progress_callback(current, total, msg):
            wx.CallAfter(self.update_progress, current, total, msg)

        # Check if we can use existing database
        if self.searcher._database_exists():
            wx.CallAfter(self.on_processing_done, True, 0, "Using existing database")
            return

        # Otherwise, process novels and create new database
        novels = self.searcher.load_novels_from_folder(path)
        if novels:
            success = self.searcher.initialize_with_novels(novels, progress_callback)
            wx.CallAfter(self.on_processing_done, success, len(novels), "Created new database")
        else:
            wx.CallAfter(self.on_processing_done, False, 0, "No novels found")

    def update_progress(self, current, total, msg):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress.SetValue(percent)
            self.progress_label.SetLabel(f"{msg} - {percent}%")

    def on_processing_done(self, success, novel_count, message):
        if success:
            self.progress.SetValue(100)
            self.progress_label.SetLabel("Ready!")
            self.status.SetStatusText(message)
            self.search_text.Enable()
            self.search_text.SetFocus()

            # Update novel list
            novels = self.searcher.get_novel_titles()
            self.novel_choice.SetItems(novels)
            self.novel_choice.SetSelection(0)

            # Update database status
            stats = self.searcher.get_stats()
            if stats:
                self.db_status.SetLabel(f"Database: {stats['novels']} novels, {stats['documents']} passages")
            else:
                self.db_status.SetLabel("Database: Ready")
        else:
            self.progress_label.SetLabel("Failed")
            self.status.SetStatusText(message)
            wx.MessageBox(f"Failed: {message}", "Error", wx.OK | wx.ICON_ERROR)

    def on_search(self, event):
        if not self.searcher.is_initialized:
            wx.MessageBox("Please load novels first", "Not Ready", wx.OK | wx.ICON_WARNING)
            return

        query = self.search_text.GetValue().strip()
        if not query:
            return

        search_type = self.type_choice.GetSelection()  # 0 = semantic, 1 = keyword
        novel_filter = self.novel_choice.GetStringSelection()
        top_k = self.results_spin.GetValue()

        if search_type == 0:
            results = self.searcher.semantic_search(query, top_k, novel_filter if novel_filter != "All" else None)
        else:
            # For keyword search, we'd need the actual novel content
            # For now, just show message
            wx.MessageBox("Keyword search requires loaded novels. Please load a folder first.", "Info",
                          wx.OK | wx.ICON_INFORMATION)
            return

        self.results_list.update_results(results, "semantic" if search_type == 0 else "regular")
        self.status.SetStatusText(f"Found {len(results)} results for '{query}'")


class ChristieApp(wx.App):
    def OnInit(self):
        self.frame = SmartChristieSearch()
        return True


if __name__ == "__main__":
    app = ChristieApp()
    app.MainLoop()
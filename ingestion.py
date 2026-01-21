"""
RAG Ingestion Pipeline using Local Engineering Documents
Loads csv metadata files and local markdown/text documents, enriches documents with metadata,
Chunks the documents and stores vectors in MongoDB.
"""

import os
import csv
from dotenv import load_dotenv
from typing import List, Dict, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

# Document source
DOCS_DIRECTORY = "engineering_docs"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class LocalFileLoader:
    """Load documents from local engineering_docs directory."""

    def __init__(self, docs_directory: str):
        self.docs_directory = Path(docs_directory)
        if not self.docs_directory.exists():
            raise ValueError(f"Directory not found: {docs_directory}")

        # Supported file extensions for documents (CSV is metadata only)
        self.supported_extensions = {'.md', '.txt'}

        # Load metadata from CSV files
        self.metadata_lookup = self._load_csv_metadata()

    def _load_csv_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Load metadata from all CSV files in the directory.
        Returns a dictionary mapping document titles to their metadata.
        """
        metadata_lookup = {}

        # Find all CSV files
        csv_files = list(self.docs_directory.rglob('*.csv'))
        print(f"Found {len(csv_files)} CSV metadata files")

        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
                    reader = csv.DictReader(f)
                    for row in reader:
                        title = row.get('Title', '').strip()
                        if title:
                            # Store all CSV fields as metadata
                            metadata_lookup[title] = {
                                'area': row.get('Area', ''),
                                'auth_requirement': row.get('Auth Requirement', ''),
                                'last_updated': row.get('Last Updated', ''),
                                'last_edited_by': row.get('Last edited by', ''),
                                'last_edited_time': row.get('Last edited time', ''),
                                'links': row.get('Links', ''),
                                'module': row.get('Module', ''),
                                'owner': row.get('Owner', ''),
                                'owners': row.get('Owners', ''),
                                'related_ticket': row.get('Related Ticket', ''),
                                'repo': row.get('Repo', ''),
                                'roles': row.get('Roles', ''),
                                'route_url': row.get('Route / URL', ''),
                                'status': row.get('Status', ''),
                                'summary': row.get('Summary', '')
                            }
            except Exception as e:
                print(f"  -> Warning: Could not parse CSV {csv_file}: {e}")

        print(f"Loaded metadata for {len(metadata_lookup)} documents from CSV files")
        return metadata_lookup

    def find_all_files(self) -> List[Path]:
        """
        Recursively find all supported files in the docs directory.

        Returns:
            List of Path objects for supported files
        """
        all_files = []

        for file_path in self.docs_directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                all_files.append(file_path)

        print(f"Found {len(all_files)} supported files in {self.docs_directory}")
        return all_files

    def read_file_content(self, file_path: Path) -> str:
        """
        Read the content of a file.

        Args:
            file_path: Path to the file

        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"  -> Error reading file {file_path}: {e}")
                return ""
        except Exception as e:
            print(f"  -> Error reading file {file_path}: {e}")
            return ""

    def get_file_title(self, file_path: Path) -> str:
        """
        Extract a title from the file path.
        Removes Notion-style hash suffixes (e.g., "Title 2be0f61f..." -> "Title")
        """
        stem = file_path.stem

        # Check if the stem ends with a Notion-style hash (32 hex characters)
        # Pattern: "Title <32-char-hex-hash>"
        parts = stem.split()
        if len(parts) > 1 and len(parts[-1]) == 32:
            # Check if last part is hex
            try:
                int(parts[-1], 16)
                # It's a valid hex hash, remove it
                return ' '.join(parts[:-1])
            except ValueError:
                pass

        return stem

    def _find_csv_metadata(self, title: str) -> Optional[Dict[str, str]]:
        """
        Find CSV metadata for a given title.
        Tries exact match first, then fuzzy matching.
        """
        # Try exact match
        if title in self.metadata_lookup:
            return self.metadata_lookup[title]

        # Try case-insensitive match
        title_lower = title.lower()
        for csv_title, metadata in self.metadata_lookup.items():
            if csv_title.lower() == title_lower:
                return metadata

        return None

    def load_documents(self) -> List[Document]:
        """
        Load all supported files from the docs directory as LangChain Documents.
        Basic metadata only - CSV metadata will be added after chunking.

        Returns:
            List of Document objects
        """
        files = self.find_all_files()
        documents = []

        for i, file_path in enumerate(files, 1):
            title = self.get_file_title(file_path)
            relative_path = file_path.relative_to(self.docs_directory)

            print(f"Processing file {i}/{len(files)}: {relative_path}")

            # Read file content
            content = self.read_file_content(file_path)

            if not content.strip():
                print(f"  -> Skipping empty file")
                continue

            # Get file stats
            stats = file_path.stat()

            # Create Document with base metadata
            # CSV metadata will be added to chunks after splitting
            metadata = {
                "source": "local_file",
                "file_path": str(file_path.absolute()),
                "relative_path": str(relative_path),
                "title": title,
                "file_type": file_path.suffix,
                "file_size": stats.st_size,
                "modified_time": stats.st_mtime
            }

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
            print(f"  -> Loaded {len(content)} characters")

        print(f"\nTotal documents loaded: {len(documents)}")
        return documents

    def enrich_chunks_with_csv_metadata(self, chunks: List[Document]) -> List[Document]:
        """
        Enrich each chunk with CSV metadata based on the document title.
        This is done after chunking to ensure metadata is properly attached to each chunk.

        Args:
            chunks: List of chunked documents

        Returns:
            List of chunks with enriched metadata
        """
        print(f"\nEnriching {len(chunks)} chunks with CSV metadata...")

        # Debug: Show what titles we're looking for vs what we have
        unique_titles = set()
        for chunk in chunks:
            title = chunk.metadata.get("title", "")
            if title:
                unique_titles.add(title)

        print(f"\n  DEBUG: Found {len(unique_titles)} unique document titles:")
        for title in sorted(list(unique_titles)[:10]):
            print(f"    - '{title}'")
        if len(unique_titles) > 10:
            print(f"    ... and {len(unique_titles) - 10} more")

        print(f"\n  DEBUG: CSV has {len(self.metadata_lookup)} titles:")
        for csv_title in sorted(list(self.metadata_lookup.keys())[:10]):
            print(f"    - '{csv_title}'")
        if len(self.metadata_lookup) > 10:
            print(f"    ... and {len(self.metadata_lookup) - 10} more")

        enriched_count = 0
        areas_found = set()
        statuses_found = set()
        unmatched_titles = set()

        for chunk_idx, chunk in enumerate(chunks):
            # Add chunk index to metadata
            chunk.metadata['chunk_index'] = chunk_idx

            # Get the title from the chunk's metadata (inherited from parent document)
            title = chunk.metadata.get("title", "")

            if title:
                # Find matching CSV metadata
                csv_metadata = self._find_csv_metadata(title)
                if csv_metadata:
                    # Update chunk metadata with CSV fields
                    chunk.metadata.update(csv_metadata)
                    enriched_count += 1

                    # Track what we found
                    if csv_metadata.get('area'):
                        areas_found.add(csv_metadata.get('area'))
                    if csv_metadata.get('status'):
                        statuses_found.add(csv_metadata.get('status'))
                else:
                    unmatched_titles.add(title)

        print(f"\n  -> Enriched {enriched_count}/{len(chunks)} chunks with CSV metadata")

        if unmatched_titles:
            print(f"\n  -> WARNING: {len(unmatched_titles)} titles had no CSV match:")
            for title in sorted(list(unmatched_titles)[:5]):
                print(f"     - '{title}'")
            if len(unmatched_titles) > 5:
                print(f"     ... and {len(unmatched_titles) - 5} more")

        print(f"  -> Areas found: {sorted(areas_found)}")
        print(f"  -> Statuses found: {sorted(statuses_found)}")

        return chunks


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    print(f"\nChunking {len(documents)} documents...")
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    return chunks



def setup_mongodb_collection():
    """Set up MongoDB collection for vector storage."""
    connection_params = {
        "serverSelectionTimeoutMS": 5000,
        "connectTimeoutMS": 5000,
        "directConnection": True,
        "tls": False,
        "retryWrites": False
    }
    client = MongoClient(MONGO_DB_URL, **connection_params)
    db = client[DB_NAME]

    # Create collection if it doesn't exist
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        # Clear existing documents for fresh ingestion
        db[COLLECTION_NAME].delete_many({})
        print(f"Cleared existing documents in: {COLLECTION_NAME}")

    return client, db[COLLECTION_NAME]


def create_vector_store(collection, documents: List[Document]):
    """Create embeddings and store in MongoDB Atlas Vector Search."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\nCreating embeddings and storing in MongoDB...")

    # Show sample metadata before storing
    if documents:
        sample_doc = documents[0]
        print(f"\nSample chunk metadata (first chunk):")
        print(f"  Title: {sample_doc.metadata.get('title', 'N/A')}")
        print(f"  Chunk Index: {sample_doc.metadata.get('chunk_index', 'N/A')}")
        print(f"  Area: {sample_doc.metadata.get('area', 'N/A')}")
        print(f"  Owner: {sample_doc.metadata.get('owner', 'N/A')}")
        print(f"  Status: {sample_doc.metadata.get('status', 'N/A')}")
        print(f"  Repo: {sample_doc.metadata.get('repo', 'N/A')}")
        print(f"  Total metadata fields: {len(sample_doc.metadata)}")
        print(f"\n  DEBUG - All metadata keys: {list(sample_doc.metadata.keys())}")

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )

    print(f"\n✓ Successfully stored {len(documents)} document chunks in MongoDB")
    print(f"  Each chunk includes CSV metadata fields (area, owner, status, etc.)")
    return vector_store


def print_vector_search_index_instructions():
    """Print instructions for creating the vector search index in MongoDB Atlas."""
    print("\n" + "=" * 70)
    print("IMPORTANT: Create Vector Search Index in MongoDB Atlas")
    print("=" * 70)
    print(f"""
Create a vector search index with filter fields for metadata:

1. Go to MongoDB Atlas → Your Cluster → Atlas Search → Create Search Index
2. Select "Atlas Vector Search" → JSON Editor
3. Use this configuration:

{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }},
    {{
      "type": "filter",
      "path": "area"
    }},
    {{
      "type": "filter",
      "path": "status"
    }},
    {{
      "type": "filter",
      "path": "owner"
    }},
    {{
      "type": "filter",
      "path": "owners"
    }},
    {{
      "type": "filter",
      "path": "repo"
    }},
    {{
      "type": "filter",
      "path": "title"
    }},
    {{
      "type": "filter",
      "path": "last_edited_by"
    }},
    {{
      "type": "filter",
      "path": "related_ticket"
    }},
    {{
      "type": "filter",
      "path": "module"
    }},
    {{
      "type": "filter",
      "path": "file_type"
    }},
    {{
      "type": "filter",
      "path": "chunk_index"
    }}
  ]
}}

4. Set the index name to: {INDEX_NAME}
5. Select database: {DB_NAME}
6. Select collection: {COLLECTION_NAME}

Wait for the index to become "Active" before running queries.
""")
    print("=" * 70)


def main():
    """Main ingestion pipeline."""
    print("=" * 50)
    print("RAG + metadata - Engineering Docs Ingestion Pipeline")
    print("=" * 50)

    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not DB_NAME:
        raise ValueError("DB_NAME environment variable not set")
    if not COLLECTION_NAME:
        raise ValueError("COLLECTION_NAME environment variable not set")
    if not INDEX_NAME:
        raise ValueError("INDEX_NAME environment variable not set")

    # Step 1: Load documents from local engineering_docs folder
    file_loader = LocalFileLoader(DOCS_DIRECTORY)
    documents = file_loader.load_documents()

    if not documents:
        raise ValueError(f"No documents loaded from {DOCS_DIRECTORY}")

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)

    # Step 3: Enrich chunks with CSV metadata
    # This ensures each chunk has the full metadata attached
    chunks = file_loader.enrich_chunks_with_csv_metadata(chunks)

    # Step 4: Setup MongoDB
    client, collection = setup_mongodb_collection()

    try:
        # Step 5: Create embeddings and store in vector database
        vector_store = create_vector_store(collection, chunks)

        print("\n✅ Ingestion complete!")
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Pages processed: {len(documents)}")
        print(f"   Chunks stored: {len(chunks)}")

        # Print instructions for creating the vector search index
        print_vector_search_index_instructions()

    finally:
        client.close()


if __name__ == "__main__":
    main()

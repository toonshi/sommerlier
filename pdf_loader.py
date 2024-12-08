from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "studyrag"  # Match your index name
DIMENSION = 1536  # Match your configuration

def initialize_pinecone():
    """Initialize Pinecone client."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        print("Pinecone initialized successfully.")
        return pc
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        return None

def create_or_recreate_index(pc):
    """Create or recreate the Pinecone index."""
    try:
        # List existing indexes
        existing_indexes = pc.list_indexes()
        print("Current indexes:", existing_indexes)

        # Recreate index if it exists
        if INDEX_NAME in [idx.name for idx in existing_indexes]:
            print(f"Index '{INDEX_NAME}' exists. Deleting it...")
            pc.delete_index(INDEX_NAME)
            print(f"Index '{INDEX_NAME}' deleted successfully.")

        print("Creating new index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric='cosine',
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )
        print(f"Index '{INDEX_NAME}' created successfully.")
        return True
    except Exception as e:
        print(f"Failed to create index: {e}")
        return False

def process_pdf(file_path, text_splitter, embeddings, index):
    """Process a single PDF file and upsert its data into Pinecone."""
    try:
        print(f"Processing {file_path}...")

        # Load PDF (pass the file_path to PyPDFLoader)
        loader = PyPDFLoader(file_path)  # Pass the file_path to the loader
        print("Loading PDF pages...")
        pages = loader.load()

        # Split text into chunks
        print("Splitting text into chunks...")
        texts = text_splitter.split_documents(pages)
        print(f"Created {len(texts)} chunks.")

        # Process chunks in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {(len(texts) + batch_size - 1) // batch_size}.")

            # Create embeddings
            contents = [text.page_content for text in batch]
            embeddings_batch = embeddings.embed_documents(contents)

            # Prepare vectors
            vectors = [
                {
                    "id": f"{os.path.basename(file_path)}-chunk-{i + j}",
                    "values": embedding,
                    "metadata": {
                        "text": text.page_content,
                        "source": os.path.basename(file_path),
                        "page": text.metadata.get("page", 0)
                    }
                }
                for j, (text, embedding) in enumerate(zip(batch, embeddings_batch))
            ]

            # Upsert vectors to Pinecone
            index.upsert(vectors=vectors)
            print(f"Uploaded {len(vectors)} vectors to Pinecone.")

        print(f"Finished processing {file_path}.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def load_pdfs_to_chroma(pdf_directory):
    """Load PDF files and upload their content to Pinecone."""
    print(f"Starting to process documents from {pdf_directory}.")

    # Initialize Pinecone
    pc = initialize_pinecone()
    if not pc:
        print("Failed to initialize Pinecone. Exiting.")
        return False

    # Create or recreate index
    if not create_or_recreate_index(pc):
        print("Failed to create index. Exiting.")
        return False

    # Wait for index readiness
    print("Waiting for index to be ready...")
    time.sleep(5)

    # Connect to index
    print("Connecting to Pinecone index...")
    index = pc.Index(INDEX_NAME)

    # Initialize tools
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = OpenAIEmbeddings()

    # Process each PDF
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files to process.")

    for file_path in pdf_files:
        process_pdf(file_path, text_splitter, embeddings, index)

    print("\nAll documents processed successfully!")
    return True

if __name__ == "__main__":
    pdf_dir = "./uploads"
    load_pdfs_to_chroma(pdf_dir)

# Import the modules
import os
import argparse
from dotenv import load_dotenv

# Project Gutenberg
from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import get_text_by_id

# LangChain & Vector Store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQAWithSourcesChain

# Supabase
from supabase import create_client, Client
from supabase.client import ClientOptions

# Constants
COOKING_KEYWORDS = ["cooking", "recipes", "cookbook", "culinary"]

# Function to pull the top 10 Gutenberg search matches by default
def search_gutenberg_titles(cache, keywords, top_n=10, start_date=None, end_date=None):
    """
    Search Project Gutenberg for cooking-related books, optionally filtered by date.
    Returns: List of (gutenbergbookid, title).
    """
    matching_books = []
    keyword_filters = " OR ".join([f"s.name LIKE '%{kw}" for kw in keywords])
    
    date_filter = ""
    if start_date and end_date:
        date_filter = f"AND b.dateissued BETWEEN '{start_date}' AND '{end_date}'"
    elif start_date:
        date_filter = f"AND b.dateissued >= '{start_date}'"
    elif end_date:
        date_filter = f"AND b.dateissued <= '{end_date}'"
        
    query = f"""
        SELECT DISTINCT b.gutenbergbookid AS gutenbeergbookid, t.name AS title
        FROM books b
        LEFT JOIN titles t ON b.id = t.bookid
        LEFT JOIN book_subjects bs ON b.id = bs.bookid
        LEFT JOIN subjects s ON bs.subjectid = s.id
        WHERE ({keyword_filters}) {date_filter}
        LIMIT {top_n};
    """
    
    results = cache.native_query(query)
    
    for row in results:
        gutenbergbookid, title = row
        matching_books.append((gutenbergbookid, title))
        
    return matching_books

# Function to accept the list of books, split the data into chunks, generate vector embeddings, and store the processed data
def download_and_store_books(matching_books, vector_store):
    """Download books, split text, generate embeddings, and store in Supabase.
    """    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    documents = []
    
    for book_id, title in matching_books:
        print(f"Processing: {title} (ID: {book_id})")
        try:
            # Download book content
            raw_text = get_text_by_id(book_id)
            content = raw_text.decode("utf-8", errors="ignore") # Decode to string
            
            # Split the text into manageable chunks
            chunks = text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                # Construct metadat as a JSON object
                metadata = {
                    "source": title, # Key must be 'source' for LangChain
                    "gutenberg_id": str(book_id),
                    "chunk_index": i,
                    "content_length": len(chunk)
                }
                
                # Create a Document object
                documents.append(Document(page_content=chunk, metadata=metadata))
               
        except Exception as e:
            print(f"Error processing {title}: {e}") 
        
        # Batch insert documents to Supabase
        batch_size = 50 # Adjust as necessary
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                print(f"Successfully upload batch {i // batch_size + 1} " 
                      f"of {len(documents) // batch_size + 1}.")
            except Exception as e:
                print(f"Error storing batch {i // batch_size + 1}: {e}")
        
        ####################
        ########
        # RAG FUNCTIONS
        ########
        ###################
        
def perform_similarity_search(query, vector_store):
    """Perform a similarity search using LangChain."""
    print("Performing similarity search...")
    
    docs = vector_store.similarity_search(query)
    # Wrap each document in an item of the "results" list
    results_list = []
    
    for doc in docs:
        results_list.append({
            "sub_query": query,
            "answer": None, # No LLM answer, just raw search results
            "sources": doc.metadata.get("source") if doc.metadata else None,
            "source_documents": [doc]
        })
    return {
        "method": "similarity search",
        "query": query,
        "results": results_list
    }
        
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from log_config import setup_logging
import uuid
import time
import logging
import os
import re
from datetime import datetime

# Set up logging
logger = setup_logging()

# Define the embedding model name as a constant for consistency across the application
EMBEDDING_MODEL = "mxbai-embed-large"

def prepare_log_for_embedding(log):
    """
    Create a semantically rich representation of a log entry for better embeddings.
    
    Args:
        log (dict): A log entry dictionary
        
    Returns:
        str: Enriched text representation optimized for embedding
    """
    enriched_text = ""
    
    if isinstance(log, dict):  
        # Extract entities from content
        if 'content' in log:
            content = log['content']
            
            # Extract IPs for network context
            ip_matches = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', content)
            if ip_matches:
                enriched_text += f"IP Addresses: {', '.join(ip_matches)} | "
            
            # Extract URLs for web context
            url_matches = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', content)
            if url_matches:
                enriched_text += f"URLs: {', '.join(url_matches)} | "
            
            # Extract numeric error codes
            code_matches = re.findall(r'(?:error|code)[\s:]+(\d+)', content, re.IGNORECASE)
            if code_matches:
                enriched_text += f"Error Codes: {', '.join(code_matches)} | "
            
            # Add the actual log content
            enriched_text += f"Message: {content}"
            
            # Add stack trace if available
            if 'stack_trace' in log and log['stack_trace']:
                stack_trace = "\n".join(log['stack_trace'])
                enriched_text += f"\nStack Trace Summary: {stack_trace[:200]}..."
    else:
        # For non-dict logs, use as is
        enriched_text = str(log)
        
    return enriched_text

def store_logs_in_vector_db(logs, max_retries=3):
    """
    Chunk, embed, and store logs in a VectorDB with retry mechanism.

    Args:
        logs (list): A list of log entries.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        bool: True if successful, False otherwise.
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Processing logs attempt {retry_count + 1}/{max_retries}")
            
            # Initialize the embedding model
            logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
            embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
            
            # Create text splitter for chunking with optimized parameters for logs
            logger.info("Creating text splitter for chunking")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,               # Smaller chunks for more precise retrieval
                chunk_overlap=50,            # Sufficient overlap to maintain context
                length_function=len,
                separators=["\n\n", "\n", " ", ""]  # Prioritize splitting on logical breaks
            )
            
            # Process chunks
            logger.info(f"Creating document chunks from {len(logs)} log entries")
            
            # Extract content from log entries based on their structure
            log_texts = []
            log_metadata_map = {}  # Map to store metadata for each text chunk
            
            if logs:
                for idx, log in enumerate(logs):
                    if isinstance(log, dict):
                        # Create semantically enriched log representation for embedding
                        log_text = prepare_log_for_embedding(log)
                        
                        # Generate a unique ID for this log entry
                        log_id = str(uuid.uuid4())
                        
                        # Store metadata mapped to this log ID
                        log_metadata = {
                            "id": log_id,
                            "index": idx,
                            "source": log.get('source', 'unknown'),
                            "timestamp": log.get('timestamp', ''),
                            "level": log.get('level', ''),
                            "type": "structured_log" 
                        }
                        
                        # Add exception-specific metadata if available
                        if 'exception_type' in log:
                            log_metadata["exception_type"] = log['exception_type']
                            log_metadata["type"] = "exception_log"
                        
                        # Store the content of the log as a string instead of the full object
                        log_metadata["log_content"] = log.get('content', '')
                        
                        # Don't store the entire raw log in metadata as it's a complex object
                        # Instead store a reference or essential parts as primitive types
                        if 'raw' in log:
                            log_metadata["has_raw_data"] = True
                            # Optionally store a limited subset of the raw data
                            # log_metadata["raw_excerpt"] = log['raw'][:100] if isinstance(log['raw'], str) else str(log['raw'])[:100]
                        
                        log_metadata_map[log_text] = log_metadata
                        log_texts.append(log_text)
                    else:
                        # For other formats, convert to string representation
                        log_text = str(log)
                        log_metadata_map[log_text] = {
                            "id": str(uuid.uuid4()),
                            "index": idx,
                            "type": "string_log"
                        }
                        log_texts.append(log_text)
            
            logger.info(f"Prepared {len(log_texts)} log entries for embedding")
            logger.info(f"log_texts:: {log_texts}")
            # Create document chunks with associated metadata
            chunks = text_splitter.create_documents(log_texts)
            
            # Add original log metadata to each chunk
            documents_with_metadata = []
            metadatas = []
            
            for chunk in chunks:
                # Find the original text this chunk came from
                original_text = None
                for text in log_texts:
                    if chunk.page_content in text or text in chunk.page_content:
                        original_text = text
                        break
                
                # Get metadata for this chunk
                metadata = {
                    "chunk_id": str(uuid.uuid4()),
                    "source": "log_analysis",
                }
                
                # Add original log metadata if available
                if original_text and original_text in log_metadata_map:
                    metadata.update(log_metadata_map[original_text])
                
                # Add chunk to list
                documents_with_metadata.append(chunk)
                metadatas.append(metadata)
            
            logger.info(f"Created {len(chunks)} document chunks with metadata")
            
            # Store in Chroma - use batch processing to avoid timeouts
            logger.info("Storing documents in Chroma vector database")
            
            # Set batch size based on number of documents
            batch_size = min(32, max(8, len(documents_with_metadata) // 10))
            logger.info(f"Using batch size of {batch_size} for {len(documents_with_metadata)} documents")
            
            try:
                # First check if database directory exists and create it if needed
                db_dir = "./chroma_db"
                if not os.path.exists(db_dir):
                    os.makedirs(db_dir)
                    logger.info(f"Created vector database directory: {db_dir}")
                
                # Initialize the database
                vectordb = Chroma(
                    persist_directory=db_dir,
                    embedding_function=embedding_model,
                    collection_name="log_collection"
                )
                
                # Process in batches to avoid timeouts
                for i in range(0, len(documents_with_metadata), batch_size):
                    batch_end = min(i + batch_size, len(documents_with_metadata))
                    batch_docs = documents_with_metadata[i:batch_end]
                    batch_metadata = metadatas[i:batch_end]
                    
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents_with_metadata) + batch_size - 1)//batch_size}: " +
                               f"documents {i+1}-{batch_end}")
                    
                    # Add timeout handling
                    try:
                        # Fix: Use the Chroma collection API directly for batch processing too
                        batch_ids = [str(uuid.uuid4()) for _ in range(len(batch_docs))]
                        batch_embeddings = []
                        batch_contents = []
                        
                        # Generate embeddings for all documents in the batch
                        logger.info(f"Generating embeddings for batch {i//batch_size + 1}")
                        for doc in batch_docs:
                            try:
                                embedding = embedding_model.embed_query(doc.page_content)
                                batch_embeddings.append(embedding)
                                batch_contents.append(doc.page_content)
                            except Exception as embed_error:
                                logger.error(f"Error generating embedding: {str(embed_error)}")
                                # Use a zero vector as fallback - this will be filtered out in search results
                                batch_embeddings.append([0.0] * 384)  # Standard embedding dimension
                                batch_contents.append(doc.page_content)
                        
                        # Add all documents in one batch operation
                        vectordb._collection.add(
                            ids=batch_ids,
                            embeddings=batch_embeddings,
                            metadatas=batch_metadata,
                            documents=batch_contents
                        )
                        logger.info(f"Successfully added batch {i//batch_size + 1} with metadata")
                    except Exception as batch_error:
                        logger.error(f"Error adding batch {i//batch_size + 1}: {str(batch_error)}")
                        # If a batch fails, try adding documents one by one with more explicit parameters
                        logger.info("Trying to add documents one by one")
                        for j, doc in enumerate(batch_docs):
                            try:
                                metadata = batch_metadata[j] if j < len(batch_metadata) else {"source": "log_analysis", "error_recovery": "true"}
                                # Fix: Use the Chroma collection API directly to avoid parameter conflicts
                                doc_id = str(uuid.uuid4())
                                embedding = embedding_model.embed_query(doc.page_content)
                                vectordb._collection.add(
                                    ids=[doc_id],
                                    embeddings=[embedding],
                                    metadatas=[metadata],
                                    documents=[doc.page_content]
                                )
                                logger.info(f"Added document {i+j+1}/{len(documents_with_metadata)}")
                            except Exception as doc_error:
                                logger.error(f"Failed to add document {i+j+1}: {str(doc_error)}")
                
                # Log database stats
                count = vectordb._collection.count()
                db_size = os.path.getsize("./chroma_db/chroma.sqlite3") / 1024 if os.path.exists("./chroma_db/chroma.sqlite3") else 0
                logger.info(f"Vector database contains {count} documents, size: {db_size:.2f} KB")
                
                logger.info("Successfully stored logs in vector database")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize or access Chroma DB: {str(e)}")
                raise e
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error processing logs: {str(e)}")
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to process logs after {max_retries} attempts")
                return False
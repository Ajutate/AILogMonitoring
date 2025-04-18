from vector_storage import store_logs_in_vector_db
from log_extraction import extract_logs
from query_processing import process_query
from log_config import setup_logging
import streamlit as st
import logging
import os
import json

# Set up logging
logger = setup_logging()

def main():
    st.title("AI Log Monitoring Application")
    
    logger.info("Application started")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        max_retries = st.slider("Max Retries", min_value=1, max_value=5, value=3)
        ollama_model = st.selectbox(
            "Ollama Model", 
            options=["llama3", "mistral", "gemma3", "llama3.2", "nomic-embed-text", "mxbai-embed-large"],
            index=0
        )
        k_documents = st.slider("Number of similar documents", min_value=1, max_value=600, value=5)
        
        # Create database directory if it doesn't exist
        if not os.path.exists("./chroma_db"):
            os.makedirs("./chroma_db")
            logger.info("Created vector database directory")
            st.info("Created vector database directory")
            
        # Add a link to view logs
        if os.path.exists("./logs/app.log"):
            log_size = os.path.getsize("./logs/app.log") / 1024  # Size in KB
            st.info(f"Application log size: {log_size:.2f} KB")
            if st.button("View Application Logs"):
                try:
                    with open("./logs/app.log", "r") as f:
                        log_content = f.read()
                    st.text_area("Application Logs", log_content, height=300)
                except Exception as e:
                    st.error(f"Error reading log file: {str(e)}")

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Log Extraction", 
        "Query Processing", 
        "Log Viewer", 
        "Traditional Log Analysis",
        "Vector DB Status",
        "Clear Vector DB",
        "Help"
    ])
    
    # Log extraction and storage tab
    with tab1:
        st.header("Extract and Process Logs")
        log_path = st.text_input("Enter the log directory or file path:")
        
        if st.button("Extract and Store Logs"):
            logger.info(f"Extracting logs from path: {log_path}")
            with st.spinner("Extracting logs..."):
                logs = extract_logs(log_path, max_retries)
                if not logs:
                    # Check if path is a file or directory for better error message
                    if os.path.isfile(log_path):
                        error_msg = f"Could not extract any logs from file: {log_path}"
                    else:
                        error_msg = f"No log files found in {log_path}"
                    logger.error(error_msg)
                    st.error(error_msg)
                else:
                    logger.info(f"Successfully extracted {len(logs)} log entries")
                    st.info(f"Extracted {len(logs)} log entries")
                    
                    with st.spinner("Processing and storing logs in vector database..."):
                        logger.info("Starting vector database storage process")
                        success = store_logs_in_vector_db(logs, max_retries)
                        
                        if success:
                            logger.info("Successfully stored logs in vector database")
                            st.success("Logs have been successfully extracted and stored in the VectorDB.")
                            
                            # Save sample logs for inspection
                            try:
                                with open("log_samples.json", "w") as f:
                                    json.dump(logs[:10], f, indent=2)
                                logger.info("Saved sample logs to log_samples.json")
                                st.info("Saved sample logs for inspection")
                            except Exception as e:
                                error_msg = f"Couldn't save log samples: {str(e)}"
                                logger.error(error_msg)
                                st.warning(error_msg)
                        else:
                            error_msg = "Failed to process and store logs. Check the application logs for details."
                            logger.error("Vector database storage process failed")
                            st.error(error_msg)

    # Query processing tab (converted to chatbot)
    with tab2:
        st.header("Log Analysis Chatbot")
        st.write("Chat with your logs using natural language. The chatbot will analyze your logs and answer your questions.")
        
        # Chat settings (moved up before first use)
        with st.sidebar:
            st.subheader("Chat Settings")
            show_context = st.checkbox("Show source logs in responses", value=False)
            
            if st.button("Clear chat history"):
                if "messages" in st.session_state:
                    st.session_state.messages = [
                        {"role": "assistant", "content": "Chat history has been cleared. How can I help you with your logs?"}
                    ]
                    st.rerun()
        
        # Initialize session state for chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your log analysis assistant. Ask me questions about your logs and I'll help you analyze them."}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask a question about your logs..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                logger.info(f"Processing chatbot query: {prompt[:50]}...")
                
                # Prepare conversation history for context
                conversation_history = ""
                if len(st.session_state.messages) > 2:
                    # Get last few exchanges to provide context (limit to last 6 messages for relevance)
                    recent_messages = st.session_state.messages[-7:-1]  # Exclude current user message
                    for msg in recent_messages:
                        conversation_history += f"{msg['role'].upper()}: {msg['content']}\n"
                
                # Enhance the query with conversation history
                enhanced_query = prompt
                if conversation_history:
                    enhanced_query = f"""Conversation history:
{conversation_history}

User's current query: {prompt}

Based on the conversation history and current query, respond to the user's latest question."""
                
                with st.spinner(""):
                    try:
                        # Get response from query processing
                        result = process_query(
                            prompt,#enhanced_query, 
                            model_name=ollama_model, 
                            max_retries=max_retries,
                            k=k_documents
                        )
                        
                        if result["error"]:
                            error_msg = f"Sorry, I encountered an error: {result['error']}"
                            logger.error(f"Chatbot query processing error: {result['error']}")
                            message_placeholder.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        else:
                            logger.info("Chatbot query processed successfully")
                            response = result["response"]
                            logger.info("Response from chatbot: %s", response[:50])
                            message_placeholder.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Show context if requested
                            if show_context and result["documents"]:
                                with st.expander("View source log entries"):
                                    logger.info(f"Displaying {len(result['documents'])} context documents")
                                    for i, doc in enumerate(result["documents"]):
                                        st.text(f"Log Entry {i+1}:\n{doc}")
                    except Exception as e:
                        error_msg = f"Sorry, something went wrong: {str(e)}"
                        logger.error(f"Unexpected error in chatbot: {str(e)}")
                        message_placeholder.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Log viewer tab
    with tab3:
        st.header("Log Viewer")
        
        try:
            if os.path.exists("log_samples.json"):
                with open("log_samples.json", "r") as f:
                    sample_logs = json.load(f)
                    
                logger.info(f"Displaying {len(sample_logs)} sample logs")
                for i, log in enumerate(sample_logs):
                    with st.expander(f"Log Entry {i+1}"):
                        st.json(log)
            else:
                logger.info("No log samples available")
                st.info("No log samples available. Extract logs first.")
        except Exception as e:
            error_msg = f"Error loading log samples: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
    
    # Traditional Log Analysis tab (new)
    with tab4:
        st.header("Traditional Log Analysis")
        st.write("Process traditional log files with stack traces and exceptions.")
        
        # Locate log files in the workspace
        default_log_files = []
        for log_file in ["traditional.log", "traditional2.log"]:
            if os.path.exists(log_file):
                default_log_files.append(log_file)
        
        # Create a dropdown to select from available log files or enter a custom path
        log_file_options = ["Custom path..."] + default_log_files
        selected_option = st.selectbox(
            "Select log file:", 
            options=log_file_options,
            index=1 if default_log_files else 0
        )
        
        # Text input for custom path or display selected file
        if selected_option == "Custom path...":
            traditional_log_path = st.text_input("Enter path to log file:")
        else:
            traditional_log_path = selected_option
            st.info(f"Selected log file: {traditional_log_path}")
        
        if st.button("Analyze Traditional Log"):
            if not traditional_log_path or not os.path.exists(traditional_log_path):
                st.error(f"Log file not found: {traditional_log_path}")
            else:
                with st.spinner("Analyzing log file..."):
                    logger.info(f"Analyzing traditional log file: {traditional_log_path}")
                    logs = extract_logs(traditional_log_path, max_retries)
                    
                    if logs:
                        logger.info(f"Extracted {len(logs)} entries from traditional log")
                        st.success(f"Successfully extracted {len(logs)} log entries")
                        
                        # Show log analytics
                        levels = {}
                        exceptions = []
                        errors = []
                        
                        for log in logs:
                            level = log.get('level', 'UNKNOWN')
                            if level in levels:
                                levels[level] += 1
                            else:
                                levels[level] = 1
                                
                            if level == 'EXCEPTION' or 'exception_type' in log:
                                exceptions.append(log)
                            elif level == 'ERROR':
                                errors.append(log)
                        
                        # Display summary
                        st.subheader("Log Summary")
                        st.write(f"Total logs: {len(logs)}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Log level distribution:")
                            for level, count in levels.items():
                                st.write(f"- {level}: {count}")
                        
                        with col2:
                            st.write(f"Exceptions: {len(exceptions)}")
                            st.write(f"Errors: {len(errors)}")
                        
                        # Display exceptions
                        if exceptions:
                            st.subheader("Exceptions")
                            for ex in exceptions:
                                with st.expander(f"{ex.get('timestamp', 'Unknown')} - {ex.get('exception_type', 'Exception')}"):
                                    st.json(ex)
                        
                        # Store in vector DB if requested
                        if st.button("Store Traditional Logs in Vector DB"):
                            with st.spinner("Storing logs in vector database..."):
                                logger.info(f"Starting to store {len(logs)} traditional logs in vector database")
                                
                                # Store the logs in the vector DB
                                success = store_logs_in_vector_db(logs, max_retries)
                                
                                if success:
                                    # Verify data was actually stored
                                    try:
                                        from langchain_community.vectorstores import Chroma
                                        from langchain_community.embeddings import OllamaEmbeddings
                                        
                                        logger.info("Verifying logs were stored in vector database")
                                        embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
                                        vectordb = Chroma(
                                            collection_name="log_collection",
                                            embedding_function=embedding_model,
                                            persist_directory="./chroma_db"
                                        )
                                        
                                        # Get collection stats before and after
                                        collection = vectordb._collection
                                        count = collection.count()
                                        
                                        logger.info(f"Vector DB now contains {count} documents")
                                        st.success(f"Traditional logs stored in vector database for querying ({count} total documents)")
                                        
                                        # Add detailed log for troubleshooting
                                        db_size = 0
                                        if os.path.exists("./chroma_db/chroma.sqlite3"):
                                            db_size = os.path.getsize("./chroma_db/chroma.sqlite3") / 1024
                                            logger.info(f"Vector DB file size: {db_size:.2f} KB")
                                        
                                    except Exception as e:
                                        logger.error(f"Error verifying vector DB: {str(e)}")
                                        st.success("Traditional logs stored in vector database for querying")
                                else:
                                    logger.error("Failed to store traditional logs in vector database")
                                    st.error("Failed to store logs in vector database")
                    else:
                        st.error(f"No log entries could be extracted from the file. Check if it has the correct log format.")

    # Vector DB Status tab (new)
    with tab5:
        st.header("Vector Database Status")
        st.write("Check if data is present in the vector database and view statistics.")
        
        if st.button("Check Vector DB Status"):
            with st.spinner("Checking vector database..."):
                try:
                    from langchain_community.vectorstores import Chroma
                    from langchain_community.embeddings import OllamaEmbeddings
                    
                    # Check if vector DB directory exists
                    if not os.path.exists("./chroma_db"):
                        st.error("Vector database directory does not exist. No data has been stored yet.")
                    else:
                        # Try to load the database
                        try:
                            embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
                            vectordb = Chroma(
                                collection_name="log_collection",
                                embedding_function=embedding_model,
                                persist_directory="./chroma_db"
                            )
                            
                            # Get collection stats
                            collection = vectordb._collection
                            count = collection.count()
                            
                            if count > 0:
                                st.success(f"✅ Vector database is operational and contains data.")
                                
                                # Display statistics
                                st.subheader("Database Statistics")
                                st.metric("Total Documents", count)
                                
                                # Get a sample of documents
                                sample_size = min(5, count)
                                st.subheader(f"Sample Documents (showing {sample_size} of {count})")
                                
                                # Get a sample of IDs
                                sample_ids = collection.get(limit=sample_size)["ids"]
                                
                                for i, doc_id in enumerate(sample_ids):
                                    # Get the document by ID
                                    doc = collection.get(ids=[doc_id])
                                    with st.expander(f"Document {i+1} (ID: {doc_id[:8]}...)"):
                                        if "metadatas" in doc and doc["metadatas"] and doc["metadatas"][0]:
                                            st.json(doc["metadatas"][0])
                                        else:
                                            st.write("No metadata available for this document")
                            else:
                                st.warning("Vector database exists but contains no documents. Try extracting and storing logs first.")
                        except Exception as e:
                            st.error(f"Error accessing vector database: {str(e)}")
                            logger.error(f"Error checking vector DB status: {str(e)}")
                except ImportError as e:
                    st.error(f"Required libraries not installed: {str(e)}")
                    logger.error(f"Import error when checking vector DB status: {str(e)}")

    # Clear Vector DB tab (new)
    with tab6:
        st.header("Clear Vector Database")
        st.write("Use this tab to clear the vector database when you want to start fresh or reset the system.")
        
        st.warning("⚠️ **Warning**: Clearing the vector database will remove all stored log data. This action cannot be undone.")
        
        confirm_clear = st.checkbox("I understand that this action will delete all vector database data")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Clear Vector DB", disabled=not confirm_clear):
                with st.spinner("Clearing vector database..."):
                    try:
                        from langchain_community.vectorstores import Chroma
                        from langchain_community.embeddings import OllamaEmbeddings
                        from vector_storage import EMBEDDING_MODEL
                        import shutil
                        import gc
                        import time
                        import platform
                        import tempfile
                        
                        logger.warning("User initiated vector database clearing")
                        
                        # Check if the directory exists
                        if os.path.exists("./chroma_db"):
                            # First try to delete through the API
                            try:
                                # Initialize the embedding model
                                embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
                                
                                # Connect to the database
                                vectordb = Chroma(
                                    collection_name="log_collection",
                                    embedding_function=embedding_model,
                                    persist_directory="./chroma_db"
                                )
                                
                                # Get collection and delete it
                                collection = vectordb._collection
                                count_before = collection.count()
                                collection.delete(where={})  # Delete all documents
                                logger.info(f"Deleted {count_before} documents from vector database via API")
                                
                                # Force persist changes
                                vectordb.persist()
                                logger.info("Vector database changes persisted")
                                
                                # Verify deletion
                                count_after = collection.count()
                                if count_after == 0:
                                    st.success(f"Successfully cleared vector database. Removed {count_before} documents.")
                                else:
                                    # If API deletion didn't work completely, try file system deletion
                                    raise Exception(f"API deletion incomplete. {count_after} documents remain.")
                                    
                            except Exception as api_error:
                                logger.error(f"Error using API to clear vector DB: {str(api_error)}")
                                logger.info("Attempting file system deletion as fallback")
                                
                                # Fallback: Remove the directory physically
                                try:
                                    # First ensure all connections are closed
                                    vectordb = None
                                    collection = None
                                    embedding_model = None
                                    
                                    # Force Python garbage collection to release file handles
                                    gc.collect()
                                    
                                    # Wait a moment for resources to be released
                                    time.sleep(1)
                                    
                                    # On Windows, we need special handling for locked files
                                    if platform.system() == "Windows":
                                        logger.info("Windows system detected, using safe directory removal")
                                        
                                        try:
                                            # Use absolute paths to avoid permission issues
                                            current_dir = os.path.abspath(os.curdir)
                                            db_dir_abs = os.path.join(current_dir, "chroma_db")
                                            backup_dir_abs = os.path.join(current_dir, f"chroma_db_backup_{int(time.time())}")
                                            
                                            logger.info(f"Attempting to rename {db_dir_abs} to {backup_dir_abs}")
                                            
                                            # Try to close any potential file handles more aggressively
                                            # Make psutil usage optional with fallback
                                            try:
                                                import psutil
                                                process = psutil.Process(os.getpid())
                                                for handler in process.open_files():
                                                    if "chroma_db" in handler.path:
                                                        logger.warning(f"Found open file handle: {handler.path}")
                                            except ImportError:
                                                logger.info("psutil module not available, skipping file handle detection")
                                                # Since we can't check file handles, wait a bit longer as fallback
                                                time.sleep(3)
                                            
                                            # Wait a bit longer
                                            time.sleep(2)
                                            
                                            # Alternative approach: Instead of renaming, create a new directory and 
                                            # just warn the user about the old one
                                            if os.path.exists(db_dir_abs):
                                                # First try to remove individual files if possible
                                                try:
                                                    for root, dirs, files in os.walk(db_dir_abs):
                                                        for file in files:
                                                            try:
                                                                file_path = os.path.join(root, file)
                                                                if os.path.exists(file_path):
                                                                    os.chmod(file_path, 0o666)  # Make writeable
                                                                    os.remove(file_path)
                                                                    logger.info(f"Removed file: {file_path}")
                                                            except Exception as file_err:
                                                                logger.warning(f"Could not remove file {file}: {str(file_err)}")
                                                except Exception as walk_err:
                                                    logger.warning(f"Error walking directory: {str(walk_err)}")
                                                
                                                # Try to create a new database directory with a different name
                                                new_db_dir = os.path.join(current_dir, "chroma_db_new")
                                                if os.path.exists(new_db_dir):
                                                    try:
                                                        shutil.rmtree(new_db_dir)
                                                    except:
                                                        pass
                                                
                                                os.makedirs(new_db_dir, exist_ok=True)
                                                
                                                # Update the user
                                                logger.info(f"Created new database directory: {new_db_dir}")
                                                st.warning(f"""
                                                Could not clear the original database directory due to Windows file locks.
                                                
                                                A new database directory has been created at '{new_db_dir}'.
                                                
                                                **Action required:** To use the new database, please:
                                                1. Restart the application
                                                2. Manually rename '{new_db_dir}' to 'chroma_db'
                                                3. Delete the old 'chroma_db' directory when the application is not running
                                                """)
                                                
                                        except Exception as win_error:
                                            logger.error(f"Windows-specific handling error: {str(win_error)}")
                                            st.error(f"""
                                            Access denied when trying to clear the database. This is often due to Windows file locks.
                                            
                                            Please try:
                                            1. Close any other applications that might be using the database files
                                            2. Restart this application
                                            3. If problems persist, manually delete the ./chroma_db folder when no applications are running
                                            """)
                                    else:
                                        # On non-Windows systems, we can use shutil.rmtree directly
                                        shutil.rmtree("./chroma_db")
                                        os.makedirs("./chroma_db")  # Recreate empty directory
                                    
                                    logger.info("Vector database directory handled appropriately")
                                    st.success("Successfully cleared vector database by creating a new empty database.")
                                except Exception as fs_error:
                                    logger.error(f"File system error: {str(fs_error)}")
                                    st.error(f"""Failed to clear vector database files: {str(fs_error)}
                                    
Please try the following:
1. Close any other applications that might be using the database
2. Restart the application
3. Try clearing the database again
4. If all else fails, manually delete the ./chroma_db folder when the application is not running""")
                        else:
                            logger.info("Vector database directory does not exist, nothing to clear")
                            st.info("Vector database does not exist yet. Nothing to clear.")
                            
                    except Exception as e:
                        error_msg = f"Failed to clear vector database: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)
        
        with col2:
            st.info("""
            **When to clear the vector database:**
            
            - When you want to start with fresh data
            - If you've stored incorrect or test data
            - To reclaim disk space
            - If you're experiencing inconsistent query results
            
            After clearing, you'll need to extract and store logs again.
            """)

    # Help tab
    with tab7:
        st.header("Help & Documentation")
        st.markdown("""
        ### How to use this application
        
        This AI-powered log monitoring application helps you analyze and query log files using natural language.
        
        #### Steps to use:
        1. **Extract Logs**: Navigate to the Log Extraction tab, enter the directory path containing your log files, and click "Extract and Store Logs".
        2. **Query Logs**: Navigate to the Query Processing tab, enter your natural language query about the logs, and click "Submit Query".
        3. **View Logs**: Navigate to the Log Viewer tab to see sample logs that have been extracted.
        4. **Traditional Log Analysis**: Process traditional log files with exception stack traces and get insights.
        
        #### Example queries:
        - Show me all error logs from the past 24 hours
        - Find logs with high CPU usage
        - Identify unusual patterns in authentication logs
        - Summarize log activity from the application server
        - List all exceptions with NullPointerException
        
        #### Log Format Support:
        - Single line logs: `2025-01-01 00:02:00 [ERROR] Database connection failed.`
        - Multi-line exception logs with stack traces
        """)
        logger.info("Help documentation viewed")

if __name__ == "__main__":
    main()
    logger.info("Application terminated")
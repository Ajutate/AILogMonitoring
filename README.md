# AI Log Monitoring Application

This application provides AI-powered log monitoring capabilities using LangChain, Ollama, and ChromaDB (vector database). It extracts logs, processes them, and allows natural language querying through a Streamlit user interface.

## Features

- **Log Extraction**: Extract logs from files with automatic error recovery
- **Vector Storage**: Chunk and embed logs using Ollama's embedding models
- **Natural Language Querying**: Process user queries with LangChain and Ollama models
- **User Interface**: Intuitive Streamlit interface with multiple tabs for different functions

## Prerequisites

- Python 3.9+ installed
- Ollama installed and running locally (see [Ollama installation](https://ollama.ai/))
- Required Ollama models:
  - llama3 (or another LLM model of your choice)
  - nomic-embed-text/mxbai-embed-large (for embeddings)

## Installation

1. Activate the virtual environment:

```bash
# On Windows
.\venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

2. Install the required packages:

```bash
# Install from requirements.txt file
pip install -r requirements.txt

# Or install packages individually
pip install langchain langchain-core langchain-community langchain-text-splitters chromadb streamlit ollama python-dotenv
```

3. Verify installation:

```bash
# Check that LangChain packages are installed correctly
pip list | findstr langchain
```

## Configuration

Ensure Ollama is running on your system. By default, the application will attempt to connect to Ollama at http://localhost:11434.

## Running the Application

Run the Streamlit application with:

```bash
streamlit run main.py
```

This will start the web server and open the application in your default browser.

## Usage

1. **Extract and Store Logs**:
   - Navigate to the "Log Extraction" tab
   - Enter the path to your log directory
   - Click "Extract and Store Logs"

2. **Query Logs**:
   - Navigate to the "Query Processing" tab
   - Enter your natural language query
   - Click "Submit Query"
   - Optionally check "Show retrieved log context" to see the log entries used for the response

3. **View Sample Logs**:
   - Navigate to the "Log Viewer" tab to see samples of extracted logs

## Customization

- Modify the retry mechanisms by adjusting the `max_retries` parameter
- Change the Ollama models in the UI dropdown
- Adjust the number of similar documents to retrieve for context

## Troubleshooting

- If you encounter Ollama connection issues, ensure the Ollama service is running
- Check the application logs for detailed error messages
- Verify that your log files end with the `.log` extension
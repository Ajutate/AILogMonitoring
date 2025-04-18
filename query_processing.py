from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta
import re
import json
from log_config import setup_logging
import time
import os
import logging
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # Updated import
from langchain_core.runnables import RunnablePassthrough  # For creating chain

# Set up logging
logger = setup_logging()

# Define constants for models
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"

def detect_filter_operator_from_query(query):
    """
    Detect whether the query implies OR or AND logic for filters.
    Returns "$or" if OR logic is detected, otherwise "$and".
    """
    or_keywords = [
        r"\bor\b",
        r"\beither\b",
        r"\bany of\b",
        r"\bone of\b",
        r"\bat least one\b",
        r"\bany\b"
    ]
    for pattern in or_keywords:
        if re.search(pattern, query, re.IGNORECASE):
            return "$or"
    return "$and"

def fetch_relevant_logs(query, top_k=10, filters=None, verbose=True, filter_operator=None):
    """
    Search vector DB for logs relevant to the query.
    
    Args:
        query (str): User query about logs
        top_k (int): Number of relevant logs to return
        filters (dict): Metadata filters for the search
        verbose (bool): Whether to print verbose output
        filter_operator (str): Operator for combining filter clauses ("$and" or "$or")
        
    Returns:
        list: List of relevant log entries
    """
    try:
        logger.info("Inside fetch_relevant_logs method")
        logger.info(f"Searching for logs relevant to: {query}")
        
        # Enhance query with semantic context
        #enhanced_query = enhance_user_query(query)
        enhanced_query = query
        if verbose:
            logger.info(f"Enhanced query: {enhanced_query}")
        
        # Initialize embedding model
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Initialize DB connection
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            logger.error(f"Vector database not found at {db_path}")
            return []
            
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model,
            collection_name="log_collection"
        )
        
        # Apply metadata filtering if provided
        filter_dict = {}
        if filters:
            filter_clauses = []
            # Process time range filters
            if 'time_range' in filters:
                time_range = filters['time_range']
                if time_range == 'last_hour':
                    start_time = (datetime.now() - timedelta(hours=1)).isoformat()
                    filter_clauses.append({"timestamp": {"$gte": start_time}})
                elif time_range == 'last_day':
                    start_time = (datetime.now() - timedelta(days=1)).isoformat()
                    filter_clauses.append({"timestamp": {"$gte": start_time}})
                elif time_range == 'last_week':
                    start_time = (datetime.now() - timedelta(weeks=1)).isoformat()
                    filter_clauses.append({"timestamp": {"$gte": start_time}})
            # Process severity level filters
            if 'level' in filters:
                levels = filters['level'] if isinstance(filters['level'], list) else [filters['level']]
                filter_clauses.append({"level": {"$in": levels}})
            # Process source filters
            if 'source' in filters:
                sources = filters['source'] if isinstance(filters['source'], list) else [filters['source']]
                filter_clauses.append({"source": {"$in": sources}})
            # Process type filters
            if 'type' in filters:
                types = filters['type'] if isinstance(filters['type'], list) else [filters['type']]
                filter_clauses.append({"type": {"$in": types}})
            # Determine filter operator automatically if not set
            if filter_operator is None:
                filter_operator = detect_filter_operator_from_query(query)
            # Compose final filter_dict
            if len(filter_clauses) == 1:
                filter_dict = filter_clauses[0]
            elif len(filter_clauses) > 1:
                filter_dict = {filter_operator: filter_clauses}
            logger.info(f"Applying filters: {filter_dict}")
            logger.info(f"Total Documents top_k: {top_k}")
            logger.info(f"enhanced_query: {enhanced_query}")
            filter_dict = None
        # Execute the similarity search with metadata filter
        try:
        
            docs = vectordb.similarity_search_with_score(
                    query,
                    k=top_k
                )
                
            # Process results
            results = []
            # Strict post-filtering by timestamp if time_range filter is present
            time_filter = None
            if filters and 'time_range' in filters:
                now = datetime.now()
                if filters['time_range'] == 'last_hour':
                    time_filter = now - timedelta(hours=1)
                elif filters['time_range'] == 'last_day':
                    time_filter = now - timedelta(days=1)
                elif filters['time_range'] == 'last_week':
                    time_filter = now - timedelta(weeks=1)
            for doc, score in docs:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                ts = metadata.get('timestamp')
                include = True
                if time_filter and ts:
                    try:
                        ts_dt = datetime.fromisoformat(ts)
                        if ts_dt < time_filter:
                            include = False
                    except Exception:
                        pass  # If timestamp is invalid, skip filtering
                if include:
                    result = {
                        "content": doc.page_content,
                        "similarity_score": float(score),
                        "metadata": metadata
                    }
                    results.append(result)
            logger.info(f"Found {len(results)} relevant logs")
            logger.info(f"Found logs from vector:: {results}")
            logger.info(f"Found logs end")
            return results
            
        except Exception as search_error:
            logger.error(f"Error during similarity search: {str(search_error)}")
            
            # Fallback to basic search without filters
            if filter_dict:
                logger.info("Attempting fallback search without filters")
                docs = vectordb.similarity_search_with_score(
                    enhanced_query,
                    k=top_k
                )
                
                # Process results
                results = []
                for doc, score in docs:
                    result = {
                        "content": doc.page_content,
                        "similarity_score": float(score),
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                    }
                    results.append(result)
                    
                logger.info(f"Fallback search found {len(results)} relevant logs")
                return results
            else:
                raise search_error
                
    except Exception as e:
        logger.error(f"Error fetching relevant logs: {str(e)}")
        return []

def enhance_user_query(query):
    """
    Enhance user query with log-related terminology and context.
    
    Args:
        query (str): Original user query
        
    Returns:
        str: Enhanced query optimized for vector search
    """
    logger.info("Inside enhance_user_query method")
    # 1. Extract potential log level mentions
    log_levels = ["error", "warning", "warn", "info", "debug", "critical", "fatal", "exception"]
    level_pattern = r'\b(' + '|'.join(log_levels) + r')s?\b'
    level_matches = re.findall(level_pattern, query.lower())
    
    # 2. Extract potential time references
    time_patterns = [
        (r'\btoday\b', "recent logs from today"),
        (r'\byesterday\b', "logs from yesterday"),
        (r'\blast (hour|day|week|month)\b', "logs from the last \\1"),
        (r'\b\d+ (hour|day|week|month)s? ago\b', "logs from that time period")
    ]
    
    time_context = None
    for pattern, replacement in time_patterns:
        if re.search(pattern, query.lower()):
            time_context = replacement
            break
    
    # 3. Extract potential error mentions
    error_patterns = [
        r'exception',
        r'error',
        r'fail(ed|ure)?',
        r'crash(ed)?',
        r'bug',
        r'issue',
        r'problem'
    ]
    is_error_query = any(re.search(pattern, query.lower()) for pattern in error_patterns)
    
    # 4. Extract IP addresses, URLs, or specific identifiers
    ip_matches = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', query)
    url_matches = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', query)
    
    # 5. Create enhanced query
    enhanced_parts = [query.strip()]
    
    # Add level context
    if level_matches:
        normalized_levels = [level.upper() for level in level_matches]
        enhanced_parts.append(f"Log entries with severity levels: {', '.join(normalized_levels)}")
    
    # Add time context
    if time_context:
        enhanced_parts.append(time_context)
    
    # Add error context
    if is_error_query:
        enhanced_parts.append("Logs showing errors, exceptions, failures or critical issues")
    
    # Add specific identifiers
    if ip_matches:
        enhanced_parts.append(f"Logs containing IP addresses: {', '.join(ip_matches)}")
    if url_matches:
        enhanced_parts.append(f"Logs containing URLs: {', '.join(url_matches)}")
    
    # Combine into enhanced query
    enhanced_query = ". ".join(enhanced_parts)
    return enhanced_query

def extract_time_filters_from_query(query, now=None):
    """
    Extracts flexible time filters from a natural language query.
    Returns a dict suitable for use as a filter, or None if no time filter found.
    """
    if now is None:
        now = datetime.now()
    query = query.lower()
    # Patterns for 'last N unit(s)' or 'past N unit(s)'
    match = re.search(r'(last|past) (\d+) (hour|day|week|month|year)s?', query)
    if match:
        n = int(match.group(2))
        unit = match.group(3)
        if unit == 'hour':
            start = now - timedelta(hours=n)
        elif unit == 'day':
            start = now - timedelta(days=n)
        elif unit == 'week':
            start = now - timedelta(weeks=n)
        elif unit == 'month':
            # Approximate a month as 30 days
            start = now - timedelta(days=30 * n)
        elif unit == 'year':
            start = now.replace(year=now.year - n)
        return {'timestamp': {'$gte': start.isoformat()}}
    # Patterns for 'last month', 'last year', etc.
    if 'last month' in query:
        year = now.year
        month = now.month - 1 if now.month > 1 else 12
        if now.month == 1:
            year -= 1
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        return {'timestamp': {'$gte': start.isoformat(), '$lt': end.isoformat()}}
    if 'last year' in query:
        start = datetime(now.year - 1, 1, 1)
        end = datetime(now.year, 1, 1)
        return {'timestamp': {'$gte': start.isoformat(), '$lt': end.isoformat()}}
    if 'past week' in query:
        start = now - timedelta(weeks=1)
        return {'timestamp': {'$gte': start.isoformat()}}
    if 'past month' in query:
        start = now - timedelta(days=30)
        return {'timestamp': {'$gte': start.isoformat()}}
    if 'past year' in query:
        start = now - timedelta(days=365)
        return {'timestamp': {'$gte': start.isoformat()}}
    # Month + year (e.g., 'January 2025')
    month_map = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    month_year_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec) (\d{4})', query)
    if month_year_match:
        month_str = month_year_match.group(1)
        year = int(month_year_match.group(2))
        month = month_map[month_str]
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        return {'timestamp': {'$gte': start.isoformat(), '$lt': end.isoformat()}}
    # Date range: 'from <date> to <date>'
    date_range_match = re.search(r'from (\w+ \d{1,2},? \d{4}) to (\w+ \d{1,2},? \d{4})', query)
    if date_range_match:
        try:
            start = datetime.strptime(date_range_match.group(1).replace(',', ''), '%B %d %Y')
            end = datetime.strptime(date_range_match.group(2).replace(',', ''), '%B %d %Y') + timedelta(days=1)
            return {'timestamp': {'$gte': start.isoformat(), '$lt': end.isoformat()}}
        except Exception:
            pass
    return None

def extract_filters_from_query(query):
    """
    Extract potential filters from natural language query, including flexible time filters.
    
    Args:
        query (str): User query string
        
    Returns:
        dict: Dictionary of extracted filters
    """
    filters = {}
    
    logger.info("Inside extract_filters_from_query method")
    # Flexible time filter extraction
    time_filter = extract_time_filters_from_query(query, now=datetime(2025, 4, 16))
    if time_filter:
        filters.update(time_filter)
    else:
        # Fallback to legacy time patterns
        time_patterns = [
            (r'\btoday\b', 'last_day'),
            (r'\blast hour\b', 'last_hour'),
            (r'\blast day\b', 'last_day'),
            (r'\blast week\b', 'last_week'),
            (r'\blast month\b', 'last_month'),
            (r'\byesterday\b', 'last_day')
        ]
        
        for pattern, value in time_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                filters['time_range'] = value
                break
    #logger.info(f"time_range: {filters['time_range']}")
    # --- Month name detection and filter ---
    # Extract month names (full and short)
    month_map = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12
    }
    month_regex = r'\\b(' + '|'.join(month_map.keys()) + r')\\b'
    month_match = re.search(month_regex, query, re.IGNORECASE)
    logger.info(f"Month match: {month_match}")

    if month_match and 'timestamp' not in filters:
        month_str = month_match.group(1).lower()
        month_num = month_map[month_str]
        year = datetime(2025, 4, 16).year
        start_of_month = datetime(year, month_num, 1)
        if month_num == 12:
            start_of_next_month = datetime(year + 1, 1, 1)
        else:
            start_of_next_month = datetime(year, month_num + 1, 1)
        filters['timestamp'] = {'$gte': start_of_month.isoformat(), '$lt': start_of_next_month.isoformat()}

    # Extract log levels
    level_patterns = [
        r'\b(error|errors)\b',
        r'\b(warning|warnings|warn|warns)\b',
        r'\b(info|information)\b',
        r'\b(debug)\b',
        r'\b(critical|criticals)\b',
        r'\b(fatal|fatals)\b',
        r'\b(exception|exceptions)\b'
    ]
    
    levels = []
    for pattern in level_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            # Map common variations to standard levels
            level = re.search(pattern, query, re.IGNORECASE).group(1).upper()
            if level in ["WARNS", "WARNING", "WARNINGS"]:
                level = "WARN"
            elif level in ["ERRORS"]:
                level = "ERROR"
            elif level in ["CRITICALS"]:
                level = "CRITICAL"
            elif level in ["FATALS"]:
                level = "FATAL"
            elif level in ["EXCEPTIONS"]:
                level = "EXCEPTION"
            elif level in ["INFORMATION"]:
                level = "INFO"
            
            levels.append(level)
    
    if levels:
        filters['level'] = levels
    
    # Extract sources (filenames)
    source_match = re.search(r'\bfrom\s+([a-zA-Z0-9_\-\.]+\.log)\b', query, re.IGNORECASE)
    if source_match:
        filters['source'] = [source_match.group(1)]
    
    # Extract exception specific filters
    if re.search(r'\bexception|error|stack\s+trace|crash|failure\b', query, re.IGNORECASE):
        filters['type'] = ['exception_log']
    
    # If the query is about error or exception, use OR logic for (level=ERROR or EXCEPTION) or type=exception_log
    if ('level' in filters and any(l in ["ERROR", "EXCEPTION"] for l in filters['level'])) or ('type' in filters and 'exception_log' in filters['type']):
        filters = {'$or': [
            {'level': {'$in': [l for l in filters.get('level', []) if l in ["ERROR", "EXCEPTION"]]}},
            {'type': {'$in': ['exception_log']}}
        ]}
    logger.info(f"Extracted filters: {filters}")
    return filters

def get_log_statistics():
    """
    Get statistics about logs in the vector database.
    
    Returns:
        dict: Statistics about the logs
    """
    logger.info("Inside get_log_statistics method")
    try:
        # Initialize embedding model
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Initialize DB connection
        db_path = "./chroma_db"
        if not os.path.exists(db_path):
            logger.error(f"Vector database not found at {db_path}")
            return {"error": "Vector database not found"}
            
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model,
            collection_name="log_collection"
        )
        
        # Get total count
        total_count = vectordb._collection.count()
        
        # Get level counts
        level_counts = {}
        try:
            levels = vectordb._collection.get(
                where={"level": {"$exists": True}},
                include=["metadatas"]
            )
            if levels and 'metadatas' in levels:
                for metadata in levels['metadatas']:
                    if 'level' in metadata:
                        level = metadata['level']
                        level_counts[level] = level_counts.get(level, 0) + 1
        except Exception as e:
            logger.error(f"Error getting level counts: {str(e)}")
            level_counts = {"unknown": "count query failed"}
        
        # Get source counts
        source_counts = {}
        try:
            sources = vectordb._collection.get(
                where={"source": {"$exists": True}},
                include=["metadatas"]
            )
            if sources and 'metadatas' in sources:
                for metadata in sources['metadatas']:
                    if 'source' in metadata:
                        source = metadata['source']
                        source_counts[source] = source_counts.get(source, 0) + 1
        except Exception as e:
            logger.error(f"Error getting source counts: {str(e)}")
            source_counts = {"unknown": "count query failed"}
        
        # Get timestamps range
        earliest = None
        latest = None
        try:
            timestamps = vectordb._collection.get(
                where={"timestamp": {"$exists": True}},
                include=["metadatas"]
            )
            
            if timestamps and 'metadatas' in timestamps:
                for metadata in timestamps['metadatas']:
                    if 'timestamp' in metadata and metadata['timestamp']:
                        ts = metadata['timestamp']
                        # Convert all timestamps to strings for safe comparison
                        ts_str = str(ts)
                        
                        if not earliest:
                            earliest = ts_str
                        elif isinstance(earliest, str) and isinstance(ts_str, str):
                            # Safe string comparison
                            if ts_str < earliest:
                                earliest = ts_str
                                
                        if not latest:
                            latest = ts_str
                        elif isinstance(latest, str) and isinstance(ts_str, str):
                            # Safe string comparison
                            if ts_str > latest:
                                latest = ts_str
        except Exception as e:
            logger.error(f"Error processing timestamps: {str(e)}")
            earliest = "unknown"
            latest = "unknown"
        
        # Compile statistics
        statistics = {
            "total_log_entries": total_count,
            "level_distribution": level_counts,
            "source_distribution": source_counts,
            "time_range": {
                "earliest": earliest,
                "latest": latest
            },
            "database_path": db_path
        }
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error getting log statistics: {str(e)}")
        return {"error": str(e)}

def generate_search_prompt(user_query, advanced=True):
    """
    Generate a semantic search prompt from the user query.
    
    Args:
        user_query (str): Original user query
        advanced (bool): Whether to use advanced prompt engineering
        
    Returns:
        str: Enhanced query prompt for semantic search
    """
    # Basic enhancement
    if not advanced:
        return enhance_user_query(user_query)
    
    # Advanced prompt engineering with explicit search objectives
    query = user_query.strip()
    
    # Define prompt components
    core_question = f"Original question: {query}"
    search_directive = "Find log entries that directly answer this question."
    
    # Extract key aspects
    has_error = any(term in query.lower() for term in ["error", "exception", "fail", "crash", "issue", "bug"])
    has_time = any(term in query.lower() for term in ["today", "yesterday", "hour", "recent", "latest", "last"])
    has_count = any(term in query.lower() for term in ["how many", "count", "frequency", "rate", "often"])
    has_comparison = any(term in query.lower() for term in ["compare", "difference", "versus", "vs", "trend"])
    
    # Add specific search objectives
    prompt_parts = [core_question, search_directive]
    
    if has_error:
        prompt_parts.append("Prioritize error and exception logs that match the query context.")
        prompt_parts.append("Look for stack traces, error messages, and warnings relevant to the issue.")
    
    if has_time:
        prompt_parts.append("Focus on the time period mentioned in the query.")
        prompt_parts.append("Consider temporal sequence of events.")
    
    if has_count:
        prompt_parts.append("Find logs showing frequency or counts relevant to the query.")
        
    if has_comparison:
        prompt_parts.append("Locate logs that can be compared based on the mentioned criteria.")
    
    # Add semantic context enhancement
    prompt_parts.append(f"Additional search context: {enhance_user_query(query)}")
    
    # Combine prompt parts
    advanced_prompt = " ".join(prompt_parts)
    logger.info(f"Generated advanced search prompt: {advanced_prompt}")
    return advanced_prompt

def process_query(query, top_k=15, analysis=True, verbose=True, model_name=None, max_retries=3, k=None):
    """
    Process a natural language query about logs.
    
    Args:
        query (str): User query about logs
        top_k (int): Number of relevant logs to return
        analysis (bool): Whether to perform LLM analysis
        verbose (bool): Whether to print verbose output
        model_name (str, optional): The name of the model to use for analysis. If provided, overrides the default model.
        max_retries (int): Maximum number of retries for API calls
        k (int, optional): Alternative parameter for top_k for backward compatibility
        
    Returns:
        dict: Results including relevant logs and analysis
    """
    start_time = time.time()
    logger.info(f"Processing query: {query}")
    
    # Use k parameter if provided (for backward compatibility)
    if k is not None:
        top_k = k
    
    # Extract potential filters from query
    filters = extract_filters_from_query(query)
    if verbose:
        logger.info(f"Extracted filters: {json.dumps(filters, indent=2)}")
    
    # Generate advanced search prompt
    #search_prompt = generate_search_prompt(query, advanced=True)
    search_prompt = query
    if verbose:
        logger.info(f"Generated search prompt: {search_prompt}")
    
    # Fetch relevant logs from Vector DB
    relevant_logs = fetch_relevant_logs(search_prompt, top_k=top_k, filters=filters, verbose=verbose)
    
    # Perform analysis if requested
    analysis_result = None
    if analysis and relevant_logs:
        try:
            # Use the specified model or default
            actual_model = model_name if model_name else LLM_MODEL
            if model_name:
                logger.info(f"Using model {model_name} for analysis")
            
            # Pass the model directly to analyze_logs
            analysis_result = analyze_logs_with_model(query, relevant_logs, actual_model, verbose=verbose)
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            analysis_result = f"Analysis failed: {str(e)}"
    
    # Prepare results
    results = {
        "query": query,
        "filters_applied": filters,
        "relevant_logs": relevant_logs,
        "analysis": analysis_result,
        "processing_time": f"{time.time() - start_time:.2f} seconds",
        "logs_found": len(relevant_logs),
        "documents": [log.get("content", "") for log in relevant_logs],
        "response": analysis_result or "No analysis available",
        "error": None
    }
    
    logger.info(f"Query processing complete. Found {len(relevant_logs)} relevant logs.")
    return results

def analyze_logs_with_model(query, log_entries, model_name, verbose=False):
    """
    Analyze log entries based on user query using a specific LLM model.
    
    Args:
        query (str): User query or question about logs
        log_entries (list): List of relevant log entries
        model_name (str): Name of the LLM model to use
        verbose (bool): Whether to print verbose information
        
    Returns:
        str: Analysis of log entries
    """
    try:
        if not log_entries:
            return "No relevant log entries found to analyze."
            
        # Count logs by level for context
        level_counts = {}
        source_counts = {}
        timestamp_range = {"earliest": None, "latest": None}
        
        for entry in log_entries:
            # Process level counts
            if 'metadata' in entry and 'level' in entry['metadata']:
                level = entry['metadata']['level']
                level_counts[level] = level_counts.get(level, 0) + 1
                
            # Process source counts
            if 'metadata' in entry and 'source' in entry['metadata']:
                source = entry['metadata']['source']
                source_counts[source] = source_counts.get(source, 0) + 1
                
            # Process timestamp range
            if 'metadata' in entry and 'timestamp' in entry['metadata']:
                timestamp = entry['metadata']['timestamp']
                if timestamp:
                    if not timestamp_range["earliest"] or timestamp < timestamp_range["earliest"]:
                        timestamp_range["earliest"] = timestamp
                    if not timestamp_range["latest"] or timestamp > timestamp_range["latest"]:
                        timestamp_range["latest"] = timestamp
        
        # Extract content for analysis
        log_contents = []
        for i, entry in enumerate(log_entries):
            content = entry['content']
            score = entry.get('similarity_score', 0)
            metadata = entry.get('metadata', {})
            level = metadata.get('level', 'UNKNOWN')
            timestamp = metadata.get('timestamp', 'UNKNOWN')
            source = metadata.get('source', 'UNKNOWN')
            
            # Format entry text with metadata context - removed log number
            entry_text = f"[{level}] [{source}] [{timestamp}]: {content}"
            log_contents.append(entry_text)
        
        # Create analysis prompt with dynamic context
        analysis_template = """
        You are an AI assistant helping to analyze and interpret system application logs.

        Context:
        The following log data is retrieved from a semantic search using a vector database. These are the most relevant snippets matching the user’s query.

        {log_entries}

        User Query:
        {query}

        Instructions:
        - Carefully analyze the log context provided above.
        - Provide a clear and concise explanation or answer to the user’s query.
        - If the query is about errors or warnings, explain possible causes and suggest troubleshooting steps.
        - Avoid hallucination. Do not guess if the context doesn't support the answer.
        - if the query is about a specific log entry, provide details about that entry.
        - If the query is about a specific time range, summarize the logs within that range.
        - If the query is about a specific log level, summarize the logs of that level.
        - If the answer is not present in the context, politely respond with “No relevant log information found related to your query.”

        Answer:
        """
        analysis_template2 = """
        You are a log analysis expert examining log entries from a system. 
        
        USER QUESTION: {query}
        
        LOG ENTRIES:
        {log_entries}
        
        Based on these log entries, provide a concise, technical analysis that directly addresses the user's query.
        
        Your analysis should:
        1. Summarize the key findings that answer the user's question
        2. Identify patterns or anomalies in the logs relevant to the query
        3. Note correlations between events if present
        4. Suggest possible root causes for issues found
        5. Recommend next troubleshooting steps if appropriate
        
        If the logs contain errors:
        - Explain what the error means
        - Identify likely causes
        - Suggest solutions
        
        Format your response in clear sections and use technical but precise language.
        """
        # Format data for the prompt
        time_range_text = "Unknown time range"
        if timestamp_range["earliest"] and timestamp_range["latest"]:
            time_range_text = f"From {timestamp_range['earliest']} to {timestamp_range['latest']}"
            
        level_distribution_text = ", ".join([f"{level}: {count}" for level, count in level_counts.items()])
        source_distribution_text = ", ".join([f"{source}: {count}" for source, count in source_counts.items()])
        
        # Initialize the LLM with the specified model - using updated class
        llm = OllamaLLM(model=model_name)
        logger.info(f"Final analysis_template: {analysis_template}")
        # Create and run the prompt
        prompt = PromptTemplate(
            template=analysis_template,
            input_variables=["query", "log_entries"]
        )
        
        # Create modern chain using the | operator instead of LLMChain
        chain = prompt | llm
        logger.info(f"prompt: {prompt}")
        logger.info(f"Analyzing logs with LLM model: {model_name}")
        logger.info(f"Query:: {query} Time_range: {time_range_text} Level: {level_distribution_text}  Source: {source_distribution_text}  Log Entries: {log_contents}")
        # Use invoke() instead of run()
        response = chain.invoke({
            "query": query,
           # "time_range": time_range_text,
            #"level_distribution": level_distribution_text,
            #"source_distribution": source_distribution_text,
           "log_entries": "\n\n".join(log_contents)
        })
        
        logger.info("Log analysis complete")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing logs: {str(e)}")
        return f"Log analysis failed: {str(e)}"
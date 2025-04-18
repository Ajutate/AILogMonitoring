import os
import logging
import time
import re
from datetime import datetime
from log_config import setup_logging

# Set up logging
logger = setup_logging()

def extract_logs(log_path, max_retries=3):
    """
    Extract logs from files in the specified directory or from a single log file with retry mechanism.

    Args:
        log_path (str): Path to the directory containing log files or a single log file.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        list: A list of log entries or empty list if failed.
    """
    if not os.path.exists(log_path):
        logger.error(f"Path does not exist: {log_path}")
        return []
        
    retry_count = 0
    logs = []
    
    while retry_count < max_retries:
        try:
            logger.info(f"Extracting logs attempt {retry_count + 1}/{max_retries}")
            logs = []
            
            # Handle both directory and single file cases
            if os.path.isdir(log_path):
                # It's a directory - find all log files
                log_directory = log_path
                log_files = [f for f in os.listdir(log_directory) 
                            if os.path.isfile(os.path.join(log_directory, f)) 
                            and f.endswith(".log")]
                
                if not log_files:
                    logger.warning(f"No log files found in directory {log_directory}")
                    return []
                    
                logger.info(f"Found {len(log_files)} log files to process in directory")
                
                # Process each log file
                for file in log_files:
                    file_path = os.path.join(log_directory, file)
                    try:
                        processed_logs = process_log_file(file_path, file)
                        logs.extend(processed_logs)
                        logger.info(f"Processed {file} - {len(processed_logs)} log entries")
                        logger.info(f"Processed {file} -  {processed_logs}")
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {str(e)}")
            
            elif os.path.isfile(log_path):
                # It's a single file-- This condtion is for single file processing
                logger.info(f"Processing single file: {log_path}")
                # Check if the file has a .log extension
                file = os.path.basename(log_path)
                if file.endswith(".log"):
                    try:
                        processed_logs = process_log_file(log_path, file)
                        logs.extend(processed_logs)
                        logger.info(f"Processed single file {file} - {len(processed_logs)} log entries")
                        logger.info(f"processed_logs varibale -  {processed_logs}")
                        logger.info(f"logs variable-  {logs}")
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {str(e)}")
                else:
                    logger.warning(f"File {file} does not have .log extension")
                    return []
            else:
                logger.error(f"Path {log_path} is neither a directory nor a file")
                return []
            
            logger.info(f"Successfully extracted {len(logs)} log entries")
            return logs
            
        except Exception as e:
            retry_count += 1
            logger.error(f"Error extracting logs: {str(e)}")
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to extract logs after {max_retries} attempts")
                return []

def process_log_file(file_path, source_file):
    """
    Process a log file, handling traditional log formats including multi-line exceptions and various timestamp/log level notations.
    
    Args:
        file_path (str): Path to the log file
        source_file (str): Name of the source file
        
    Returns:
        list: List of processed log entries as dictionaries
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    log_entries = []

    # Enhanced regex: supports [LEVEL], - LEVEL -, and no brackets; supports optional microseconds; allows extra whitespace
    log_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[,.]\d+)?)\s*(?:\[|-)?\s*(INFO|WARN|WARNING|ERROR|DEBUG|EXCEPTION|CRITICAL|FATAL)\s*(?:\]|-)?\s*(.*?)(?=\n\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[,.]\d+)?\s*(?:\[|-)?\s*(?:INFO|WARN|WARNING|ERROR|DEBUG|EXCEPTION|CRITICAL|FATAL)|\Z)'
    logger.info(f"content:: {content}")
    # Find all log entries (including multi-line exceptions)
    log_matches = re.findall(log_pattern, content, re.DOTALL)
    logger.info(f"Log_matches:: {log_matches}")
    for timestamp_str, level, message in log_matches:
        normalized_timestamp = normalize_timestamp(timestamp_str)
        #logger.info(f"Lsource_file:: {source_file}")
        logger.info(f"Log fetched message:: {message.strip()}")
        logger.info(f"timestamp_str:: {timestamp_str}")
        log_entry = {
            "timestamp": timestamp_str,
            "level": level,
            "content": message.strip(),
            "source": source_file,
            "raw": f"{timestamp_str} {level} {message.strip()}"
        }
        # Enhanced: catch multi-line exceptions and stack traces
        if level == "EXCEPTION" or (level == "ERROR" and ("Exception" in message or "Error" in message)):
            exception_data = extract_exception_data(message.strip())
            if exception_data:
                log_entry.update(exception_data)
        log_entries.append(log_entry)

    # Fallback: catch lines not matched by regex (e.g., syslog, Apache, or custom formats)
    lines = content.splitlines()
    for line in lines:
        logger.info("Inside catch lines not matched by regex (e.g., syslog, Apache, or custom formats)")
        if not any(log_entry['content'] in line for log_entry in log_entries):
            ts = extract_timestamp(line)
            if ts:
                log_entries.append({
                    'timestamp': ts,
                    'level': 'UNKNOWN',
                    'content': line.strip(),
                    'source': source_file,
                })
            logger.info(f"Added unknown log entry: {line.strip()} with timestamp: {ts}")
            
    return log_entries

def normalize_timestamp(timestamp_str):
    """
    Convert various timestamp formats to a standardized ISO format.
    
    Args:
        timestamp_str (str): A timestamp string from a log entry
        
    Returns:
        str: ISO formatted timestamp
    """
    # Handle common timestamp formats
    try:
        # Replace comma with period for consistent parsing
        timestamp_str = timestamp_str.replace(',', '.')
        
        # Handle format: 2025-04-14 09:15:01.212
        if re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+', timestamp_str):
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            return dt.isoformat(timespec='milliseconds')
        
        # Handle format without milliseconds: 2025-04-14 09:15:01
        elif re.match(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$', timestamp_str):
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            return dt.isoformat(timespec='seconds')
        
        # If no specific format matched, return the original
        return timestamp_str
    except Exception as e:
        logger.warning(f"Could not normalize timestamp '{timestamp_str}': {str(e)}")
        return timestamp_str

def extract_exception_data(message):
    """
    Extract structured data from exception log messages.
    
    Args:
        message (str): The exception message text
        
    Returns:
        dict: Dictionary with exception details or None if not an exception
    """
    exception_data = {}
    
    # Try to extract exception type and location
    exception_match = re.search(r'(\w+(?:Exception|Error))\s+in\s+([\w\.]+):(\d+)', message)
    if exception_match:
        exception_data['exception_type'] = exception_match.group(1)
        exception_data['exception_location'] = exception_match.group(2)
        exception_data['exception_line'] = exception_match.group(3)
    
    # Extract stack trace
    stack_trace_lines = []
    for line in message.split('\n'):
        if line.strip().startswith('at '):
            stack_trace_lines.append(line.strip())
    
    if stack_trace_lines:
        exception_data['stack_trace'] = stack_trace_lines
    
    # Extract error message
    error_message_match = re.search(r'Error:(.*?)(?=\n\s*at|\Z)', message, re.DOTALL)
    if error_message_match:
        exception_data['error_message'] = error_message_match.group(1).strip()
    
    return exception_data if exception_data else None

def extract_timestamp(log_line):
    """
    Attempt to extract a timestamp from a log line.
    
    Args:
        log_line (str): A line from a log file.
        
    Returns:
        str: ISO formatted timestamp if found, None otherwise.
    """
    # Common timestamp formats to try
    import re
    
    # Try common timestamp patterns
    patterns = [
        # 2023-04-13 14:22:33,123
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]?\d*)',
        # 13/Apr/2023:14:22:33 +0000
        r'(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[\+\-]\d{4})',
        # Apr 13 14:22:33
        r'([A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
        # 13-Apr-2023 14:22:33
        r'(\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2}:\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, log_line)
        if match:
            try:
                # Return the matched timestamp string directly
                return match.group(1)
            except Exception:
                pass
    
    return None
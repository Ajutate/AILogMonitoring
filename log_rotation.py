import os
import shutil
from datetime import datetime
import logging

def rotate_logs():
    """
    Archive the current app.log file with today's date and create a new empty log file.
    
    Returns:
        tuple: (bool, str) - Success status and message
    """
    try:
        # Set up paths
        log_dir = "./logs"
        log_file = os.path.join(log_dir, "app.log")
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")
            return True, f"Created log directory: {log_dir}"
            
        # Check if app.log exists
        if not os.path.exists(log_file):
            # Create an empty log file
            with open(log_file, "w") as f:
                pass
            print(f"Created new empty log file: {log_file}")
            return True, f"Created new empty log file: {log_file}"
            
        # Generate archive filename with current date
        current_date = datetime.now().strftime("%Y_%m_%d")
        archive_filename = f"app_{current_date}.log"
        archive_path = os.path.join(log_dir, archive_filename)
        
        # Handle case where archive file already exists
        counter = 1
        while os.path.exists(archive_path):
            archive_filename = f"app_{current_date}_{counter}.log"
            archive_path = os.path.join(log_dir, archive_filename)
            counter += 1
            
        # Copy existing log to archive name
        shutil.copy2(log_file, archive_path)
        
        # Get size information
        original_size = os.path.getsize(log_file) / 1024  # KB
        
        # Clear the existing log file
        with open(log_file, "w") as f:
            f.write(f"Log file created after archiving previous log to {archive_filename} on {datetime.now().isoformat()}\n")
        
        message = f"Archived log file to {archive_filename} ({original_size:.2f} KB) and created new log file."
        print(message)
        return True, message
        
    except Exception as e:
        error_message = f"Error rotating logs: {str(e)}"
        print(error_message)
        return False, error_message

if __name__ == "__main__":
    success, message = rotate_logs()
    exit_code = 0 if success else 1
    
    # Configure logging in the new file
    if success:
        logging.basicConfig(
            filename="./logs/app.log",
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.info("Log rotation completed successfully")
        logging.info(message)
    
    exit(exit_code)
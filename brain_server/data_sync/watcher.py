"""
File watcher for Obsidian vault using watchdog.
Monitors for file changes and automatically ingests new/modified markdown files.
Recursively watches ALL subdirectories in the vault.
"""
import asyncio
import os
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_pipeline import ingest_document
from core.logger import setup_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)

WATCH_DIRECTORY = os.getenv("WATCH_DIRECTORY", "/data/obsidian")
# Folders to exclude from watching
EXCLUDED_FOLDERS = {'.obsidian', '.trash', '.git', 'node_modules', '.DS_Store'}
# Debounce delay to avoid multiple rapid ingestions
DEBOUNCE_SECONDS = 3


class ObsidianFileHandler(FileSystemEventHandler):
    """Handler for file system events in Obsidian vault"""
    
    def __init__(self):
        super().__init__()
        self.pending_files = {}  # file_path: last_modified_time
        self.loop = None
    
    def should_process_file(self, file_path: str) -> bool:
        """
        Check if file should be processed.
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            bool: True if file should be processed
        """
        # Only process markdown files
        if not file_path.endswith('.md'):
            return False
        
        # Check if file is in excluded folder
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part in EXCLUDED_FOLDERS:
                logger.debug(f"Skipping file in excluded folder: {file_path}")
                return False
        
        return True
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self.should_process_file(file_path):
            logger.info(f"New file detected: {file_path}")
            self.schedule_ingestion(file_path)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if self.should_process_file(file_path):
            logger.info(f"File modified: {file_path}")
            self.schedule_ingestion(file_path)
    
    def schedule_ingestion(self, file_path: str):
        """
        Schedule a file for ingestion with debouncing.
        
        Args:
            file_path (str): Path to the file to ingest
        """
        current_time = time.time()
        self.pending_files[file_path] = current_time
        
        # Schedule debounced ingestion if we have an event loop
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.debounced_ingest(file_path, current_time),
                self.loop
            )
    
    async def debounced_ingest(self, file_path: str, scheduled_time: float):
        """
        Ingest file after debounce delay.
        
        Args:
            file_path (str): Path to the file
            scheduled_time (float): Time when ingestion was scheduled
        """
        # Wait for debounce period
        await asyncio.sleep(DEBOUNCE_SECONDS)
        
        # Check if this is still the latest schedule for this file
        if self.pending_files.get(file_path) == scheduled_time:
            await self.ingest_file(file_path)
            # Remove from pending
            self.pending_files.pop(file_path, None)
    
    async def ingest_file(self, file_path: str):
        """
        Ingest a single file.
        
        Args:
            file_path (str): Path to the file to ingest
        """
        try:
            # Extract metadata from file path
            rel_path = os.path.relpath(file_path, WATCH_DIRECTORY)
            folder = os.path.dirname(rel_path)
            file_name = os.path.basename(file_path)
            
            metadata = {
                "folder": folder if folder else "root",
                "ingestion_type": "auto_watch"
            }
            
            logger.info(f"Ingesting: {rel_path}")
            result = await ingest_document(file_path, metadata)
            
            if result.get('success'):
                logger.info(f"✓ Successfully ingested {file_name} ({result.get('chunks_added', 0)} chunks)")
            else:
                logger.error(f"✗ Failed to ingest {file_name}: {result.get('message')}")
        
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}", exc_info=True)


async def watch_vault():
    """Main watcher function - monitors vault for changes"""
    logger.info(f"Starting file watcher for: {WATCH_DIRECTORY}")
    logger.info(f"Excluding folders: {', '.join(EXCLUDED_FOLDERS)}")
    
    # Check if watch directory exists
    if not os.path.exists(WATCH_DIRECTORY):
        logger.error(f"Watch directory does not exist: {WATCH_DIRECTORY}")
        logger.error("Please check your WATCH_DIRECTORY environment variable")
        return
    
    # Create event handler
    event_handler = ObsidianFileHandler()
    event_handler.loop = asyncio.get_running_loop()
    
    # Use PollingObserver for Docker + Windows compatibility
    # This works reliably with mounted volumes where native events don't propagate
    observer = PollingObserver(timeout=1.0)  # Check every 1 second
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=True)
    observer.start()
    
    logger.info("✓ File watcher started successfully")
    logger.info("📁 Monitoring ALL subdirectories recursively")
    logger.info("Watching for changes... (Press Ctrl+C to stop)")
    
    try:
        # Keep the watcher running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()
    
    observer.join()
    logger.info("File watcher stopped")


def main():
    """Entry point for watcher service"""
    try:
        asyncio.run(watch_vault())
    except Exception as e:
        logger.error(f"Watcher failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

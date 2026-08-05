"""File watcher for the Obsidian vault.

Monitors the vault recursively and re-ingests markdown files when they are
created or modified. Changes are debounced so rapid saves only trigger one
ingestion.
"""
import asyncio
import os
import time
from pathlib import Path
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_pipeline import ingest_document
from core.logger import setup_logger
from core import config

logger = setup_logger(__name__)


class ObsidianFileHandler(FileSystemEventHandler):
    """Handles file system events in the Obsidian vault."""

    def __init__(self):
        super().__init__()
        self.pending_files = {}  # file_path -> last scheduled time
        self.loop = None

    def should_process_file(self, file_path: str) -> bool:
        """Only process markdown files outside excluded folders."""
        if not file_path.endswith(".md"):
            return False

        for part in Path(file_path).parts:
            if part in config.EXCLUDED_FOLDERS:
                return False

        return True

    def on_created(self, event):
        if event.is_directory:
            return
        if self.should_process_file(event.src_path):
            logger.info(f"New file detected: {event.src_path}")
            self.schedule_ingestion(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self.should_process_file(event.src_path):
            logger.info(f"File modified: {event.src_path}")
            self.schedule_ingestion(event.src_path)

    def schedule_ingestion(self, file_path: str):
        """Schedule a file for ingestion after the debounce delay."""
        current_time = time.time()
        self.pending_files[file_path] = current_time

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.debounced_ingest(file_path, current_time),
                self.loop,
            )

    async def debounced_ingest(self, file_path: str, scheduled_time: float):
        """Ingest a file once no new events have arrived for it."""
        await asyncio.sleep(config.DEBOUNCE_SECONDS)

        # Only run if this is still the latest schedule for the file
        if self.pending_files.get(file_path) == scheduled_time:
            await self.ingest_file(file_path)
            self.pending_files.pop(file_path, None)

    async def ingest_file(self, file_path: str):
        """Ingest a single file with vault-relative metadata."""
        try:
            rel_path = os.path.relpath(file_path, config.WATCH_DIRECTORY)
            folder = os.path.dirname(rel_path)
            file_name = os.path.basename(file_path)

            metadata = {
                "folder": folder if folder else "root",
                "ingestion_type": "auto_watch",
            }

            result = await ingest_document(file_path, metadata)

            if result.get("success"):
                chunks_added = result.get("chunks_added", 0)
                chunks_deleted = result.get("chunks_deleted", 0)

                if result.get("is_update"):
                    logger.info(
                        f"Updated {file_name}: {chunks_deleted} old chunks "
                        f"replaced by {chunks_added}"
                    )
                else:
                    logger.info(f"Ingested {file_name}: {chunks_added} chunks")
            else:
                logger.error(f"Failed to ingest {file_name}: {result.get('message')}")

        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}", exc_info=True)


async def watch_vault():
    """Run the watcher until interrupted."""
    logger.info(f"Starting file watcher for: {config.WATCH_DIRECTORY}")
    logger.info(f"Excluding folders: {', '.join(config.EXCLUDED_FOLDERS)}")

    if not os.path.exists(config.WATCH_DIRECTORY):
        logger.error(f"Watch directory does not exist: {config.WATCH_DIRECTORY}")
        logger.error("Check the WATCH_DIRECTORY environment variable")
        return

    event_handler = ObsidianFileHandler()
    event_handler.loop = asyncio.get_running_loop()

    # PollingObserver works reliably with Docker-mounted volumes where
    # native filesystem events don't propagate.
    observer = PollingObserver(timeout=1.0)
    observer.schedule(event_handler, config.WATCH_DIRECTORY, recursive=True)
    observer.start()

    logger.info("File watcher started, monitoring all subdirectories recursively")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()

    observer.join()
    logger.info("File watcher stopped")


def main():
    try:
        asyncio.run(watch_vault())
    except Exception as e:
        logger.error(f"Watcher failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

"""Bulk ingestion of all markdown files in the Obsidian vault.

Recursively scans the vault and ingests every .md file, processing files in
small concurrent batches.
"""
from pathlib import Path
import asyncio
import os
from typing import List
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain_server.core.rag_pipeline import ingest_document
from brain_server.core.logger import setup_logger
from brain_server.core import config

logger = setup_logger(__name__)


def find_markdown_files(vault_path: str) -> List[str]:
    """Recursively find all markdown files in the vault."""
    markdown_files = []

    if not Path(vault_path).exists():
        logger.error(f"Vault path does not exist: {vault_path}")
        return []

    logger.info(f"Scanning vault: {vault_path}")

    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [d for d in dirs if d not in config.EXCLUDED_FOLDERS]
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))

    logger.info(f"Found {len(markdown_files)} markdown files")
    return markdown_files


async def ingest_all_files(file_paths: List[str], batch_size: int = 5):
    """Ingest all files in concurrent batches, logging a summary at the end."""
    total_files = len(file_paths)
    successful = 0
    failed = 0

    logger.info(f"Starting bulk ingestion of {total_files} files")

    for i in range(0, total_files, batch_size):
        batch = file_paths[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_num}/{total_batches}")

        tasks = []
        for file_path in batch:
            rel_path = os.path.relpath(file_path, config.WATCH_DIRECTORY)
            folder = os.path.dirname(rel_path)
            metadata = {
                "folder": folder if folder else "root",
                "ingestion_type": "bulk_import",
            }
            tasks.append(ingest_document(file_path, metadata))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            file_name = os.path.basename(batch[idx])

            if isinstance(result, Exception):
                logger.error(f"Failed to ingest {file_name}: {result}")
                failed += 1
            elif isinstance(result, dict) and result.get("success"):
                logger.info(f"Ingested {file_name} ({result.get('chunks_added', 0)} chunks)")
                successful += 1
            else:
                logger.warning(f"Failed {file_name}: {result.get('message', 'Unknown error')}")
                failed += 1

    logger.info(
        f"Bulk ingestion complete: {successful}/{total_files} succeeded, "
        f"{failed} failed ({(successful / total_files) * 100:.1f}% success rate)"
    )


async def main():
    try:
        markdown_files = find_markdown_files(config.WATCH_DIRECTORY)

        if not markdown_files:
            logger.warning("No markdown files found to ingest")
            return

        print(f"Found {len(markdown_files)} markdown files in: {config.WATCH_DIRECTORY}")
        print(f"Excluded folders: {', '.join(config.EXCLUDED_FOLDERS)}")

        # Auto-proceed inside Docker, ask for confirmation locally
        if os.path.exists("/.dockerenv"):
            proceed = "y"
        else:
            proceed = input("Proceed with bulk ingestion? (y/n): ").lower().strip()

        if proceed == "y":
            await ingest_all_files(markdown_files)
        else:
            logger.info("Bulk ingestion cancelled by user")

    except Exception as e:
        logger.error(f"Bulk ingestion failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

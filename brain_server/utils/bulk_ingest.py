"""
Utility script to bulk ingest all markdown files from Obsidian vault.
Recursively scans directories and ingests all .md files.
"""
import asyncio
import os
from pathlib import Path
from typing import List
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
# Folders to exclude from ingestion
EXCLUDED_FOLDERS = {'.obsidian', '.trash', '.git', 'node_modules'}


def find_markdown_files(vault_path: str) -> List[str]:
    """
    Recursively find all markdown files in the vault.
    
    Args:
        vault_path (str): Path to Obsidian vault root
    
    Returns:
        List[str]: List of absolute paths to .md files
    """
    markdown_files = []
    vault_path_obj = Path(vault_path)
    
    if not vault_path_obj.exists():
        logger.error(f"Vault path does not exist: {vault_path}")
        return []
    
    logger.info(f"Scanning vault: {vault_path}")
    
    for root, dirs, files in os.walk(vault_path):
        # Remove excluded directories from the walk
        dirs[:] = [d for d in dirs if d not in EXCLUDED_FOLDERS]
        
        # Find all .md files in current directory
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                markdown_files.append(file_path)
    
    logger.info(f"Found {len(markdown_files)} markdown files")
    return markdown_files


async def ingest_all_files(file_paths: List[str], batch_size: int = 5):
    """
    Ingest all files with progress tracking and error handling.
    
    Args:
        file_paths (List[str]): List of file paths to ingest
        batch_size (int): Number of files to process concurrently
    """
    total_files = len(file_paths)
    successful = 0
    failed = 0
    
    logger.info(f"Starting bulk ingestion of {total_files} files...")
    
    # Process files in batches for better concurrency
    for i in range(0, total_files, batch_size):
        batch = file_paths[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        # Process batch concurrently
        tasks = []
        for file_path in batch:
            # Extract relative path for metadata
            rel_path = os.path.relpath(file_path, WATCH_DIRECTORY)
            folder = os.path.dirname(rel_path)
            
            metadata = {
                "folder": folder if folder else "root",
                "ingestion_type": "bulk_import"
            }
            
            tasks.append(ingest_document(file_path, metadata))
        
        # Wait for batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        for idx, result in enumerate(results):
            file_path = batch[idx]
            file_name = os.path.basename(file_path)
            
            if isinstance(result, Exception):
                logger.error(f"Failed to ingest {file_name}: {result}")
                failed += 1
            elif isinstance(result, dict) and result.get('success'):
                logger.info(f"✓ Ingested {file_name} ({result.get('chunks_added', 0)} chunks)")
                successful += 1
            else:
                logger.warning(f"✗ Failed {file_name}: {result.get('message', 'Unknown error')}")
                failed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Bulk ingestion complete!")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {(successful/total_files)*100:.1f}%")
    logger.info("=" * 60)


async def main():
    """Main entry point for bulk ingestion"""
    try:
        # Find all markdown files
        markdown_files = find_markdown_files(WATCH_DIRECTORY)
        
        if not markdown_files:
            logger.warning("No markdown files found to ingest!")
            return
        
        # Ask for confirmation
        print(f"\n{'='*60}")
        print(f"Found {len(markdown_files)} markdown files in: {WATCH_DIRECTORY}")
        print(f"Excluded folders: {', '.join(EXCLUDED_FOLDERS)}")
        print(f"{'='*60}\n")
        
        # Auto-proceed in Docker environment, ask for confirmation locally
        if os.path.exists('/.dockerenv'):
            proceed = 'y'
        else:
            proceed = input("Proceed with bulk ingestion? (y/n): ").lower().strip()
        
        if proceed == 'y':
            await ingest_all_files(markdown_files)
        else:
            logger.info("Bulk ingestion cancelled by user")
    
    except Exception as e:
        logger.error(f"Bulk ingestion failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

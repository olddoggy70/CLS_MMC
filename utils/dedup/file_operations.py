"""
File operations module for deduplication pipeline.
Handles file discovery, grouping, and reading operations.
"""

import concurrent.futures
import os
import re
import time  # ✅ NEW: Import time for sleep testing
from collections import defaultdict
from pathlib import Path

import polars as pl

# ✅ NEW IMPORT: Add tqdm for progress tracking
from tqdm import tqdm


def discover_and_group_files(daily_files_dir, logger):
    """
    Discover and group Excel files by date extracted from filename.
    Optimized version using os.scandir for better performance on large directories.

    Args:
        daily_files_dir: Path object or string path to directory
        logger: Logger instance

    Returns:
        dict: Dictionary with date strings as keys and lists of file paths as values
    """
    # Pre-compile regex pattern once (major performance improvement)
    date_pattern = re.compile(r'(\d{8})')
    grouped = defaultdict(list)

    try:
        # Convert to Path object for validation
        daily_files_dir = Path(daily_files_dir)

        # Validate directory exists and is actually a directory
        if not daily_files_dir.exists():
            logger.error(f'Directory does not exist: {daily_files_dir}')
            return {}

        if not daily_files_dir.is_dir():
            logger.error(f'Path is not a directory: {daily_files_dir}')
            return {}

        # Use os.scandir for better performance on large directories
        file_count = 0
        with os.scandir(daily_files_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.xlsx'):
                    file_count += 1
                    match = date_pattern.search(entry.name)
                    if match:
                        date_str = match.group(1)
                        grouped[date_str].append(entry.path)
                    else:
                        logger.warning(f'No date found in filename: {entry.name}')

        if file_count == 0:
            logger.debug(f'No .xlsx files found in {daily_files_dir}')
            return {}

        logger.debug(f'Discovered {len(grouped)} date groups with {file_count} total files.')

    except OSError as e:
        logger.error(f'Error accessing directory {daily_files_dir}: {e}')
        return {}
    except Exception as e:
        logger.error(f'Unexpected error in file discovery: {e}')
        return {}

    return dict(grouped)


# ❌ REMOVED: estimate_total_lines() function - no longer needed


# ✅ MODIFIED FUNCTION: Added progress_bar parameter
def read_and_combine_files(files_in_group, max_workers, logger, progress_bar=None):
    """
    Read and combine Excel files using threading with robust error handling.

    Args:
        files_in_group (list[str]): List of file paths to read
        max_workers (int): Maximum number of worker threads
        logger: Logger instance
        progress_bar (tqdm, optional): Progress bar to update with line counts

    Returns:
        polars.DataFrame or None: Combined DataFrame or None if no valid files
    """
    if not files_in_group:
        logger.warning('No files provided to read.')
        return None

    # ✅ MODIFIED: Added progress tracking to read_file function
    def read_file(file_path):
        """Read a single Excel file safely, return DataFrame or None."""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                logger.error(f'File does not exist: {file_path}')
                return None
            if file_path.stat().st_size == 0:
                logger.warning(f'File is empty: {file_path}')
                return None

            df = pl.read_excel(file_path, infer_schema_length=0)

            if df.is_empty():
                logger.warning(f'File contains no data: {file_path}')
                return None

            # ✅ NEW: Update progress bar with actual rows read
            if progress_bar:
                progress_bar.update(df.height)
                # ✅ TESTING: Add sleep to see progress bars in action
                # time.sleep(2)  # Remove this after testing!

            return df

        except OSError as e:
            logger.error(f'OS error reading {file_path}: {e}')
        except pl.exceptions.PolarsError as e:
            logger.error(f'Polars error reading {file_path}: {e}')
        except Exception as e:
            logger.error(f'Unexpected error reading {file_path}: {e}')
        return None

    # Optimize thread pool size
    workers = min(max_workers, len(files_in_group), 8)
    logger.debug(f'Reading {len(files_in_group)} files using {workers} workers')

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            dfs = list(executor.map(read_file, files_in_group))

        # Separate successes and failures
        successful_dfs = [df for df in dfs if df is not None]
        failed_files_count = len(dfs) - len(successful_dfs)

        if failed_files_count:
            logger.warning(f'{failed_files_count} files failed to read.')

        if not successful_dfs:
            logger.error('No valid files were read from the group.')
            return None

        try:
            combined_df = pl.concat(successful_dfs, how='vertical_relaxed')
            logger.debug(f'Combined {len(successful_dfs)} files, total rows: {combined_df.height}, columns: {combined_df.width}')
            return combined_df
        except pl.exceptions.PolarsError as e:
            logger.error(f'Error combining DataFrames: {e}')
            return None

    except Exception as e:
        logger.error(f'Unexpected error in thread execution: {e}')
        return None

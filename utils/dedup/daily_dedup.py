"""
Daily deduplication module.
Handles within-day duplicate detection and removal.
"""

from pathlib import Path

import polars as pl

from .hash_utils import create_daily_hash


def determine_run_id(combined_df, date_str, db_files_dir, match_keyword, logger):
    """
    Determine the run ID for the current processing batch.
    Compares DataFrame content with existing parquet files to detect duplicates
    and allow overwriting when data matches exactly.

    Args:
        combined_df: DataFrame being processed for content comparison
        date_str: Date string in YYYYMMDD format
        db_files_dir: Directory containing database files
        match_keyword: Keyword pattern for matching files
        logger: Logger instance

    Returns:
        int: Run ID for this processing batch
    """
    run_id = 0

    while True:
        parquet_file = Path(db_files_dir) / f'{match_keyword}_{date_str}_{run_id}.parquet'

        if not parquet_file.exists():
            # No existing file found, use this run_id
            logger.debug(f'No existing file found. Using run_id: {run_id}')
            break

        try:
            # Load existing parquet file for comparison
            existing_df = pl.read_parquet(parquet_file)

            # Compare DataFrames - if they match exactly, we can overwrite
            if combined_df.equals(existing_df):
                logger.info(f'Overwriting {date_str} run {run_id}: matches existing parquet file')
                break
            else:
                # Data differs, try next run_id
                logger.debug(f'Data differs from existing run {run_id}, trying next run_id')
                run_id += 1

        except Exception as e:
            logger.warning(f'Error reading existing parquet file {parquet_file}: {e}')
            # If we can't read the existing file, increment run_id to avoid conflicts
            run_id += 1

    logger.debug(f'Determined run_id: {run_id}')
    return run_id


def daily_deduplicate(
    combined_df, hash_daily_columns, date_str, run_id, date_output_dir, logger, file_ext, write_func, system_name
):
    """
    Perform daily deduplication on the combined DataFrame.

    Args:
        combined_df: Combined DataFrame to deduplicate
        hash_daily_columns: Columns to use for daily hash creation
        date_str: Date string in YYYYMMDD format
        run_id: Run identifier
        date_output_dir: Output directory for date-specific files
        logger: Logger instance
        file_ext: File extension for output files
        write_func: Function to write output files
        system_name: Name of the system being processed

    Returns:
        tuple: (deduped_df, duplicate_file_path)
    """
    # Add daily hash column
    combined_df = combined_df.with_columns([create_daily_hash(hash_daily_columns)])

    # Find duplicates and deduplicate
    duplicates = combined_df.filter(pl.col('hash_key_daily').is_duplicated())
    deduped = combined_df.unique(subset=['hash_key_daily'], keep='last', maintain_order=True)

    # Only write duplicates file if duplicates exist
    if duplicates.height > 0:
        duplicate_file = date_output_dir / f'{system_name}_daily_duplicates_{date_str}_{run_id}.{file_ext}'
        write_func(duplicates, duplicate_file)
        logger.debug(f'Found {duplicates.height} daily duplicates, written to {duplicate_file}')
    else:
        duplicate_file = None
        logger.debug('No daily duplicates found, skipping duplicates file creation')

    return deduped, duplicate_file


def enrich_data(df, logger):
    """
    Enrich data with additional processing.
    Currently a passthrough but can be extended with business logic.

    Args:
        df: DataFrame to enrich
        logger: Logger instance

    Returns:
        polars.DataFrame: Enriched DataFrame
    """
    logger.debug('Enrichment step - currently passthrough.')
    return df


def check_spending(df, logger):
    """
    Check spending logic.
    Currently a passthrough but can be extended with business logic.

    Args:
        df: DataFrame to check
        logger: Logger instance

    Returns:
        polars.DataFrame: Processed DataFrame
    """
    logger.debug('Spending check step - currently passthrough.')
    return df

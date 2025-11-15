"""
Consolidated parquet management module.
Handles loading, writing, and optimizing large parquet files for historical data storage.
"""

import hashlib
from pathlib import Path

import polars as pl


def load_existing_parquet(parquet_path, logger, data_type):
    """
    Load existing consolidated parquet file if it exists.

    Args:
        parquet_path: Path to parquet file
        logger: Logger instance
        data_type: Type of data ("raw" or "deduped") for logging

    Returns:
        pl.DataFrame or None: Existing data or None if file doesn't exist/error
    """
    try:
        if parquet_path.exists():
            logger.debug(f'Loading existing {data_type} parquet: {parquet_path}')
            existing_data = pl.read_parquet(parquet_path)
            logger.debug(f'Loaded {existing_data.height:,} existing {data_type} records')
            return existing_data
        else:
            logger.debug(f'No existing {data_type} parquet found at {parquet_path}')
            return None
    except Exception as e:
        logger.warning(f'Error loading existing {data_type} parquet: {e}')
        return None


def write_consolidated_parquet_files(all_raw_data, all_deduped_data, all_raw_parquet_path, all_deduped_parquet_path, logger):
    """
    Write consolidated parquet files with ALL data (existing + new) and optimized settings.

    Args:
        all_raw_data: List of raw DataFrames to consolidate
        all_deduped_data: List of deduped DataFrames to consolidate
        all_raw_parquet_path: Path for consolidated raw parquet file
        all_deduped_parquet_path: Path for consolidated deduped parquet file
        logger: Logger instance
    """
    try:
        if all_raw_data:
            _write_consolidated_raw_data(all_raw_data, all_raw_parquet_path, logger)

        if all_deduped_data:
            _write_consolidated_deduped_data(all_deduped_data, all_deduped_parquet_path, logger)

    except Exception as e:
        logger.error(f'Error writing consolidated parquet files: {e}')


def _write_consolidated_raw_data(all_raw_data, all_raw_parquet_path, logger):
    """
    Write consolidated raw data parquet with deduplication to prevent reprocessing issues.

    Args:
        all_raw_data: List of raw DataFrames
        all_raw_parquet_path: Output path for consolidated raw parquet
        logger: Logger instance
    """
    logger.debug('Writing consolidated raw data parquet with ALL data...')
    consolidated_raw = pl.concat(all_raw_data, how='vertical_relaxed')

    # Remove duplicates based on processing_date + run_id + original row content
    # to avoid duplicate entries when reprocessing
    if 'processing_date' in consolidated_raw.columns and 'run_id' in consolidated_raw.columns:
        consolidated_raw = _deduplicate_consolidated_data(consolidated_raw, logger)

    # Sort by processing_date for better organization and query performance
    if 'processing_date' in consolidated_raw.columns:
        consolidated_raw = consolidated_raw.sort('processing_date', 'run_id')

    # Write with optimized settings
    _write_optimized_parquet(consolidated_raw, all_raw_parquet_path, row_group_size=50000)

    logger.debug(f'Consolidated raw data written: {all_raw_parquet_path} ({consolidated_raw.height:,} total rows)')


def _write_consolidated_deduped_data(all_deduped_data, all_deduped_parquet_path, logger):
    """
    Write consolidated deduped data parquet.

    Args:
        all_deduped_data: List of deduped DataFrames
        all_deduped_parquet_path: Output path for consolidated deduped parquet
        logger: Logger instance
    """
    logger.debug('Writing consolidated deduped data parquet with ALL data...')
    consolidated_deduped = pl.concat(all_deduped_data, how='vertical_relaxed')

    # Sort by processing_date for better organization
    if 'processing_date' in consolidated_deduped.columns:
        consolidated_deduped = consolidated_deduped.sort('processing_date', 'run_id')

    # Write with optimized settings
    _write_optimized_parquet(consolidated_deduped, all_deduped_parquet_path, row_group_size=50000)

    logger.debug(f'Consolidated deduped data written: {all_deduped_parquet_path} ({consolidated_deduped.height:,} total rows)')


def _deduplicate_consolidated_data(consolidated_raw, logger):
    """
    Remove duplicate entries from consolidated raw data to prevent reprocessing issues.

    Args:
        consolidated_raw: Combined raw DataFrame
        logger: Logger instance

    Returns:
        pl.DataFrame: Deduplicated consolidated DataFrame
    """
    # Create a unique key for deduplication
    hash_columns = [
        col
        for col in consolidated_raw.columns
        if col not in ['processing_date', 'run_id', 'hash_key_daily', 'hash_key_historical']
    ]

    def create_unique_key(row):
        content = '|'.join([str(row[col]) for col in hash_columns])
        return f'{row["processing_date"]}_{row["run_id"]}_{hashlib.md5(content.encode()).hexdigest()[:16]}'

    consolidated_raw = consolidated_raw.with_columns(
        pl.struct([*hash_columns, 'processing_date', 'run_id'])
        .map_elements(create_unique_key, return_dtype=pl.Utf8)
        .alias('unique_key')
    )

    # Deduplicate and remove the temporary unique_key column
    deduplicated = consolidated_raw.unique(subset=['unique_key'], keep='last').drop('unique_key')

    removed_count = consolidated_raw.height - deduplicated.height
    if removed_count > 0:
        logger.debug(f'Removed {removed_count:,} duplicate entries from consolidated raw data')

    return deduplicated


def write_optimized_individual_parquet(df, file_path, logger):
    """
    Write individual parquet files with optimized settings for better performance.

    Args:
        df: DataFrame to write
        file_path: Output file path
        logger: Logger instance
    """
    try:
        _write_optimized_parquet(df, file_path, row_group_size=25000)
    except Exception as e:
        logger.warning(f'Failed optimized parquet write, falling back to default: {e}')
        df.write_parquet(file_path)  # Fallback to default settings


def _write_optimized_parquet(df, file_path, row_group_size=50000):
    """
    Write parquet file with optimized settings.

    Args:
        df: DataFrame to write
        file_path: Output file path
        row_group_size: Size of row groups for optimization
    """
    df.write_parquet(
        file_path,
        compression='snappy',  # Fast compression, good balance of speed/size
        row_group_size=row_group_size,  # Optimize for read performance
        use_pyarrow=True,  # Better performance
        statistics=True,  # Enable statistics for better query performance
    )


def get_parquet_info(parquet_path, logger):
    """
    Get information about a parquet file without loading all data.

    Args:
        parquet_path: Path to parquet file
        logger: Logger instance

    Returns:
        dict: Information about the parquet file (rows, columns, size, etc.)
    """
    try:
        if not parquet_path.exists():
            return {'exists': False}

        # Use scan to get metadata without loading full file
        lazy_df = pl.scan_parquet(parquet_path)

        # Get basic info
        file_size = parquet_path.stat().st_size
        columns = lazy_df.columns

        # Get row count efficiently
        row_count = lazy_df.select(pl.count()).collect().item()

        # Get date range if processing_date column exists
        date_range = None
        if 'processing_date' in columns:
            date_info = lazy_df.select(
                [pl.col('processing_date').min().alias('min_date'), pl.col('processing_date').max().alias('max_date')]
            ).collect()

            date_range = {'min_date': date_info['min_date'][0], 'max_date': date_info['max_date'][0]}

        return {
            'exists': True,
            'rows': row_count,
            'columns': len(columns),
            'column_names': columns,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'date_range': date_range,
        }

    except Exception as e:
        logger.warning(f'Error getting parquet info for {parquet_path}: {e}')
        return {'exists': True, 'error': str(e)}


def optimize_parquet_for_queries(parquet_path, output_path, sort_columns=None, logger=None):
    """
    Optimize an existing parquet file for better query performance.

    Args:
        parquet_path: Input parquet file path
        output_path: Output optimized parquet file path
        sort_columns: Columns to sort by for better query performance
        logger: Logger instance

    Returns:
        bool: True if optimization succeeded, False otherwise
    """
    try:
        if logger:
            logger.debug(f'Optimizing parquet file: {parquet_path}')

        df = pl.read_parquet(parquet_path)

        # Sort by specified columns or default to processing_date
        if sort_columns:
            df = df.sort(sort_columns)
        elif 'processing_date' in df.columns:
            df = df.sort('processing_date', 'run_id' if 'run_id' in df.columns else 'processing_date')

        # Write with optimized settings
        _write_optimized_parquet(df, output_path, row_group_size=100000)

        if logger:
            logger.debug(f'Optimized parquet written: {output_path}')

        return True

    except Exception as e:
        if logger:
            logger.error(f'Error optimizing parquet file: {e}')
        return False

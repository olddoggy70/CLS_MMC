"""
Deduplication Pipeline v12 - Modular Architecture with Enrichment Integration

Refactored pipeline with clear separation of concerns:
- File operations: Discovery, reading, and Excel handling
- Daily deduplication: Within-day duplicate detection
- Historical deduplication: Cross-day duplicate tracking with SQLite
- Hash utilities: MD5 hashing functions
- Parquet management: Optimized parquet operations
- Output writing: Multi-format output handling
- Enrichment: Data enrichment with reference files
"""

import os
import sys
import time
from pathlib import Path

import polars as pl
from tqdm import tqdm
from utils.dedup.daily_dedup import daily_deduplicate, determine_run_id
from utils.dedup.enrichment import enrich_data_with_config
from utils.dedup.file_operations import discover_and_group_files, read_and_combine_files
from utils.dedup.hash_utils import create_main_record_type
from utils.dedup.historical_dedup import historical_deduplicate
from utils.dedup.output_writer import write_and_zip_outputs
from utils.dedup.parquet_manager import load_existing_parquet, write_consolidated_parquet_files
from utils.dedup.spending_check import check_spending_with_config
from utils.file_utils import get_system_logger

# ----------------------------
# Constants for output formats
# ----------------------------
FORMAT_EXTENSIONS = {'csv': 'csv', 'xlsx': 'xlsx', 'parquet': 'parquet'}

FORMAT_WRITERS = {
    'csv': lambda df, path: df.write_csv(path, quote_style='always'),
    'xlsx': lambda df, path: df.write_excel(path),
    'parquet': lambda df, path: df.write_parquet(path),
}

# Color constants for progress display
COLOR_BLUE = '\033[94m'  # bright blue
COLOR_GREEN = '\033[92m'  # bright green
COLOR_YELLOW = '\033[93m'  # yellow
COLOR_RESET = '\033[0m'


# ----------------------------
# 🆕 NEW: Helper function to align schemas for parquet consolidation
# ----------------------------
def align_dataframe_schemas(dataframes, logger):
    """
    Align schemas of multiple DataFrames to ensure they have the same columns
    for safe concatenation during parquet consolidation.

    Args:
        dataframes: List of polars DataFrames
        logger: Logger instance

    Returns:
        List of DataFrames with aligned schemas
    """
    if not dataframes:
        return dataframes

    # Get all unique column names and their data types across all DataFrames
    all_columns = {}  # column_name -> polars_dtype
    for df in dataframes:
        for col_name, col_dtype in zip(df.columns, df.dtypes):
            if col_name in all_columns:
                # Verify compatible types for existing columns
                if all_columns[col_name] != col_dtype:
                    logger.warning(f"Column '{col_name}' has different types: {all_columns[col_name]} vs {col_dtype}")
            else:
                all_columns[col_name] = col_dtype

    logger.debug(f'Schema alignment: Found {len(all_columns)} unique columns across {len(dataframes)} DataFrames')

    aligned_dataframes = []
    for i, df in enumerate(dataframes):
        missing_columns = set(all_columns.keys()) - set(df.columns)
        if missing_columns:
            logger.debug(f'DataFrame {i}: Adding {len(missing_columns)} missing columns: {sorted(missing_columns)}')

            # Add missing columns with appropriate null values and correct data types
            for col in missing_columns:
                expected_dtype = all_columns[col]
                df = df.with_columns(pl.lit(None, dtype=expected_dtype).alias(col))

        aligned_dataframes.append(df)

    return aligned_dataframes


# ----------------------------
# 🆕 NEW: Helper function to drop generated hash key columns from output
# ----------------------------
def drop_hash_key_columns_for_output(df, logger):
    """
    Drop generated hash key columns from DataFrame before writing output files.
    These are the computed hash columns ('hash_key_daily', 'hash_key_historical'),
    not the source columns used for hashing.

    Args:
        df: Input DataFrame
        logger: Logger instance

    Returns:
        DataFrame with hash key columns removed
    """
    # Define the generated hash key columns to drop
    hash_key_columns_to_drop = ['hash_key_daily', 'hash_key_historical']

    # Only drop columns that actually exist in the DataFrame
    existing_hash_key_columns = [col for col in hash_key_columns_to_drop if col in df.columns]

    if existing_hash_key_columns:
        logger.debug(f'Dropping generated hash key columns from output: {existing_hash_key_columns}')
        df = df.drop(existing_hash_key_columns)

    return df


# ----------------------------
# Main pipeline function
# ----------------------------


def run_dedup_pipeline(system_config, domain, system_name, function_dirs, global_config, BASE_DIR, orchestrator_logger=None):
    """
    Runs the deduplication pipeline for a given system within a domain.
    Modularized for multi-system support with clean separation of concerns.

    Args:
        system_config: Configuration specific to the system
        domain: Domain name (e.g., 'healthcare', 'finance')
        system_name: Name of the system being processed
        function_dirs: Dictionary of directory mappings
        global_config: Global configuration settings
        BASE_DIR: Base directory for all operations
        orchestrator_logger: Optional orchestrator logger for high-level summary info
    """
    # Setup directories and logger
    domain_dir = os.path.join(BASE_DIR, domain)
    logger = get_system_logger(domain_dir, domain, system_name)

    daily_files_dir = Path(domain_dir) / function_dirs['daily_files'] / system_name
    db_files_dir = Path(domain_dir) / function_dirs['db_files'] / system_name
    output_files_dir = Path(domain_dir) / function_dirs['output_files'] / system_name
    os.makedirs(output_files_dir, exist_ok=True)

    # Extract configuration
    match_keyword = system_config['match_keyword']
    outfile_format = system_config['output_format']
    hash_daily_columns = global_config['hash_daily_columns']
    hash_historical_columns = global_config['hash_historical_columns']

    # âœ… NEW: Extract enrichment configuration
    enrichment_config = system_config.get('extra_functions', {}).get('enrichment', {})
    enrichment_enabled = enrichment_config.get('enabled', False)

    # âœ… NEW: Extract spending configuration
    spending_config = system_config.get('extra_functions', {}).get('spending', {})
    spending_enabled = spending_config.get('enabled', False)

    # Setup file paths
    historical_db_file = f'{match_keyword}_historical_hash.db'
    historical_db_path = db_files_dir / historical_db_file

    all_raw_parquet_file = f'{system_name}_all_raw_data.parquet'
    all_raw_parquet_path = db_files_dir / all_raw_parquet_file

    all_deduped_parquet_file = f'{system_name}_all_deduped.parquet'
    all_deduped_parquet_path = db_files_dir / all_deduped_parquet_file

    # Validate and setup output format
    output_format = outfile_format.lower()
    if output_format not in FORMAT_EXTENSIONS:
        logger.warning(f"Unknown format '{outfile_format}', defaulting to CSV")
        output_format = 'csv'

    file_ext = FORMAT_EXTENSIONS[output_format]
    write_func = FORMAT_WRITERS[output_format]

    logger.debug(f'=== Starting dedup pipeline for {domain}/{system_name} ===')
    logger.debug(f'Output format: {output_format} (.{file_ext})')
    logger.debug(f'Enrichment enabled: {enrichment_enabled}')
    logger.debug(f'Spending enabled: {spending_enabled}')
    logger.debug(f'Consolidated parquet files: {all_raw_parquet_path}, {all_deduped_parquet_path}')

    # Step 1: Discover daily files grouped by date (lightweight operation)
    file_groups = discover_and_group_files(daily_files_dir, logger)
    if not file_groups:
        logger.warning('No daily files found to process. Skipping dedup.')
        return  # Early exit - no memory wasted on loading existing data

    # Initialize collectors for NEW data only (existing data loaded later)
    new_raw_data = []
    new_deduped_data = []

    # Setup processing parameters
    max_workers = min(4, os.cpu_count() or 1)

    # Step 2 - Process each date group with LINE-LEVEL progress
    for date_str, files_in_group in sorted(file_groups.items()):
        start_time = time.time()

        # Create indeterminate progress bar
        line_pbar = tqdm(
            total=1,
            desc=f'{COLOR_GREEN}Processing {date_str}{COLOR_RESET}',
            unit='lines',
            file=sys.stderr,
            leave=False,  # Don't leave progress bar after completion
        )

        # Read and combine files WITH progress tracking
        combined_df = read_and_combine_files(files_in_group, max_workers, logger, line_pbar)
        if combined_df is None:
            line_pbar.close()
            continue

        # Set total and position after we know the actual count
        line_pbar.total = combined_df.height
        line_pbar.n = combined_df.height  # Set to completed
        line_pbar.refresh()

        # Create date-specific output directory
        date_output_dir = output_files_dir / date_str
        os.makedirs(date_output_dir, exist_ok=True)
        logger.debug(f'Created date-specific output directory: {date_output_dir}')

        # Determine run ID for this processing batch
        run_id = determine_run_id(combined_df, date_str, db_files_dir, match_keyword, logger)
        logger.debug(f'Processing {date_str} run {run_id} data')

        # Add processing metadata to combined_df for tracking
        combined_df = combined_df.with_columns([pl.lit(date_str).alias('processing_date'), pl.lit(run_id).alias('run_id')])

        # Collect NEW raw data for consolidated parquet
        new_raw_data.append(combined_df)

        # Step 3: Daily deduplication
        daily_deduped, duplicate_file = daily_deduplicate(
            combined_df, hash_daily_columns, date_str, run_id, date_output_dir, logger, file_ext, write_func, system_name
        )

        # âœ… UPDATED: Step 4 - Enrichment with configuration support
        if enrichment_enabled:
            logger.debug(f'Running enrichment for {date_str} data...')
            enriched_daily_deduped = enrich_data_with_config(daily_deduped, enrichment_config, domain_dir, system_name, logger)
        else:
            logger.debug('Enrichment disabled, skipping enrichment step')
            enriched_daily_deduped = daily_deduped

        # Step 5: Add main record type classification
        enriched_daily_deduped = enriched_daily_deduped.with_columns([create_main_record_type()])

        # Step 6: Historical deduplication
        historical_deduped = historical_deduplicate(
            enriched_daily_deduped,
            hash_historical_columns,
            historical_db_path,
            date_str,
            run_id,
            logger,
            date_output_dir,
            system_name,
            file_ext,
            write_func,
        )

        # 🆕 NEW: Step 7 - Enhanced spending check with empty DataFrame handling
        if spending_enabled and historical_deduped.height > 0:
            logger.debug(f'Running spending check for {date_str} data ({historical_deduped.height:,} rows)...')
            historical_deduped = check_spending_with_config(historical_deduped, spending_config, domain_dir, system_name, logger)
        elif spending_enabled and historical_deduped.height == 0:
            logger.debug(f'Skipping spending check for {date_str} - no records after historical deduplication')
            # Add columns [Match_Key, Matched_String, Similarity_Score] to align the structure
            historical_deduped = historical_deduped.with_columns(
                Match_Key=pl.lit(''), Matched_String=pl.lit(''), Similarity_Score=pl.lit(0.0, dtype=pl.Float64)
            )
        else:
            logger.debug('Spending disabled, skipping spending step')

        # # Add processing metadata to deduped data
        # historical_deduped_with_meta = historical_deduped.with_columns(
        #     [pl.lit(date_str).alias('processing_date'), pl.lit(run_id).alias('run_id')]
        # )
        new_deduped_data.append(historical_deduped)

        # 🆕 NEW: Step 8 - Drop generated hash key columns before writing output files
        daily_deduped_clean = drop_hash_key_columns_for_output(daily_deduped, logger)
        historical_deduped_clean = drop_hash_key_columns_for_output(historical_deduped, logger)

        # Step 9: Write output files (with hash columns removed)
        write_and_zip_outputs(
            daily_deduped_clean,
            historical_deduped_clean,
            date_str,
            run_id,
            duplicate_file,
            date_output_dir,
            logger,
            file_ext,
            write_func,
            system_name,
        )

        # Close the progress bar
        line_pbar.close()

        # Log processing summary
        spend_rows = historical_deduped.filter((pl.col('Similarity_Score') > 0) & (pl.col('main_rec_type') != 'H')).height
        processing_time = time.time() - start_time

        summary_msg = (
            f'Processed {date_str} run {run_id}: Total rows={combined_df.height:,} '
            f'Unique rows={historical_deduped.height:,} Spend rows={spend_rows:,} Time={processing_time:.2f} s'
        )
        tqdm.write(
            f'{COLOR_YELLOW}Processed {date_str} run {run_id}: Total rows={combined_df.height:,}  '
            f'Unique rows={historical_deduped.height:,} Spend rows={spend_rows:,} Time={processing_time:.2f} s{COLOR_RESET}'
        )
        # Log to system logger
        logger.debug(summary_msg)

        # Also log to orchestrator logger if provided
        if orchestrator_logger:
            orchestrator_logger.debug(f'{domain}::{system_name} - {summary_msg}')

    # 🆕 NEW: Step 10 - Enhanced parquet consolidation with schema alignment
    # Only load existing data when we actually need it for consolidation
    logger.debug('Loading existing consolidated data for final consolidation...')
    tqdm.write('[INFO] Loading existing consolidated data for final consolidation...')

    existing_raw_data = load_existing_parquet(all_raw_parquet_path, logger, 'raw')
    existing_deduped_data = load_existing_parquet(all_deduped_parquet_path, logger, 'deduped')

    # Combine existing + new data
    all_raw_data = []
    if existing_raw_data is not None:
        all_raw_data.append(existing_raw_data)
    all_raw_data.extend(new_raw_data)

    all_deduped_data = []
    if existing_deduped_data is not None:
        all_deduped_data.append(existing_deduped_data)
    all_deduped_data.extend(new_deduped_data)

    # 🆕 NEW: Align schemas before writing consolidated parquet files
    if all_raw_data:
        logger.debug('Aligning schemas for raw data consolidation...')
        all_raw_data = align_dataframe_schemas(all_raw_data, logger)

    if all_deduped_data:
        logger.debug('Aligning schemas for deduped data consolidation...')
        all_deduped_data = align_dataframe_schemas(all_deduped_data, logger)

    # Write consolidated parquet files (much faster than individual writes)
    write_consolidated_parquet_files(all_raw_data, all_deduped_data, all_raw_parquet_path, all_deduped_parquet_path, logger)

    logger.debug(f'=== Dedup pipeline completed for {domain}/{system_name} ===')

"""
Count validation utilities for verifying data integrity between SQLite and parquet files.
"""

import sqlite3
from pathlib import Path

import polars as pl


def validate_historical_counts(db_files_dir, match_keyword, logger, tolerance=0):
    """
    Validate that SQLite hash count matches the total deduplicated parquet records.
    Runs after each domain->system processing is complete.

    Args:
        db_files_dir: Directory containing database and parquet files
        match_keyword: System match keyword for file naming
        logger: Logger instance
        tolerance: Acceptable difference between counts (default: 0 for exact match)

    Returns:
        bool: True if counts match within tolerance, False otherwise
    """
    try:
        # File paths
        historical_db_file = Path(db_files_dir) / f'{match_keyword}_historical_hash.db'
        deduped_parquet_file = Path(db_files_dir) / f'{match_keyword}_all_deduped.parquet'

        # Check if files exist
        if not historical_db_file.exists():
            logger.warning(f'SQLite database not found: {historical_db_file}')
            return False

        if not deduped_parquet_file.exists():
            logger.warning(f'Deduped parquet file not found: {deduped_parquet_file}')
            return False

        # Get SQLite count
        sqlite_count = get_sqlite_hash_count(historical_db_file, logger)
        if sqlite_count is None:
            return False

        # Get parquet count
        parquet_count = get_parquet_count(deduped_parquet_file, logger)
        if parquet_count is None:
            return False

        # Compare counts
        difference = abs(sqlite_count - parquet_count)

        if difference <= tolerance:
            logger.info(f'[PASS] Count validation passed: SQLite={sqlite_count:,}, Parquet={parquet_count:,}')
            return True
        else:
            logger.warning(
                f'[FAIL] Count validation failed: SQLite={sqlite_count:,}, Parquet={parquet_count:,}, Diff={difference:,}'
            )
            return False

    except Exception as e:
        logger.error(f'Error during count validation: {e}')
        return False


def get_sqlite_hash_count(db_path, logger):
    """
    Get total count of hash records in SQLite database.

    Args:
        db_path: Path to SQLite database
        logger: Logger instance

    Returns:
        int: Count of records, or None if error
    """
    try:
        with sqlite3.connect(db_path, timeout=30) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM hash_keys')
            count = cursor.fetchone()[0]
            logger.debug(f'SQLite hash count: {count:,}')
            return count

    except sqlite3.Error as e:
        logger.error(f'SQLite error getting count from {db_path}: {e}')
        return None
    except Exception as e:
        logger.error(f'Unexpected error getting SQLite count: {e}')
        return None


def get_parquet_count(parquet_path, logger):
    """
    Get total count of records in deduplicated parquet file.

    Args:
        parquet_path: Path to parquet file
        logger: Logger instance

    Returns:
        int: Count of records, or None if error
    """
    try:
        # Use Polars lazy loading for better performance on large files
        df = pl.scan_parquet(parquet_path)
        count = df.select(pl.len()).collect().item()
        logger.debug(f'Parquet record count: {count:,}')
        return count

    except Exception as e:
        logger.error(f'Error reading parquet count from {parquet_path}: {e}')
        return None


def get_detailed_validation_info(db_files_dir, match_keyword, logger):
    """
    Get detailed validation information for troubleshooting.

    Args:
        db_files_dir: Directory containing database and parquet files
        match_keyword: System match keyword
        logger: Logger instance

    Returns:
        dict: Detailed information about counts and file status
    """
    info = {
        'sqlite_count': None,
        'parquet_count': None,
        'sqlite_file_exists': False,
        'parquet_file_exists': False,
        'sqlite_file_size': None,
        'parquet_file_size': None,
        'validation_passed': False,
    }

    try:
        historical_db_file = Path(db_files_dir) / f'{match_keyword}_historical_hash.db'
        deduped_parquet_file = Path(db_files_dir) / f'{match_keyword}_all_deduped.parquet'

        # Check file existence and sizes
        info['sqlite_file_exists'] = historical_db_file.exists()
        info['parquet_file_exists'] = deduped_parquet_file.exists()

        if info['sqlite_file_exists']:
            info['sqlite_file_size'] = historical_db_file.stat().st_size
            info['sqlite_count'] = get_sqlite_hash_count(historical_db_file, logger)

        if info['parquet_file_exists']:
            info['parquet_file_size'] = deduped_parquet_file.stat().st_size
            info['parquet_count'] = get_parquet_count(deduped_parquet_file, logger)

        # Determine validation status
        if (
            info['sqlite_count'] is not None
            and info['parquet_count'] is not None
            and info['sqlite_count'] == info['parquet_count']
        ):
            info['validation_passed'] = True

        logger.debug(f'Detailed validation info: {info}')
        return info

    except Exception as e:
        logger.error(f'Error getting detailed validation info: {e}')
        return info


def log_validation_summary(validation_info, system_name, logger):
    """
    Log a comprehensive validation summary.

    Args:
        validation_info: Dictionary from get_detailed_validation_info()
        system_name: Name of the system being validated
        logger: Logger instance
    """
    logger.info(f'=== Count Validation Summary for {system_name} ===')

    if validation_info['validation_passed']:
        logger.info(f'[PASS] Both counts match at {validation_info["sqlite_count"]:,} records')
    else:
        logger.warning('[FAIL] Count mismatch detected')

    logger.info(
        f'SQLite DB: {validation_info["sqlite_count"]:,} records ({validation_info["sqlite_file_size"]:,} bytes)'
        if validation_info['sqlite_count']
        else 'N/A'
    )

    logger.info(
        f'Parquet:   {validation_info["parquet_count"]:,} records ({validation_info["parquet_file_size"]:,} bytes)'
        if validation_info['parquet_count']
        else 'N/A'
    )

    if (
        validation_info['sqlite_count'] is not None
        and validation_info['parquet_count'] is not None
        and not validation_info['validation_passed']
    ):
        diff = abs(validation_info['sqlite_count'] - validation_info['parquet_count'])
        logger.warning(f'Difference: {diff:,} records')


# Example integration function for your main pipeline
def run_post_processing_validation(domain, system_name, sys_config, function_dirs, cls_dir, logger):
    """
    Run validation after completing all processing for a domain->system.
    This would be called at the end of your main processing loop.

    Args:
        domain: Domain name
        system_name: System name
        sys_config: System configuration
        function_dirs: Function directories configuration
        cls_dir: Base class directory
        logger: Logger instance
    """
    try:
        logger.info(f'Running post-processing validation for {domain}::{system_name}')

        # Get paths
        domain_dir = Path(cls_dir) / domain
        db_files_dir = domain_dir / function_dirs['db_files'] / system_name
        match_keyword = sys_config['match_keyword']

        # Run validation
        validation_passed = validate_historical_counts(db_files_dir, match_keyword, logger)

        if validation_passed:
            logger.info(f'[PASS] Validation completed successfully for {domain}::{system_name}')
        else:
            # Get detailed info for troubleshooting
            detailed_info = get_detailed_validation_info(db_files_dir, match_keyword, logger)
            log_validation_summary(detailed_info, f'{domain}::{system_name}', logger)

            # Don't fail the entire process, just log the issue
            logger.warning(f'Validation failed for {domain}::{system_name} - please investigate')

        return validation_passed

    except Exception as e:
        logger.error(f'Error running post-processing validation for {domain}::{system_name}: {e}')
        return False

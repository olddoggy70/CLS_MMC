"""
Historical deduplication module with SQLite operations.
This module handles the heaviest processing: hash creation, SQLite deduplication, and database management.
"""

import sqlite3
from pathlib import Path

import polars as pl

# Import hash utilities
from .hash_utils import create_historical_hash, create_main_record_type


def historical_deduplicate(
    df, hash_historical_columns, historical_db_path, date_str, run_id, logger, date_output_dir, system_name, file_ext, write_func
):
    """
    Optimized historical deduplication with improved performance and error handling.

    Args:
        df: Input DataFrame to deduplicate
        hash_historical_columns: Columns to use for historical hashing
        historical_db_path: Path to SQLite database for hash tracking
        date_str: Processing date string (YYYYMMDD)
        run_id: Run identifier
        logger: Logger instance
        date_output_dir: Output directory path
        system_name: System name for file naming
        file_ext: File extension for duplicates file
        write_func: Write function for duplicates file

    Returns:
        pl.DataFrame: Deduplicated DataFrame with only new unique records
    """
    if df.is_empty():
        logger.warning('Empty DataFrame provided for historical deduplication')
        return df

    if not hash_historical_columns:
        logger.error('No hash columns specified for historical deduplication')
        return df

    # Validate required columns exist
    missing_cols = [col for col in hash_historical_columns if col not in df.columns]
    if missing_cols:
        logger.error(f'Missing required columns for hashing: {missing_cols}')
        return df

    try:
        # Create main_rec_type using hash_utils function
        df = df.with_columns([create_main_record_type()])

        # Create hash key using hash_utils function
        df = df.with_columns([create_historical_hash(hash_historical_columns)])

        # Perform deduplication against historical database
        deduped_df = _deduplicate_with_sqlite(
            df, date_str, run_id, historical_db_path, logger, date_output_dir, system_name, file_ext, write_func
        )

        if deduped_df is not None:
            logger.debug(f'Historical dedup complete. {deduped_df.height} new unique records added to DB.')
            return deduped_df
        else:
            logger.error('Deduplication failed, returning original DataFrame')
            return df

    except Exception as e:
        logger.error(f'Error in historical deduplication: {e}')
        return df


def _deduplicate_with_sqlite(
    data: pl.DataFrame,
    date_str: str,
    run_id: str,
    historical_db_path: str,
    logger,
    date_output_dir,
    system_name,
    file_ext,
    write_func,
):
    """
    Optimized SQLite deduplication with improved performance and error handling.

    Args:
        data: DataFrame with hash_key_historical column
        date_str: Processing date string (YYYYMMDD)
        run_id: Run identifier
        historical_db_path: Path to SQLite database
        logger: Logger instance
        date_output_dir: Output directory path
        system_name: System name for file naming
        file_ext: File extension for duplicates file
        write_func: Write function for duplicates file

    Returns:
        pl.DataFrame: DataFrame containing only new unique records
    """
    if data.is_empty():
        logger.warning('Empty DataFrame provided for SQLite deduplication')
        return data

    if 'hash_key_historical' not in data.columns:
        logger.error("Column 'hash_key_historical' not found in the dataframe")
        return data

    readable_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
    initial_rows = len(data)

    try:
        # Ensure database directory exists
        db_path = Path(historical_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f'Processing {initial_rows:,} rows with historical data for {readable_date}')

        # Optimize SQLite connection settings
        with sqlite3.connect(
            historical_db_path,
            timeout=60,  # Increased timeout
            # isolation_level=None,  # Autocommit mode for better performance
        ) as conn:
            cursor = conn.cursor()

            # Setup database with optimal performance settings
            _setup_database(cursor)

            # Get total records in database before processing
            total_db_records_before = _get_total_db_records(cursor)

            # Insert current batch hashes into temp table
            hash_values = data['hash_key_historical'].to_list()
            _create_temp_table_and_insert(cursor, hash_values, logger)

            # Find duplicates and new records
            duplicate_info = _find_duplicates(cursor)
            new_hashes = _find_new_hashes(cursor)

            # Handle duplicate export if needed
            if duplicate_info:
                _export_duplicates(
                    data, duplicate_info, date_str, run_id, logger, date_output_dir, system_name, file_ext, write_func
                )

            # Update existing records and insert new ones
            updated_count = _update_existing_records(cursor, date_str)
            _insert_new_records(cursor, new_hashes, date_str)
            conn.commit()

            # Filter data for new records only
            historical_deduped = data.filter(pl.col('hash_key_historical').is_in(list(new_hashes)))

            # Get total records in database after processing
            total_db_records_after = _get_total_db_records(cursor)

            # Log comprehensive results
            _log_deduplication_results(
                logger,
                readable_date,
                initial_rows,
                len(duplicate_info),
                updated_count,
                len(new_hashes),
                len(historical_deduped),
                total_db_records_before,
                total_db_records_after,
            )

            # Cleanup
            cursor.execute('DROP TABLE IF EXISTS temp_hashes')

            return historical_deduped

    except sqlite3.OperationalError as e:
        logger.error(f'SQLite operational error for {readable_date}: {e!s}')
        return data
    except sqlite3.DatabaseError as e:
        logger.error(f'SQLite database error for {readable_date}: {e!s}')
        return data
    except Exception as e:
        logger.error(f'Unexpected deduplication error for {readable_date}: {e!s}')
        return data


def _setup_database(cursor):
    """
    Setup database tables and optimize SQLite settings for performance.

    Args:
        cursor: SQLite cursor object
    """
    # Optimize SQLite settings for performance
    cursor.execute('PRAGMA journal_mode = WAL')  # Write-Ahead Logging
    cursor.execute('PRAGMA synchronous = NORMAL')  # Balance safety/speed
    cursor.execute('PRAGMA cache_size = -64000')  # 64MB cache
    cursor.execute('PRAGMA temp_store = MEMORY')  # Use memory for temp tables

    # Create main table with proper indexes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hash_keys (
            hash_key_historical TEXT PRIMARY KEY,
            first_date_str TEXT NOT NULL,
            last_date_str TEXT,
            recent_date_str TEXT NOT NULL
        )
    """)

    # Create index if it doesn't exist
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hash_keys_recent_date
        ON hash_keys(recent_date_str)
    """)


def _create_temp_table_and_insert(cursor, hash_values, logger):
    """
    Create temporary table and insert hash values efficiently.

    Args:
        cursor: SQLite cursor object
        hash_values: List of hash values to insert
        logger: Logger instance
    """
    # Create temporary table with better structure
    cursor.execute("""
        CREATE TEMPORARY TABLE temp_hashes (
            hash_key_historical TEXT PRIMARY KEY
        )
    """)

    # Batch insert temp data more efficiently
    hash_tuples = [(hash_val,) for hash_val in hash_values]
    cursor.executemany('INSERT OR IGNORE INTO temp_hashes (hash_key_historical) VALUES (?)', hash_tuples)

    logger.debug(f'Inserted {len(hash_values):,} hash values into temp table')


def _find_duplicates(cursor):
    """
    Find duplicate hash records from historical database.

    Args:
        cursor: SQLite cursor object

    Returns:
        list: List of duplicate record tuples (hash, recent_date, first_date)
    """
    cursor.execute("""
        SELECT
            temp.hash_key_historical,
            hist.recent_date_str,
            hist.first_date_str
        FROM temp_hashes temp
        INNER JOIN hash_keys hist ON temp.hash_key_historical = hist.hash_key_historical
    """)

    return cursor.fetchall()


def _find_new_hashes(cursor):
    """
    Find new hash records not in historical database.

    Args:
        cursor: SQLite cursor object

    Returns:
        set: Set of new hash values
    """
    cursor.execute("""
        SELECT temp.hash_key_historical
        FROM temp_hashes temp
        LEFT JOIN hash_keys hist ON temp.hash_key_historical = hist.hash_key_historical
        WHERE hist.hash_key_historical IS NULL
    """)

    new_hash_rows = cursor.fetchall()
    return {row[0] for row in new_hash_rows}


def _export_duplicates(data, duplicate_info, date_str, run_id, logger, date_output_dir, system_name, file_ext, write_func):
    """
    Export duplicate records to file using configurable format (aligned with daily duplicates pattern).

    Args:
        data: Original DataFrame
        duplicate_info: List of duplicate record information
        date_str: Processing date string
        run_id: Run identifier
        logger: Logger instance
        date_output_dir: Output directory path
        system_name: System name for file naming
        file_ext: File extension
        write_func: Write function
    """
    try:
        duplicates_df = pl.DataFrame(
            {
                'hash_key_historical': [row[0] for row in duplicate_info],
                'last_date_str': [row[1] for row in duplicate_info],
                'first_date_str': [row[2] for row in duplicate_info],
            }
        )

        existing_hashes = {row[0] for row in duplicate_info}
        duplicate_lines = data.filter(pl.col('hash_key_historical').is_in(list(existing_hashes))).join(
            duplicates_df, on='hash_key_historical', how='inner'
        )

        # Use configurable output directory and naming pattern (aligned with daily duplicates)
        duplicate_file = date_output_dir / f'{system_name}_historical_duplicates_{date_str}_{run_id}.{file_ext}'

        # Write using the provided write function
        write_func(duplicate_lines, duplicate_file)

        logger.debug(f'Exported {len(duplicate_lines):,} historical duplicate lines to {duplicate_file}')

    except Exception as e:
        logger.warning(f'Could not export historical duplicates: {e}')


def _update_existing_records(cursor, date_str):
    """
    Update existing hash records with new processing date.

    Args:
        cursor: SQLite cursor object
        date_str: Processing date string

    Returns:
        int: Number of records actually updated
    """
    cursor.execute(
        """
        UPDATE hash_keys
        SET
            last_date_str = recent_date_str,
            recent_date_str = ?
        WHERE hash_key_historical IN (
            SELECT hash_key_historical FROM temp_hashes
        )
        AND recent_date_str < ?
    """,
        (date_str, date_str),
    )

    return cursor.rowcount


def _insert_new_records(cursor, new_hashes, date_str):
    """
    Insert new hash records into historical database.

    Args:
        cursor: SQLite cursor object
        new_hashes: Set of new hash values
        date_str: Processing date string
    """
    if new_hashes:
        new_records = [(hash_key, date_str, None, date_str) for hash_key in new_hashes]
        cursor.executemany(
            """
            INSERT INTO hash_keys
            (hash_key_historical, first_date_str, last_date_str, recent_date_str)
            VALUES (?, ?, ?, ?)
        """,
            new_records,
        )


def _get_total_db_records(cursor):
    """
    Get total number of records in the hash_keys database table.

    Args:
        cursor: SQLite cursor object

    Returns:
        int: Total number of records in the database
    """
    cursor.execute('SELECT COUNT(*) FROM hash_keys')
    return cursor.fetchone()[0]


def _log_deduplication_results(
    logger,
    readable_date,
    initial_rows,
    duplicate_count,
    updated_count,
    new_count,
    deduped_count,
    total_db_records_before,
    total_db_records_after,
):
    """
    Log comprehensive deduplication results.

    Args:
        logger: Logger instance
        readable_date: Human-readable date string
        initial_rows: Initial number of rows
        duplicate_count: Number of duplicates found
        updated_count: Number of records updated in DB
        new_count: Number of new records
        deduped_count: Final deduplicated count
        total_db_records_before: Total records in database before processing
        total_db_records_after: Total records in database after processing
    """
    logger.debug(f'Historical deduplication for {readable_date}:')
    logger.debug(f'  Total records in DB (before): {total_db_records_before:,}')
    logger.debug(f'  Input rows: {initial_rows:,}')
    logger.debug(f'  Duplicates found: {duplicate_count:,}')
    logger.debug(f'  Records updated in DB: {updated_count:,}')
    logger.debug(f'  New records inserted: {new_count:,}')
    logger.debug(f'  Unique rows returned: {deduped_count:,}')
    logger.debug(f'  Total records in DB (after): {total_db_records_after:,}')

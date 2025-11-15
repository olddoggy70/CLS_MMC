"""
Output writer module for handling different file formats and optimized writing.
"""

import os
import zipfile

import polars as pl


def write_and_zip_outputs(
    daily_deduped,
    historical_deduped,
    date_str,
    run_id,
    duplicate_file,
    date_output_dir,
    logger,
    file_ext,
    write_func,
    system_name,
):
    """
    Write individual date outputs with optimized settings for different formats and zip them.

    Args:
        daily_deduped: DataFrame with daily deduplicated data
        historical_deduped: DataFrame with historically deduplicated data
        date_str: Date string in YYYYMMDD format
        run_id: Run identifier
        duplicate_file: Path to duplicates file (for reference)
        date_output_dir: Output directory for date-specific files
        logger: Logger instance
        file_ext: File extension for output files
        write_func: Function to write output files
        system_name: Name of the system being processed
    """
    daily_file = date_output_dir / f'{system_name}_daily_deduped_{date_str}_{run_id}.{file_ext}'
    historical_file = date_output_dir / f'{system_name}_historical_deduped_{date_str}_{run_id}.{file_ext}'

    # Use optimized writing for parquet files

    # daily_deduped = daily_deduped.drop(['hash_key_daily'])
    # historical_deduped = historical_deduped.drop(
    #     ['hash_key_daily', 'hash_key_historical']
    # )  # drop 2 hash key to keep the consitency

    if file_ext == 'parquet':
        write_optimized_parquet(daily_deduped, daily_file, logger)
        write_optimized_parquet(historical_deduped, historical_file, logger)
    else:
        write_func(daily_deduped, daily_file)
        write_func(historical_deduped, historical_file)

    logger.debug(f'Wrote daily deduped: {daily_file} ({daily_deduped.height} rows)')
    logger.debug(f'Wrote historical deduped: {historical_file} ({historical_deduped.height} rows)')

    # Simple zip logic: find all output files for this system/date/run and zip them
    zip_file_path = date_output_dir / f'{system_name}_{date_str}_{run_id}.zip'

    try:
        # Define all possible output files (the 4 types you mentioned)
        possible_files = [
            date_output_dir / f'{system_name}_daily_deduped_{date_str}_{run_id}.{file_ext}',
            date_output_dir / f'{system_name}_historical_deduped_{date_str}_{run_id}.{file_ext}',
            date_output_dir / f'{system_name}_daily_duplicates_{date_str}_{run_id}.{file_ext}',
            date_output_dir / f'{system_name}_historical_duplicates_{date_str}_{run_id}.{file_ext}',
        ]

        # Find which files actually exist
        files_to_zip = [f for f in possible_files if f.exists()]

        if files_to_zip:
            # Create the zip file
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files_to_zip:
                    zipf.write(file_path, file_path.name)
                    logger.debug(f'Added {file_path.name} to zip')

            logger.debug(f'Created zip file: {zip_file_path} with {len(files_to_zip)} files')

            # Remove the original files after successful zipping
            for file_path in files_to_zip:
                try:
                    os.remove(file_path)
                    logger.debug(f'Removed original file: {file_path.name}')
                except Exception as e:
                    logger.warning(f'Failed to remove original file {file_path}: {e}')

            logger.debug(f'Cleaned up {len(files_to_zip)} original output files')

        else:
            logger.warning(f'No output files found to zip for {system_name}_{date_str}_{run_id}')

    except Exception as e:
        logger.error(f'Error creating zip file {zip_file_path}: {e}')
        # Continue execution even if zipping fails


def write_optimized_parquet(df, file_path, logger):
    """
    Write parquet files with optimized settings for better performance.

    Args:
        df: DataFrame to write
        file_path: Path where to write the file
        logger: Logger instance
    """
    try:
        df.write_parquet(
            file_path,
            compression='snappy',  # Fast compression, good balance of speed/size
            row_group_size=25000,  # Smaller row groups for individual files
            use_pyarrow=True,  # Better performance
            statistics=True,  # Enable statistics for better query performance
        )
    except Exception as e:
        logger.warning(f'Failed optimized parquet write, falling back to default: {e}')
        df.write_parquet(file_path)  # Fallback to default settings


def create_system_output_zip(output_files, zip_path, logger, remove_originals=False):
    """
    Utility function to create a zip file from a list of output files.

    Args:
        output_files: List of file paths to include in zip
        zip_path: Path where to create the zip file
        logger: Logger instance
        remove_originals: Whether to remove original files after zipping

    Returns:
        bool: True if zip was created successfully, False otherwise
    """
    try:
        existing_files = [(f, os.path.basename(f)) for f in output_files if os.path.exists(f)]

        if not existing_files:
            if logger:
                logger.warning('No existing files found to zip')
            return False

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arc_name in existing_files:
                zipf.write(file_path, arc_name)
                if logger:
                    logger.debug(f'Added {arc_name} to zip')

        if logger:
            logger.info(f'Created zip file: {zip_path} with {len(existing_files)} files')

        # Remove original files if requested
        if remove_originals:
            for file_path, _ in existing_files:
                try:
                    os.remove(file_path)
                    if logger:
                        logger.debug(f'Removed original file: {file_path}')
                except Exception as e:
                    if logger:
                        logger.warning(f'Failed to remove original file {file_path}: {e}')

        return True

    except Exception as e:
        if logger:
            logger.error(f'Error creating zip file {zip_path}: {e}')
        return False

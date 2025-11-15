"""
Spending Check Module with System-Specific FAISS Cache

Performs fuzzy matching between processed data and spending data using FAISS
with system-specific cache directories to prevent cache collisions.
"""

from pathlib import Path
from typing import Any

import polars as pl

# Updated import to use the enhanced version with system-specific cache
from utils.enhanced_faiss_cache import vectorized_string_matching_optimized


def check_spending_with_config(
    df: pl.DataFrame, spending_config: dict[str, Any], domain_dir: str, system_name: str, logger
) -> pl.DataFrame:
    """
    Main spending check function that uses configuration with system-specific caching.

    Args:
        df: DataFrame to check against spending data
        spending_config: Spending configuration from config.json
        domain_dir: Domain directory path
        system_name: System name for logging and cache isolation
        logger: Logger instance

    Returns:
        pl.DataFrame: DataFrame with spending match results added
    """
    if not spending_config.get('enabled', False):
        logger.debug('Spending check is disabled')
        return df

    # Get reference files from config
    reference_files = spending_config.get('reference_files', {})

    # Log the reference files being used
    logger.debug(f'Spending check enabled for {system_name} with reference files: {list(reference_files.keys())}')

    # System-specific spending strategies
    if system_name.lower() == 'allscripts' and 'spending_data' in reference_files:
        return _perform_spending_check(df, spending_config, domain_dir, system_name, logger)
    elif 'spending_data' in reference_files:
        # Generic spending check for other systems
        logger.debug(f'Using standard spending check for {system_name}')
        return _perform_spending_check(df, spending_config, domain_dir, system_name, logger)
    else:
        logger.warning(f'No spending strategy defined for system {system_name}')
        logger.debug(f'Available reference files: {list(reference_files.keys())}')
        return df


def _perform_spending_check(
    df: pl.DataFrame, spending_config: dict[str, Any], domain_dir: str, system_name: str, logger
) -> pl.DataFrame:
    """
    Perform spending check using configuration with system-specific caching.

    Args:
        df: DataFrame to check against spending data
        spending_config: Spending configuration from config.json
        domain_dir: Domain directory path
        system_name: System name for cache isolation
        logger: Logger instance

    Returns:
        pl.DataFrame: DataFrame with spending match results added
    """
    try:
        # Get spending file path from config
        reference_files = spending_config.get('reference_files', {})
        spending_file_rel = reference_files.get('spending_data')

        if not spending_file_rel:
            logger.warning('No spending_data file specified in spending config')
            return df

        spending_file = Path(domain_dir) / spending_file_rel

        # Check if spending file exists
        if not spending_file.exists():
            logger.warning(f'Spending file not found: {spending_file}')
            return df

        logger.debug(f'Loading spending file for matching: {spending_file}')

        # Get FAISS fuzzy matching parameters from config
        fuzzy_params = spending_config.get('fuzzy_matching', {})
        score_cutoff = fuzzy_params.get('score_cutoff', 90)
        length_tolerance = fuzzy_params.get('length_tolerance', 0.25)
        faiss_k = fuzzy_params.get('faiss_k', 10)
        faiss_dim = fuzzy_params.get('faiss_dim', 16)

        # NEW: Get cache configuration
        cache_base_dir = None
        if 'cache_directory' in reference_files:
            # Use cache directory from config if specified
            cache_base_dir = str(Path(domain_dir) / reference_files['cache_directory'])
            logger.debug(f'Using configured cache directory: {cache_base_dir}')
        else:
            logger.debug('No cache directory configured, using default location')

        logger.debug(
            f'Performing FAISS fuzzy matching with parameters: '
            f'score_cutoff={score_cutoff}, length_tolerance={length_tolerance}, '
            f'faiss_k={faiss_k}, faiss_dim={faiss_dim}, system={system_name}'
        )

        return _perform_fuzzy_spending_match(
            df, str(spending_file), score_cutoff, length_tolerance, faiss_k, faiss_dim, cache_base_dir, system_name, logger
        )

    except Exception as e:
        logger.error(f'Error during spending check: {e!s}')
        logger.warning('Returning original data without spending matches')
        return df


def _perform_fuzzy_spending_match(
    historical_deduped: pl.DataFrame,
    spending_file_path: str,
    score_cutoff: int,
    length_tolerance: float,
    faiss_k: int,
    faiss_dim: int,
    cache_base_dir: str,
    system_name: str,
    logger,
) -> pl.DataFrame:
    """
    Perform fuzzy matching with spending data using FAISS with system-specific cache.

    Args:
        historical_deduped: DataFrame to match against spending data
        spending_file_path: Path to the spending data file
        score_cutoff: Minimum score for matches (0-100)
        length_tolerance: Length tolerance for candidate filtering
        faiss_k: Number of candidates to retrieve from FAISS
        faiss_dim: Vector dimension for FAISS encoding
        cache_base_dir: Base directory for cache (from config)
        system_name: System name for cache isolation
        logger: Logger instance

    Returns:
        pl.DataFrame: DataFrame with fuzzy match results
    """
    try:
        logger.debug(f'Creating Match_Key column for fuzzy matching ({system_name})')

        # Create a new 'Match_Key' column: use 'Distributor Part Number' if non-null/non-empty, else 'Vendor Part Number'
        historical_deduped = historical_deduped.with_columns(
            pl.when(pl.col('Distributor Part Number').is_not_null() & (pl.col('Distributor Part Number') != ''))
            .then(pl.col('Distributor Part Number'))
            .otherwise(pl.col('Vendor Part Number'))
            .alias('Match_Key')
        )

        # Extract s1 from 'Match_Key', ensuring unique non-null values
        # Filter with line rows to prevent header rows for spending check
        logger.debug(f'Extracting unique Match_Key values from Line records ({system_name})')
        s1 = historical_deduped.filter(pl.col('Record Type') == 'Line')['Match_Key'].unique()
        s1 = s1.filter(s1.is_not_null())

        if s1.len() == 0:
            logger.warning(f'No valid Match_Key values found in Line records ({system_name})')

            # Add columns [Matched_String] with value blank and [Similarity_Score] with value 0.0 to align the structure
            historical_deduped = historical_deduped.with_columns(
                Matched_String=pl.lit(''), Similarity_Score=pl.lit(0.0, dtype=pl.Float64)
            )

            return historical_deduped

        # Load spending data and prepare s2
        logger.debug(f'Loading spending data from: {spending_file_path} ({system_name})')

        # Handle different file formats
        if spending_file_path.endswith('.parquet'):
            spend_df = pl.read_parquet(spending_file_path)
        elif spending_file_path.endswith(('.xlsx', '.xls')):
            spend_df = pl.read_excel(spending_file_path, infer_schema_length=0, sheet_id=1)
        elif spending_file_path.endswith('.csv'):
            spend_df = pl.read_csv(spending_file_path, infer_schema_length=0)
        else:
            logger.warning(f'Unsupported spending file format: {spending_file_path} ({system_name})')
            return historical_deduped

        # Extract VPN column for matching
        if 'VPN' not in spend_df.columns:
            logger.warning(f'VPN column not found in spending data ({system_name})')
            return historical_deduped

        s2 = spend_df['VPN'].unique()  # Get VPN from spending file
        s2 = s2.filter(s2.is_not_null())  # Remove null rows

        if s2.len() == 0:
            logger.warning(f'No valid VPN values found in spending data ({system_name})')
            return historical_deduped

        logger.debug(f'Performing fuzzy matching: {s1.len()} unique keys vs {s2.len()} spending VPNs ({system_name})')

        # Debug parameters
        logger.debug(f'Parameters for {system_name}: score_cutoff={score_cutoff}, faiss_k={faiss_k}')

        # NEW: Perform FAISS-based fuzzy matching with system-specific cache and logger
        results_df = vectorized_string_matching_optimized(
            s1,
            s2,
            score_cutoff=score_cutoff,
            length_tolerance=length_tolerance,
            faiss_k=faiss_k,
            faiss_dim=faiss_dim,
            stable_ties=True,
            adaptive_k=True,
            use_disk_cache=True,
            cache_base_dir=cache_base_dir,  # Pass cache base directory
            system_name=system_name,  # Pass system name for cache isolation
            logger=logger,  # Pass logger for proper logging
        )

        if results_df.height == 0:
            logger.debug(f'No fuzzy matches found above the score cutoff ({system_name})')
            historical_deduped = historical_deduped.with_columns(
                Matched_String=pl.lit(''), Similarity_Score=pl.lit(0.0, dtype=pl.Float64)
            )
            return historical_deduped

        # # Optional: Save results for debugging (with system name in filename)
        # debug_results_path = f'C:\\Projects\\data-science\\CLS\\contracts\\db_files\\{system_name}\\results.csv'
        # try:
        #     Path(debug_results_path).parent.mkdir(parents=True, exist_ok=True)
        #     results_df.write_csv(debug_results_path)
        #     logger.debug(f'Debug results saved to: {debug_results_path}')
        # except Exception as e:
        #     logger.debug(f'Could not save debug results: {e}')

        # Join results back to historical_deduped using 'Match_Key'
        actual_matches = results_df.filter(pl.col('Matched_String').is_not_null()).height
        logger.debug(
            f'Found {actual_matches} actual matches out of {results_df.height} queries for {system_name}, joining back to main dataset'
        )

        matched_df = historical_deduped.join(results_df, left_on='Match_Key', right_on='Original_String', how='left')

        # Log match statistics
        total_records = matched_df.height
        matched_records = matched_df.filter(pl.col('Matched_String').is_not_null()).height
        match_percentage = (matched_records / total_records * 100) if total_records > 0 else 0

        logger.debug(
            f'Spending check completed for {system_name}: {matched_records}/{total_records} records matched ({match_percentage:.1f}%)'
        )

        return matched_df

    except Exception as e:
        logger.error(f'Error during spending check for {system_name}: {e!s}')
        logger.warning('Returning original data without spending matches')
        return historical_deduped

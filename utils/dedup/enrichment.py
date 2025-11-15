"""
Enrichment module for deduplication pipeline

Handles data enrichment with configurable reference files and business logic.
"""

from pathlib import Path

import polars as pl


def enrich_data_with_config(daily_deduped: pl.DataFrame, enrichment_config: dict, domain_dir: str, system_name: str, logger):
    """
    Main enrichment function that uses configuration to determine enrichment logic.

    Args:
        daily_deduped: DataFrame to enrich
        enrichment_config: Enrichment configuration from config.json
        domain_dir: Domain directory path
        system_name: System name for logging
        logger: Logger instance

    Returns:
        pl.DataFrame: Enriched DataFrame
    """
    if not enrichment_config.get('enabled', False):
        logger.debug('Enrichment is disabled')
        return daily_deduped

    # Get reference files from config
    reference_files = enrichment_config.get('reference_files', {})

    # Log the reference files being used
    logger.debug(f'Enrichment enabled for {system_name} with reference files: {list(reference_files.keys())}')

    # System-specific enrichment strategies
    if system_name.lower() == 'allscripts' and 'vendor_master' in reference_files:
        return _enrich_with_shared_vendor_master(daily_deduped, reference_files, domain_dir, logger)
    elif 'vendor_master' in reference_files:
        # Generic enrichment for other systems that use shared vendor_master
        logger.info(f'Using shared vendor_master enrichment for {system_name}')
        return _enrich_with_shared_vendor_master(daily_deduped, reference_files, domain_dir, logger)
    else:
        logger.warning(f'No enrichment strategy defined for system {system_name}')
        logger.debug(f'Available reference files: {list(reference_files.keys())}')
        return daily_deduped


def _enrich_with_shared_vendor_master(
    daily_deduped: pl.DataFrame, reference_files: dict, domain_dir: str, logger
) -> pl.DataFrame:
    """
    Enrich daily_deduped data using the shared CLS_Awarded_Vendor_to_Distributor.xlsx file.
    Handles Distributor Part Number and MMC Distributor No. enrichment.

    Args:
        daily_deduped: DataFrame to enrich
        reference_files: Dictionary of reference file paths from config
        domain_dir: Domain directory path
        logger: Logger instance

    Returns:
        pl.DataFrame: Enriched DataFrame
    """
    try:
        # Get shared contract file path from config (CLS_Awarded_Vendor_to_Distributor.xlsx)
        contract_file_rel = reference_files.get('vendor_master')
        if not contract_file_rel:
            logger.warning('No vendor_master file specified in enrichment config')
            return daily_deduped

        contract_file = Path(domain_dir) / contract_file_rel

        # Check if contract file exists
        if not contract_file.exists():
            logger.warning(f'Shared contract file not found: {contract_file}')
            return daily_deduped

        logger.debug(f'Loading shared contract file for enrichment: {contract_file}')

        # Get unique BC contract with 1 Contract Vendor for distributor check rows
        # from shared CLS - Awarded Vendor to Distributor file
        df1 = pl.read_excel(str(contract_file), infer_schema_length=0, sheet_id=1)

        # Count occurrences of each value in the Contract Number column
        counts = df1.group_by('Contract Number').agg(pl.len().alias('count'))

        # Join counts back to the original DataFrame
        df_with_counts = df1.join(counts, on='Contract Number')

        # Filter to keep only rows where the count is 1 (unique values) and distributor Check is 1 Contract Vendor
        unique_contract_rows = df_with_counts.filter(
            (pl.col('count') == 1) & (pl.col('Distributor Check') == '1 Contract Vendor')
        )
        unique_contract_rows_list = unique_contract_rows['Contract Number'].to_list()

        # Check if MMC Distributor No. is blank
        mmc_dis_no_blank = (pl.col('MMC Distributor No.').is_null()) | (pl.col('MMC Distributor No.').str.strip_chars().eq(''))

        # Check if Distributor Part Number is blank
        dpn_blank = (pl.col('Distributor Part Number').is_null()) | (pl.col('Distributor Part Number').str.strip_chars().eq(''))

        # Define conditions
        cond1 = (
            (pl.col('Record Type') == 'Line')
            & dpn_blank
            & (pl.col('MMC Distributor No.') == pl.col('MMC Awarded Vendor No.'))
            & (~mmc_dis_no_blank)
        )

        cond2 = (
            (pl.col('Record Type') == 'Line')
            & dpn_blank
            & mmc_dis_no_blank
            & pl.col('Business Central Contract Number').is_in(unique_contract_rows_list)
        )

        # Apply enrichment logic
        enriched_daily_deduped = daily_deduped.with_columns(
            [
                pl.when(cond1 | cond2)
                .then(pl.col('Vendor Part Number'))  # fulfill by VPN
                .otherwise(pl.col('Distributor Part Number'))
                .alias('Distributor Part Number'),
                pl.when(cond2)
                .then(pl.col('MMC Awarded Vendor No.'))  # fulfill by MMC Awarded Vendor No
                .otherwise(pl.col('MMC Distributor No.'))
                .alias('MMC Distributor No.'),
                (cond1 | cond2).alias('Enriched'),
            ]
        )

        # Log enrichment results
        enriched_count = enriched_daily_deduped.filter(pl.col('Enriched')).height
        total_count = enriched_daily_deduped.height

        logger.debug(f'Enrichment completed: {enriched_count} rows enriched of total {total_count}')

        return enriched_daily_deduped

    except Exception as e:
        logger.error(f'Error during enrichment: {e!s}')
        logger.warning('Returning original data without enrichment')
        return daily_deduped


def enrich_data(daily_deduped: pl.DataFrame, logger) -> pl.DataFrame:
    """
    Legacy enrichment function for backward compatibility.
    This is the original function that was called by daily_dedup.py

    Args:
        daily_deduped: DataFrame to enrich
        logger: Logger instance

    Returns:
        pl.DataFrame: DataFrame (unchanged, as this is just for compatibility)
    """
    logger.debug('Legacy enrich_data function called - no enrichment applied')
    return daily_deduped

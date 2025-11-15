import argparse
import json
import logging
import os
from datetime import datetime

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from utils.dedup_pipeline_v14 import run_dedup_pipeline
from utils.file_utils import (
    archive_processed_daily_files,
    backup_files,
    check_daily_files,
    cleanup_old_backups,
    process_zip_files,
)
from utils.validation_utils import run_post_processing_validation


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


def setup_orchestrator_logger(cls_dir):
    log_dir = os.path.join(cls_dir, 'orchestrator_logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'dedup_main_{datetime.today().strftime("%Y-%m-%d")}.log')

    logger = logging.getLogger('orchestrator')
    logger.setLevel(logging.DEBUG)  # capture DEBUG for file

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


def main():
    # ANSI color codes
    COLOR_BLUE = '\033[94m'  # bright blue
    COLOR_GREEN = '\033[92m'  # bright green
    COLOR_YELLOW = '\033[93m'  # yellow
    COLOR_RESET = '\033[0m'

    CLS_DIR = os.path.dirname(os.path.abspath(__file__))
    logger = setup_orchestrator_logger(CLS_DIR)
    logger.info('Main deduplication log started')

    parser = argparse.ArgumentParser(description='Daily deduplication with extraction.')
    parser.add_argument('--domains', nargs='+', help='Domains to process (e.g. contracts, items). If omitted, process all.')
    parser.add_argument('--systems', nargs='+', help='Systems to process. If omitted, process all in selected domains.')
    parser.add_argument('--config', default=os.path.join(CLS_DIR, 'config', 'systems_config.json'), help='Path to config JSON.')
    args = parser.parse_args()

    config = load_config(args.config)
    function_dirs = config['global']['function_dirs']
    global_hash_columns = config['global']
    retention_days = config['global'].get('retention_days', 5)

    selected_domains = args.domains if args.domains else [d for d in config if d not in ['global']]
    selected_systems = args.systems

    logger.info(f'Domains selected: {selected_domains}')
    logger.info(f'Systems selected: {selected_systems if selected_systems else "All systems in selected domains"}')

    # Step 0. Process zip files per domain
    with logging_redirect_tqdm():
        for domain in selected_domains:
            # Display domain header
            tqdm.write(f'{COLOR_BLUE}=== Domain: {domain} ==={COLOR_RESET}')

            domain_dir = os.path.join(CLS_DIR, domain)
            zip_source_dir = os.path.join(CLS_DIR, 'daily_zipped')

            # Process zip files with enhanced error handling and messaging
            process_zip_files(zip_source_dir, config, domain, function_dirs, CLS_DIR, logger)

            # Process systems
            systems = config[domain].items()
            for system_name, sys_config in systems:
                if selected_systems and system_name not in selected_systems:
                    continue

                # Check daily files directory and validate files
                daily_files_dir = os.path.join(domain_dir, function_dirs['daily_files'], system_name)
                daily_dir_exists, daily_files = check_daily_files(daily_files_dir, logger)

                if not daily_dir_exists:
                    # tqdm.write(f'[WARNING] Daily files directory does not exist for {domain}::{system_name}')
                    logger.warning(f'Skipping {domain}::{system_name} - daily files directory missing')
                    continue

                if not daily_files:
                    # tqdm.write(f'[INFO] No files to process in daily_files for {domain}::{system_name}')
                    logger.info(f'No files to process for {domain}::{system_name}')
                    continue

                # tqdm.write(f'[INFO] Processing {len(daily_files)} file(s) for {domain}::{system_name}: {", ".join(daily_files)}')
                tqdm.write(f'[INFO] Processing {len(daily_files)} file(s) for {domain}::{system_name}')

                # Setup database and backup directories
                db_files_dir = os.path.join(domain_dir, function_dirs['db_files'], system_name)
                os.makedirs(db_files_dir, exist_ok=True)

                match_keyword = sys_config['match_keyword']
                historical_db_file = f'{match_keyword}_historical_hash.db'
                deduped_parquet_file = f'{match_keyword}_all_deduped.parquet'

                backup_dir = os.path.join(db_files_dir, 'backups')
                os.makedirs(backup_dir, exist_ok=True)

                # Use existing backup functions
                backup_files(
                    src_dir=db_files_dir,
                    backup_dir=backup_dir,
                    file_patterns=[historical_db_file, deduped_parquet_file],
                    add_date_suffix=True,
                )
                cleanup_old_backups(backup_dir, retention_days=retention_days)

                # Run dedup pipeline
                try:
                    run_dedup_pipeline(
                        sys_config, domain, system_name, function_dirs, global_hash_columns, CLS_DIR, orchestrator_logger=logger
                    )
                    tqdm.write(f'[OK] Completed dedup pipeline for {domain}::{system_name}')

                    # NEW: Run post-processing validation
                    try:
                        validation_passed = run_post_processing_validation(
                            domain=domain,
                            system_name=system_name,
                            sys_config=sys_config,
                            function_dirs=function_dirs,
                            cls_dir=CLS_DIR,
                            logger=logger,
                        )

                        if validation_passed:
                            tqdm.write(f'[OK] Validation passed for {domain}::{system_name}')
                        else:
                            tqdm.write(f'[WARNING] Validation failed for {domain}::{system_name}')

                    except Exception as e:
                        logger.error(f'Validation error for {domain}::{system_name}: {e}')
                        tqdm.write(f'[WARNING] Could not validate counts for {domain}::{system_name}')

                    # Archive processed daily files after successful pipeline run
                    archive_processed_daily_files(daily_files_dir, logger)

                except Exception as e:
                    logger.error(f'Error running dedup pipeline for {domain}::{system_name}: {e!s}')
                    tqdm.write(f'[ERROR] Failed dedup pipeline for {domain}::{system_name}: {e!s}')
                    continue

    logger.info('Main deduplication process completed.')


if __name__ == '__main__':
    main()

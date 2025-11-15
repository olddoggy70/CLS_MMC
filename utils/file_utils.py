import logging
import os
import re
import shutil
import time
import zipfile
from datetime import datetime

from tqdm import tqdm


def backup_files(src_dir, backup_dir, file_patterns=None, add_date_suffix=True):
    os.makedirs(backup_dir, exist_ok=True)
    today_str = datetime.today().strftime('%Y%m%d')

    for file_name in os.listdir(src_dir):
        if file_patterns and not any(p in file_name for p in file_patterns):
            continue

        src_file = os.path.join(src_dir, file_name)
        if not os.path.isfile(src_file):
            continue

        name, ext = os.path.splitext(file_name)
        backup_file_name = f'{name}_{today_str}{ext}' if add_date_suffix else file_name
        dest_file = os.path.join(backup_dir, backup_file_name)
        shutil.copy2(src_file, dest_file)
        logging.info(f'Backed up {file_name} to {dest_file}')


def cleanup_old_backups(backup_dir, retention_days=5):
    now = time.time()
    cutoff = now - (retention_days * 86400)

    if not os.path.exists(backup_dir):
        logging.info(f'Backup directory does not exist: {backup_dir}')
        return

    for entry in os.listdir(backup_dir):
        path = os.path.join(backup_dir, entry)
        logging.info(f'Checking old backup file: {path}')
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                logging.info(f'Deleted old backup file: {path}')
            elif os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path)
                logging.info(f'Deleted old backup folder: {path}')
        except Exception as e:
            logging.warning(f'Failed to delete {path}: {e}')


def archive_processed_zip_file(zip_file_path, archive_root_dir):
    if not os.path.exists(zip_file_path):
        logging.warning(f'Zip file not found for archiving: {zip_file_path}')
        return

    file_name = os.path.basename(zip_file_path)
    name, ext = os.path.splitext(file_name)

    match = re.search(r'(\d{8})$', name)
    if match:
        file_date = match.group(1)
    else:
        logging.warning(f'No date found in filename: {file_name}. Skipping archive.')
        return

    archive_dir = os.path.join(archive_root_dir, file_date)
    os.makedirs(archive_dir, exist_ok=True)

    archived_file_path = os.path.join(archive_dir, file_name)
    shutil.move(zip_file_path, archived_file_path)
    logging.info(f'Archived {file_name} to {archived_file_path}')


def extract_contract_files_from_zip(zip_path, domain_configs, function_dirs, BASE_DIR, domain):
    if not os.path.exists(zip_path):
        logging.warning(f'Zip file not found: {zip_path}')
        return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if 'contract' in file_name.lower() and file_name.endswith('.xlsx'):
                matched_system = None
                for system_name, config in domain_configs.items():
                    match_keyword = config.get('match_keyword', '').lower()
                    if match_keyword and match_keyword in file_name.lower():
                        matched_system = system_name
                        break

                if matched_system:
                    daily_files_dir = os.path.join(BASE_DIR, domain, function_dirs['daily_files'], matched_system)
                    os.makedirs(daily_files_dir, exist_ok=True)
                    zip_ref.extract(file_name, daily_files_dir)
                    logging.info(f'Extracted {file_name} to {daily_files_dir} for {domain}/{matched_system}')
                else:
                    logging.warning(f'No matching system found for file in zip: {file_name}')


def get_system_logger(domain_dir, domain, system_name):
    """
    Logger for a specific domain/system.
    DEBUG to file only, INFO+ to file AND propagates to orchestrator console.
    """
    today_str = datetime.today().strftime('%Y%m%d')
    log_dir = os.path.join(domain_dir, 'system_logs', system_name)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'{system_name}_dedup_{today_str}.log')

    # Child logger under orchestrator hierarchy
    logger_name = f'orchestrator.{domain}.{system_name}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler for ALL levels
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [%(module)s] - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        # Console handler for INFO+ (goes to orchestrator's parent)
        # This mimics what propagation would do
        orchestrator_logger = logging.getLogger('orchestrator')

        class PropagateInfoHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.INFO:
                    # Send to orchestrator's handlers
                    orchestrator_logger.handle(record)

        propagate_handler = PropagateInfoHandler()
        logger.addHandler(propagate_handler)

    logger.propagate = False  # We handle everything locally
    return logger


# NEW FUNCTIONS FOR ENHANCED PROCESSING


def process_zip_files(zip_source_dir, config, domain, function_dirs, cls_dir, logger):
    """Process zip files and extract to daily_files directory."""
    archive_root_dir = os.path.join(zip_source_dir, 'archive')
    zip_files_found = False

    # Check if zip source directory exists
    if not os.path.exists(zip_source_dir):
        logger.warning(f'Zip source directory does not exist: {zip_source_dir}')
        # tqdm.write(f'[WARNING] Zip source directory does not exist: {zip_source_dir}')
        return zip_files_found

    # Look for zip files
    zip_files = [f for f in os.listdir(zip_source_dir) if f.endswith('.zip')]

    if not zip_files:
        logger.debug(f'No zip files found in {zip_source_dir}')
        # tqdm.write(f'[INFO] No zip files found in {zip_source_dir}')
    else:
        zip_files_found = True
        logger.info(f'Found {len(zip_files)} zip file(s) to process: {", ".join(zip_files)}')
        # tqdm.write(f'[INFO] Found {len(zip_files)} zip file(s) to process: {", ".join(zip_files)}')

        for file_name in zip_files:
            try:
                zip_path = os.path.join(zip_source_dir, file_name)
                logger.info(f'Processing zip file: {file_name}')
                # tqdm.write(f'[PROCESSING] Extracting {file_name}...')

                # Use existing functions
                extract_contract_files_from_zip(zip_path, config[domain], function_dirs, cls_dir, domain)
                archive_processed_zip_file(zip_path, archive_root_dir)

                logger.info(f'Successfully processed and archived: {file_name}')
                # tqdm.write(f'[OK] Successfully processed and archived: {file_name}')

            except Exception as e:
                logger.error(f'Error processing zip file {file_name}: {e!s}')
                # tqdm.write(f'[ERROR] Failed to process {file_name}: {str(e)}')
                continue

    return zip_files_found


def check_daily_files(daily_files_dir, logger):
    """Check if daily files directory exists and has files."""
    if not os.path.exists(daily_files_dir):
        logger.warning(f'Daily files directory does not exist: {daily_files_dir}')
        return False, []

    # Get all files (not directories)
    files = [
        f for f in os.listdir(daily_files_dir) if os.path.isfile(os.path.join(daily_files_dir, f)) and not f.startswith('.')
    ]  # Exclude hidden files

    if not files:
        logger.info(f'No files found in daily files directory: {daily_files_dir}')
        return True, []
    else:
        logger.info(f'Found {len(files)} file(s) in daily files directory')
        return True, files


def archive_processed_daily_files(daily_files_dir, logger):
    """Archive processed files from daily_files directory."""
    try:
        if not os.path.exists(daily_files_dir):
            logger.warning(f'Daily files directory does not exist: {daily_files_dir}')
            return

        # Get list of files to archive (exclude directories and hidden files)
        files_to_archive = [
            f for f in os.listdir(daily_files_dir) if os.path.isfile(os.path.join(daily_files_dir, f)) and not f.startswith('.')
        ]

        if not files_to_archive:
            logger.info(f'No files to archive in: {daily_files_dir}')
            return

        # Create simple archive directory
        archive_dir = os.path.join(daily_files_dir, 'archive')
        os.makedirs(archive_dir, exist_ok=True)

        # Move files to archive (overwrites existing files)
        archived_count = 0
        for file_name in files_to_archive:
            try:
                src_path = os.path.join(daily_files_dir, file_name)
                dst_path = os.path.join(archive_dir, file_name)

                shutil.move(src_path, dst_path)
                archived_count += 1
                logger.debug(f'Archived: {file_name}')

            except Exception as e:
                logger.error(f'Error archiving file {file_name}: {e!s}')
                continue

        if archived_count > 0:
            logger.debug(f'Successfully archived {archived_count} file(s) to: {archive_dir}')
            # tqdm.write(f'[OK] Archived {archived_count} processed file(s) to: {os.path.relpath(archive_dir)}')
        else:
            logger.warning('No files were successfully archived')

    except Exception as e:
        logger.error(f'Error during archiving process: {e!s}')
        # tqdm.write(f'[ERROR] Failed to archive processed files: {str(e)}')


def zip_system_daily_outputs(domain_dir, system_name, match_keyword, function_dirs, logger):
    """
    Zip all daily output files for a system into a single zip file.

    Args:
        domain_dir: Path to the domain directory
        system_name: Name of the system
        match_keyword: Match keyword for file naming
        function_dirs: Dictionary containing function directory names
        logger: Logger instance for logging

    Returns:
        str: Path to created zip file if successful, None otherwise
    """
    try:
        today_str = datetime.today().strftime('%Y%m%d')

        # Define potential output file patterns based on your mention of 4 file types
        output_patterns = [
            f'{match_keyword}_daily_dedup*.csv',
            f'{match_keyword}_daily_dup*.csv',
            f'{match_keyword}_historical_dedup*.csv',
            f'{match_keyword}_historical_dup*.csv',
            f'{match_keyword}_daily_dedup*.xlsx',
            f'{match_keyword}_daily_dup*.xlsx',
            f'{match_keyword}_historical_dedup*.xlsx',
            f'{match_keyword}_historical_dup*.xlsx',
        ]

        # Look for output files in various possible locations
        potential_dirs = [
            os.path.join(domain_dir, function_dirs.get('output_files', 'output_files'), system_name),
            os.path.join(domain_dir, function_dirs.get('db_files', 'db_files'), system_name),
            os.path.join(domain_dir, 'output', system_name),
            os.path.join(domain_dir, system_name, 'output'),
            os.path.join(domain_dir, system_name),
        ]

        files_to_zip = []
        source_dirs = []

        # Search for files matching our patterns
        for search_dir in potential_dirs:
            if os.path.exists(search_dir):
                for file_name in os.listdir(search_dir):
                    file_path = os.path.join(search_dir, file_name)
                    if os.path.isfile(file_path):
                        # Check if file matches any of our patterns
                        for pattern in output_patterns:
                            import fnmatch

                            if fnmatch.fnmatch(file_name, pattern):
                                files_to_zip.append((file_path, file_name))
                                if search_dir not in source_dirs:
                                    source_dirs.append(search_dir)
                                break

        if not files_to_zip:
            logger.info(f'No daily output files found to zip for {system_name}')
            return None

        # Create zip file in the first source directory found, or create an outputs directory
        if source_dirs:
            zip_dir = source_dirs[0]
        else:
            zip_dir = os.path.join(domain_dir, function_dirs.get('output_files', 'output_files'), system_name)
            os.makedirs(zip_dir, exist_ok=True)

        zip_filename = f'{match_keyword}_daily_outputs_{today_str}.zip'
        zip_path = os.path.join(zip_dir, zip_filename)

        # Create the zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, file_name in files_to_zip:
                # Add file to zip with just the filename (no directory structure)
                zipf.write(file_path, file_name)
                logger.debug(f'Added {file_name} to zip')

        logger.info(f'Created zip file with {len(files_to_zip)} files: {zip_path}')
        tqdm.write(f'[OK] Created daily outputs zip: {os.path.basename(zip_path)} ({len(files_to_zip)} files)')

        # Optionally remove the original files after zipping
        # Uncomment the following lines if you want to delete the original files:
        """
        for file_path, _ in files_to_zip:
            try:
                os.remove(file_path)
                logger.debug(f'Removed original file: {file_path}')
            except Exception as e:
                logger.warning(f'Failed to remove original file {file_path}: {e}')
        """

        return zip_path

    except Exception as e:
        logger.error(f'Error creating zip file for {system_name}: {e!s}')
        tqdm.write(f'[ERROR] Failed to create zip file for {system_name}: {str(e)}')
        return None


def zip_system_outputs_flexible(search_dirs, system_name, match_keyword, output_dir=None, logger=None):
    """
    Flexible function to zip system output files from specified directories.

    Args:
        search_dirs: List of directories to search for output files
        system_name: Name of the system
        match_keyword: Match keyword for file naming
        output_dir: Directory to save the zip file (defaults to first search dir)
        logger: Logger instance

    Returns:
        str: Path to created zip file if successful, None otherwise
    """
    try:
        today_str = datetime.today().strftime('%Y%m%d')

        # Common output file extensions and patterns
        file_extensions = ['.parquet', '.xlsx', '.csv', '.json']
        output_keywords = ['dedup', 'dup', 'daily', 'historical', 'output']

        files_to_zip = []

        # Search for relevant files
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                if logger:
                    logger.debug(f'Search directory does not exist: {search_dir}')
                continue

            for file_name in os.listdir(search_dir):
                file_path = os.path.join(search_dir, file_name)
                if not os.path.isfile(file_path):
                    continue

                # Check if file is relevant based on match_keyword and common patterns
                file_lower = file_name.lower()
                if (
                    match_keyword.lower() in file_lower
                    and any(ext in file_lower for ext in file_extensions)
                    and any(keyword in file_lower for keyword in output_keywords)
                ):
                    files_to_zip.append((file_path, file_name))
                    if logger:
                        logger.debug(f'Found output file to zip: {file_name}')

        if not files_to_zip:
            if logger:
                logger.info(f'No output files found to zip for {system_name}')
            return None

        # Determine output directory for zip file
        if output_dir is None:
            output_dir = search_dirs[0] if search_dirs else os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        zip_filename = f'{match_keyword}_outputs_{today_str}.zip'
        zip_path = os.path.join(output_dir, zip_filename)

        # Create the zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path, file_name in files_to_zip:
                zipf.write(file_path, file_name)
                if logger:
                    logger.debug(f'Added {file_name} to zip')

        if logger:
            logger.info(f'Created output zip with {len(files_to_zip)} files: {zip_path}')

        return zip_path

    except Exception as e:
        if logger:
            logger.error(f'Error creating zip file for {system_name}: {e!s}')
        return None

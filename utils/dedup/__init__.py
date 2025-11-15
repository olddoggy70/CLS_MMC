"""
Deduplication package for processing daily files with historical tracking.

This package provides modular components for:
- File discovery and reading operations
- Daily deduplication logic
- Historical deduplication with SQLite tracking
- Hash utilities for MD5 operations
- Parquet file management
"""

from .daily_dedup import check_spending, daily_deduplicate, determine_run_id, enrich_data
from .file_operations import discover_and_group_files, read_and_combine_files
from .hash_utils import create_daily_hash, create_historical_hash, create_main_record_type, fast_md5_hash_series
from .historical_dedup import historical_deduplicate

__all__ = [  # noqa: RUF022
    # Hash utilities
    'create_daily_hash',
    'create_historical_hash',
    'create_main_record_type',
    'fast_md5_hash_series',
    # Daily deduplication
    'daily_deduplicate',
    'determine_run_id',
    'enrich_data',
    'check_spending',
    # Historical deduplication
    'historical_deduplicate',
    # File operations
    'discover_and_group_files',
    'read_and_combine_files',
]

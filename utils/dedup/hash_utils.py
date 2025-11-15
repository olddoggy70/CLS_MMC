"""
Hash utilities for deduplication operations.
Provides MD5 hashing functions for daily and historical deduplication.
"""

import hashlib

import polars as pl


def fast_md5_hash_series(series, chunk_size=10000):
    """
    Optimized MD5 hashing with chunked processing for millions of rows.

    Args:
        series: Polars Series containing strings to hash
        chunk_size: Number of rows to process per chunk (default: 10000)

    Returns:
        pl.Series: Series of MD5 hashes (32 characters each)
    """
    values = series.to_list()
    result = []

    for i in range(0, len(values), chunk_size):
        chunk = values[i : i + chunk_size]
        # Pre-allocate list for better performance
        chunk_hashes = []
        for val in chunk:
            chunk_hashes.append(hashlib.md5(val.encode('utf-8')).hexdigest()[:32])
        result.extend(chunk_hashes)

    return pl.Series(result)


def create_daily_hash(hash_columns):
    """
    Create a Polars expression for daily hash generation.

    Args:
        hash_columns: List of column names to include in hash

    Returns:
        pl.Expr: Polars expression that generates MD5 hash
    """

    def row_hash(row):
        concat_str = '|'.join([str(row[col]) for col in hash_columns])
        return hashlib.md5(concat_str.encode()).hexdigest()[:32]

    return pl.struct(hash_columns).map_elements(row_hash, return_dtype=pl.Utf8).alias('hash_key_daily')


def create_historical_hash(hash_columns):
    """
    Create a Polars expression for historical hash generation with optimized performance.

    Args:
        hash_columns: List of column names to include in hash

    Returns:
        pl.Expr: Polars expression that generates MD5 hash
    """
    concat_expr = pl.concat_str([pl.col(col).cast(pl.Utf8) for col in hash_columns], separator='|')
    return concat_expr.map_batches(fast_md5_hash_series, return_dtype=pl.Utf8).alias('hash_key_historical')


def create_main_record_type():
    """
    Create a Polars expression for main record type classification.

    Returns:
        pl.Expr: Polars expression that classifies record types
    """
    return (
        pl.when(pl.col('Record Type') == 'Header')
        .then(pl.lit('H'))
        .when((pl.col('Record Type') == 'Line') & (pl.col('Maintenance Type').str.contains('Line Removal')))
        .then(pl.lit('LR'))
        .when(pl.col('Record Type') == 'Line')
        .then(pl.lit('LN'))
        .otherwise(pl.lit(None))
        .alias('main_rec_type')
    )

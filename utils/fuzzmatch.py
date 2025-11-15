import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process


def fnv1a_hash(token: str) -> int:
    """Compute a deterministic hash using FNV-1a algorithm."""
    FNV_PRIME = 16777619
    FNV_OFFSET = 2166136261
    h = FNV_OFFSET
    for char in token.encode('utf-8'):
        h ^= char
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def compute_minhash_multi(args: tuple[list[str], int, list[tuple[np.ndarray, np.ndarray]], int]) -> list[np.ndarray]:
    """Compute multiple MinHash signatures with different hash functions."""
    tokens, num_hashes, hash_params_list, prime = args
    if not tokens:
        return [np.full(num_hashes, prime, dtype=np.uint32) for _ in hash_params_list]

    token_hashes = np.array([fnv1a_hash(token) for token in tokens], dtype=np.uint32)
    token_hashes_expanded = token_hashes[:, np.newaxis]

    signatures = []
    for a, b in hash_params_list:
        hashes = (token_hashes_expanded * a[np.newaxis, :] + b[np.newaxis, :]) % prime
        signatures.append(np.min(hashes, axis=0).astype(np.uint32))

    return signatures


def get_ngrams(s: str, n: int = 4) -> list[str]:
    """Generate n-grams for a single string."""
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def preprocess_strings(strings: list[str], n_procs: int | None = None) -> list[list[str]]:
    """Parallelized preprocessing to generate n-grams."""
    if n_procs is None:
        n_procs = min(4, os.cpu_count() or 1)

    with Pool(processes=n_procs) as pool:
        return pool.map(get_ngrams, strings)


class AdaptiveMultiSeedMinHashLSH:
    """Optimized MinHash LSH with adaptive multi-seed strategy."""

    def __init__(self, num_hashes: int = 64, num_bands: int = 8, num_seeds: int = 2, seed: int = 42) -> None:
        if num_hashes % num_bands != 0:
            raise ValueError('num_hashes must be divisible by num_bands')

        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.num_seeds = num_seeds
        self.prime = 2**31 - 1
        self.buckets = defaultdict(list)

        # Generate multiple sets of hash functions with better diversity
        self.hash_params = []
        for i in range(num_seeds):
            # Use larger prime offsets for better independence
            np.random.seed(seed + i * 31397)  # Large prime offset
            a = np.random.randint(1, 2**32 - 1, size=num_hashes, dtype=np.uint32)
            b = np.random.randint(1, 2**32 - 1, size=num_hashes, dtype=np.uint32)
            self.hash_params.append((a, b))

    def compute_minhash(self, tokens: list[str]) -> list[np.ndarray]:
        """Compute multiple MinHash signatures."""
        if not tokens:
            return [np.full(self.num_hashes, self.prime, dtype=np.uint32) for _ in range(self.num_seeds)]

        token_hashes = np.array([fnv1a_hash(token) for token in tokens], dtype=np.uint32)
        token_hashes_expanded = token_hashes[:, np.newaxis]

        signatures = []
        for a, b in self.hash_params:
            hashes = (token_hashes_expanded * a[np.newaxis, :] + b[np.newaxis, :]) % self.prime
            signatures.append(np.min(hashes, axis=0).astype(np.uint32))

        return signatures

    def _get_band_hashes(self, signature: np.ndarray) -> list[int]:
        """Convert signature to band hashes."""
        return [hash(tuple(signature[i : i + self.rows_per_band])) for i in range(0, self.num_hashes, self.rows_per_band)]

    def insert_batch(self, signatures_batch: list[list[np.ndarray]]) -> None:
        """Insert batch of multi-signature items."""
        for idx, signatures in enumerate(signatures_batch):
            for seed_idx, signature in enumerate(signatures):
                for band_idx, band_hash in enumerate(self._get_band_hashes(signature)):
                    self.buckets[(seed_idx, band_idx, band_hash)].append(idx)

    def query_adaptive(self, signatures: list[np.ndarray], max_candidates: int = 50, fallback_threshold: int = 3) -> set:
        """
        Adaptive query strategy:
        1. Start with first seed (fastest)
        2. Only use additional seeds if insufficient candidates found
        """
        candidates = set()

        # Phase 1: Query with first seed only
        for band_idx, band_hash in enumerate(self._get_band_hashes(signatures[0])):
            candidates.update(self.buckets.get((0, band_idx, band_hash), []))

        # Phase 2: Always try second seed for better recall, but with early exit
        if self.num_seeds > 1:
            for seed_idx in range(1, self.num_seeds):
                for band_idx, band_hash in enumerate(self._get_band_hashes(signatures[seed_idx])):
                    candidates.update(self.buckets.get((seed_idx, band_idx, band_hash), []))

                # Early exit if we have plenty of candidates
                if len(candidates) >= max_candidates:
                    break

        return candidates

    def get_stats(self) -> dict:
        """Get statistics about the LSH structure."""
        bucket_sizes = [len(v) for v in self.buckets.values()]
        return {
            'num_buckets': len(self.buckets),
            'avg_bucket_size': np.mean(bucket_sizes) if bucket_sizes else 0,
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0,
            'min_bucket_size': min(bucket_sizes) if bucket_sizes else 0,
            'num_seeds': self.num_seeds,
        }


class HybridLSH:
    """Hybrid approach: Single-seed LSH with intelligent fallbacks."""

    def __init__(self, num_hashes: int = 64, num_bands: int = 8, seed: int = 42) -> None:
        if num_hashes % num_bands != 0:
            raise ValueError('num_hashes must be divisible by num_bands')

        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.prime = 2**31 - 1
        self.buckets = defaultdict(list)

        # Single set of hash functions
        np.random.seed(seed)
        self.a = np.random.randint(1, 2**32 - 1, size=num_hashes, dtype=np.uint32)
        self.b = np.random.randint(1, 2**32 - 1, size=num_hashes, dtype=np.uint32)

        # Length-based index for fallback
        self.length_index = defaultdict(list)

    def compute_minhash(self, tokens: list[str]) -> np.ndarray:
        """Compute single MinHash signature."""
        if not tokens:
            return np.full(self.num_hashes, self.prime, dtype=np.uint32)

        token_hashes = np.array([fnv1a_hash(token) for token in tokens], dtype=np.uint32)
        token_hashes_expanded = token_hashes[:, np.newaxis]

        hashes = (token_hashes_expanded * self.a[np.newaxis, :] + self.b[np.newaxis, :]) % self.prime
        return np.min(hashes, axis=0).astype(np.uint32)

    def _get_band_hashes(self, signature: np.ndarray) -> list[int]:
        """Convert signature to band hashes."""
        return [hash(tuple(signature[i : i + self.rows_per_band])) for i in range(0, self.num_hashes, self.rows_per_band)]

    def insert_batch(self, signatures_batch: list[np.ndarray], string_lengths: list[int]) -> None:
        """Insert batch of signatures and build length index."""
        for idx, (signature, length) in enumerate(zip(signatures_batch, string_lengths)):
            # Insert into LSH buckets
            for band_idx, band_hash in enumerate(self._get_band_hashes(signature)):
                self.buckets[(band_idx, band_hash)].append(idx)

            # Insert into length index
            self.length_index[length].append(idx)

    def query_hybrid(self, signature: np.ndarray, string_length: int, max_candidates: int = 50) -> set:
        """
        Hybrid query strategy:
        1. Primary: LSH-based lookup
        2. Fallback: Length-based lookup for short strings or when LSH fails
        """
        candidates = set()

        # Phase 1: LSH lookup
        for band_idx, band_hash in enumerate(self._get_band_hashes(signature)):
            candidates.update(self.buckets.get((band_idx, band_hash), []))

        # Phase 2: Intelligent fallback
        if len(candidates) < 5 or string_length <= 8:  # LSH often fails on short strings
            # Length-based fallback with adaptive tolerance
            tolerance = max(2, int(string_length * 0.4))  # More generous tolerance
            for length in range(max(1, string_length - tolerance), string_length + tolerance + 1):
                candidates.update(self.length_index.get(length, []))
                if len(candidates) >= max_candidates:
                    break

        return candidates


def clean_string(s: str) -> str:
    """Clean a single string by removing non-alphanumeric characters."""
    import re

    return re.sub(r'[^a-zA-Z\d]', '', s).strip().lower()


def vectorized_string_matching_optimized(
    s3: pl.Series,
    s4: pl.Series,
    batch_size: int = 20000,
    score_cutoff: float = 80.0,
    length_tolerance: float = 0.25,
    num_hashes: int = 64,
    num_bands: int = 8,
    strategy: str = 'hybrid',  # 'hybrid', 'adaptive', or 'multi'
    num_seeds: int = 2,
    max_candidates: int = 50,
    n_procs: int | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """
    Optimized vectorized string matching with multiple strategies.

    Args:
        strategy: 'hybrid' (single-seed + fallbacks), 'adaptive' (conditional multi-seed),
                 or 'multi' (full multi-seed)
    """
    if n_procs is None:
        n_procs = min(4, os.cpu_count() or 1)

    # Clean strings and filter nulls
    s3_mask = s3.is_not_null()
    s4_mask = s4.is_not_null()

    s3_cleaned = s3.filter(s3_mask).map_elements(clean_string, return_dtype=pl.String)
    s4_cleaned = s4.filter(s4_mask).map_elements(clean_string, return_dtype=pl.String)

    s3_original = s3.filter(s3_mask)
    s4_original = s4.filter(s4_mask)

    # Convert to lists
    s4_list = s4_cleaned.to_list()
    s3_list = s3_cleaned.to_list()
    s3_orig_list = s3_original.to_list()
    s4_orig_list = s4_original.to_list()

    if verbose:
        print(f'Processing {len(s3_list)} queries against {len(s4_list)} candidates')
        print(f'Using strategy: {strategy}')

    # Build index based on strategy
    t_start = time.time()
    s4_tokens = preprocess_strings(s4_list, n_procs)
    s4_lengths = [len(s) for s in s4_list]

    if strategy == 'hybrid':
        # Hybrid LSH (single-seed + length-based fallback)
        lsh = HybridLSH(num_hashes=num_hashes, num_bands=num_bands)

        with Pool(processes=n_procs) as pool:
            s4_signatures = pool.starmap(lsh.compute_minhash, [(tokens,) for tokens in s4_tokens])

        lsh.insert_batch(s4_signatures, s4_lengths)

    elif strategy == 'adaptive':
        # Adaptive multi-seed LSH
        lsh = AdaptiveMultiSeedMinHashLSH(num_hashes=num_hashes, num_bands=num_bands, num_seeds=num_seeds)

        with Pool(processes=n_procs) as pool:
            s4_signatures_batch = pool.map(
                compute_minhash_multi, [(tokens, num_hashes, lsh.hash_params, lsh.prime) for tokens in s4_tokens]
            )

        lsh.insert_batch(s4_signatures_batch)

    else:  # strategy == 'multi'
        # Original multi-seed approach
        from improved_fuzzmatch import MultiSeedMinHashLSH

        lsh = MultiSeedMinHashLSH(num_hashes=num_hashes, num_bands=num_bands, num_seeds=num_seeds)

        with Pool(processes=n_procs) as pool:
            s4_signatures_batch = pool.map(
                compute_minhash_multi, [(tokens, num_hashes, lsh.hash_params, lsh.prime) for tokens in s4_tokens]
            )

        lsh.insert_batch(s4_signatures_batch)

    if verbose:
        if hasattr(lsh, 'get_stats'):
            stats = lsh.get_stats()
            print(f'Index built in {time.time() - t_start:.2f}s')
            print(f'LSH stats: {stats}')

    # Process queries in batches
    results = []
    total_batches = (len(s3_list) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(s3_list), batch_size):
        if verbose and len(s3_list) > batch_size:
            print(f'Processing batch {batch_idx // batch_size + 1}/{total_batches}')

        batch_end = min(batch_idx + batch_size, len(s3_list))
        s3_batch = s3_list[batch_idx:batch_end]
        s3_orig_batch = s3_orig_list[batch_idx:batch_end]

        # Generate tokens and signatures for current batch
        s3_tokens = preprocess_strings(s3_batch, n_procs)
        s3_lengths = [len(s) for s in s3_batch]

        if strategy == 'hybrid':
            with Pool(processes=n_procs) as pool:
                s3_signatures = pool.starmap(lsh.compute_minhash, [(tokens,) for tokens in s3_tokens])
        else:
            with Pool(processes=n_procs) as pool:
                s3_signatures_batch = pool.map(
                    compute_minhash_multi, [(tokens, num_hashes, lsh.hash_params, lsh.prime) for tokens in s3_tokens]
                )

        # Process each query
        for i, (s3_string, s3_orig) in enumerate(zip(s3_batch, s3_orig_batch)):
            if strategy == 'hybrid':
                candidates = lsh.query_hybrid(s3_signatures[i], s3_lengths[i], max_candidates)
            elif strategy == 'adaptive':
                candidates = lsh.query_adaptive(s3_signatures_batch[i], max_candidates)
            else:  # multi
                candidates = lsh.query(s3_signatures_batch[i])

            if not candidates:
                results.append((s3_orig, None, 0.00))
                continue

            # Length filtering
            query_length = len(s3_string)
            valid_candidates = []

            adaptive_tolerance = length_tolerance
            if query_length <= 6:
                adaptive_tolerance = max(length_tolerance, 0.4)
            elif query_length <= 10:
                adaptive_tolerance = max(length_tolerance, 0.3)

            for idx in candidates:
                candidate_length = len(s4_list[idx])
                if abs(candidate_length - query_length) <= query_length * adaptive_tolerance:
                    valid_candidates.append(idx)
                    if len(valid_candidates) >= max_candidates:
                        break

            if not valid_candidates:
                results.append((s3_orig, None, 0.00))
                continue

            # Fuzzy matching on valid candidates
            candidate_strings = [s4_list[idx] for idx in valid_candidates]
            matches = process.extract(s3_string, candidate_strings, scorer=fuzz.ratio, score_cutoff=score_cutoff, limit=None)

            if matches:
                best_match = max(matches, key=lambda x: x[1])
                match_string, score = best_match[0], round(best_match[1] / 100.0, 2)
                match_idx = best_match[2]
                original_idx = valid_candidates[match_idx]
                results.append((s3_orig, s4_orig_list[original_idx], score))
            else:
                results.append((s3_orig, None, 0.00))

    # Create results DataFrame
    results_cleaned = [(str(r[0]), str(r[1]) if r[1] is not None else None, float(r[2])) for r in results]

    return pl.DataFrame(
        results_cleaned,
        schema={'Original_String': pl.String, 'Matched_String': pl.String, 'Similarity_Score': pl.Float64},
        orient='row',
    )


def run_strategy_comparison():
    """Compare different strategies."""
    print('=== Strategy Comparison ===\n')

    # Test the specific problematic case
    s3 = pl.Series('s3', ['100-126', '310-505', 'test-string', 'short', 'a1b2c3'])
    s4 = pl.Series('s4', ['10012', '31050', 'teststring', 'other', 'abc123', 'a1b2c3d', 'shore', 'short'])

    strategies = [('hybrid', 'Hybrid LSH'), ('adaptive', 'Adaptive Multi-seed'), ('multi', 'Full Multi-seed')]

    for strategy, desc in strategies:
        print(f'{desc} results:')
        start_time = time.time()

        results_df = vectorized_string_matching_optimized(
            s3,
            s4,
            batch_size=1000,
            score_cutoff=75,  # Slightly lower threshold for better recall
            strategy=strategy,
            num_seeds=2,
            verbose=False,
        )

        end_time = time.time()
        matches = len(results_df.filter(pl.col('Similarity_Score') > 0))
        print(f'Time: {end_time - start_time:.3f}s | Matches: {matches}/5')
        print(results_df)
        print()


if __name__ == '__main__':
    run_strategy_comparison()

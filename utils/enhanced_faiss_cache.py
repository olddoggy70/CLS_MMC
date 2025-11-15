import hashlib
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl
from rapidfuzz import fuzz


# Disk cache management functions
def get_cache_directory(cache_base_dir=None, system_name=None):
    """
    Get or create a cache directory, optionally system-specific.

    Args:
        cache_base_dir: Base directory for cache (from config), if None uses default
        system_name: System name for system-specific cache subdirectory

    Returns:
        Path: Cache directory path
    """
    if cache_base_dir:
        cache_dir = Path(cache_base_dir)
        if system_name:
            cache_dir = cache_dir / f'faiss_cache_{system_name}'
    else:
        cache_dir = Path.home() / '.faiss_string_matcher_cache'
        if system_name:
            cache_dir = cache_dir / system_name

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def create_cache_key(s4_list, faiss_dim, faiss_k, stable_ties, tie_threshold, system_name=None):
    """Create a unique cache key based on candidates and parameters."""
    candidates_str = '|'.join(sorted(s4_list))  # Sort for consistent hashing
    params_str = f'{faiss_dim}_{faiss_k}_{stable_ties}_{tie_threshold}'
    system_str = f'_{system_name}' if system_name else ''

    full_string = f'{candidates_str}#{params_str}{system_str}'
    cache_key = hashlib.sha256(full_string.encode()).hexdigest()[:16]
    return cache_key


def save_matcher_to_disk(matcher, cache_key, cache_base_dir=None, system_name=None, logger=None):
    """Save FAISS matcher to disk with metadata."""
    cache_dir = get_cache_directory(cache_base_dir, system_name)
    cache_file = cache_dir / f'faiss_matcher_{cache_key}.pkl'
    metadata_file = cache_dir / f'faiss_matcher_{cache_key}_meta.txt'

    try:
        # Save the matcher
        with open(cache_file, 'wb') as f:
            pickle.dump(matcher, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata for debugging
        with open(metadata_file, 'w') as f:
            f.write(f'Cache created: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'System: {system_name or "default"}\n')
            f.write(
                f'Candidates count: {len(matcher.candidate_strings) if hasattr(matcher, "candidate_strings") else "unknown"}\n'
            )
            f.write(f'Dimensions: {matcher.dim}\n')
            f.write(f'K value: {matcher.k}\n')
            f.write(f'Cache key: {cache_key}\n')
            f.write(f'Cache directory: {cache_dir}\n')

        if logger:
            logger.debug(f'FAISS matcher saved to cache for {system_name or "default"} (key: {cache_key})')
        return True

    except Exception as e:
        if logger:
            logger.error(f'Failed to save FAISS matcher to cache: {e}')
        return False


def load_matcher_from_disk(cache_key, cache_base_dir=None, system_name=None, logger=None):
    """Load FAISS matcher from disk."""
    cache_dir = get_cache_directory(cache_base_dir, system_name)
    cache_file = cache_dir / f'faiss_matcher_{cache_key}.pkl'

    try:
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                matcher = pickle.load(f)

            if logger:
                logger.debug(f'FAISS matcher loaded from cache for {system_name or "default"} (key: {cache_key})')
            return matcher
        else:
            if logger:
                logger.debug(f'No cached FAISS matcher found for {system_name or "default"} (key: {cache_key})')
            return None

    except Exception as e:
        if logger:
            logger.warning(f'Failed to load cached FAISS matcher: {e}')
        return None


def cleanup_old_cache_files(max_age_days=7, cache_base_dir=None, system_name=None, logger=None):
    """Remove cache files older than specified days."""
    cache_dir = get_cache_directory(cache_base_dir, system_name)
    current_time = time.time()
    cutoff_time = current_time - (max_age_days * 24 * 60 * 60)

    removed_count = 0
    for cache_file in cache_dir.glob('faiss_matcher_*.pkl'):
        if cache_file.stat().st_mtime < cutoff_time:
            try:
                cache_file.unlink()
                # Also remove corresponding metadata file
                meta_file = cache_file.with_suffix('').with_suffix('') + '_meta.txt'
                if meta_file.exists():
                    meta_file.unlink()
                removed_count += 1
            except Exception as e:
                if logger:
                    logger.warning(f'Failed to remove old cache file {cache_file}: {e}')

    if logger and removed_count > 0:
        logger.debug(f'Cleaned up {removed_count} old FAISS cache files for {system_name or "default"}')


def get_cache_stats(cache_base_dir=None, system_name=None, logger=None):
    """Get statistics about the disk cache."""
    cache_dir = get_cache_directory(cache_base_dir, system_name)
    cache_files = list(cache_dir.glob('faiss_matcher_*.pkl'))

    if not cache_files:
        if logger:
            logger.debug(f'No FAISS cache files found for {system_name or "default"}')
        return {'count': 0, 'total_size_mb': 0, 'cache_directory': str(cache_dir)}

    total_size = sum(f.stat().st_size for f in cache_files)
    total_size_mb = total_size / 1024 / 1024

    stats = {
        'count': len(cache_files),
        'total_size_mb': round(total_size_mb, 2),
        'cache_directory': str(cache_dir),
        'system_name': system_name,
    }

    if logger:
        logger.debug(f'FAISS cache stats for {system_name or "default"}: {stats["count"]} files, {stats["total_size_mb"]} MB')

    return stats


def str_to_vec(s: str, dim: int = 32) -> np.ndarray:
    """Enhanced string-to-vector encoding with better similarity preservation."""
    clean_s = ''.join(ch for ch in s if ch.isalnum()).lower()
    arr = np.zeros(dim, dtype=np.float32)

    if not clean_s:
        return arr

    # Allocate dimension ranges for different features
    char_dim = dim // 4  # 25% for character frequencies
    bigram_dim = dim // 4  # 25% for bigrams
    trigram_dim = dim // 4  # 25% for trigrams
    position_dim = dim - char_dim - bigram_dim - trigram_dim  # Remaining for positional

    # 1. Character frequency encoding
    for ch in clean_s:
        idx = ord(ch) % char_dim
        arr[idx] += 1

    # 2. Bigram encoding with better hash distribution
    if len(clean_s) >= 2:
        for i in range(len(clean_s) - 1):
            bigram = clean_s[i : i + 2]
            hash_val = 0
            for j, c in enumerate(bigram):
                hash_val += ord(c) * (31**j)  # Prime number base
            idx = char_dim + (hash_val % bigram_dim)
            arr[idx] += 1

    # 3. Trigram encoding for better context capture
    if len(clean_s) >= 3:
        for i in range(len(clean_s) - 2):
            trigram = clean_s[i : i + 3]
            hash_val = 0
            for j, c in enumerate(trigram):
                hash_val += ord(c) * (37**j)  # Different prime
            idx = char_dim + bigram_dim + (hash_val % trigram_dim)
            arr[idx] += 1

    # 4. Positional encoding - different positions contribute differently
    position_start = char_dim + bigram_dim + trigram_dim
    string_len = len(clean_s)

    # Encode start characters (first 3)
    for i in range(min(3, string_len)):
        ch = clean_s[i]
        pos_weight = 1.0 / (i + 1)  # Earlier positions have higher weight
        idx = position_start + ((ord(ch) + i * 256) % position_dim)
        arr[idx] += pos_weight

    # Encode end characters (last 3)
    for i in range(min(3, string_len)):
        ch = clean_s[string_len - 1 - i]
        pos_weight = 1.0 / (i + 1)
        idx = position_start + ((ord(ch) + (i + 10) * 256) % position_dim)
        arr[idx] += pos_weight

    # 5. Length-based features
    if position_dim > 10:
        length_bucket = min(string_len // 2, position_dim - 1)
        arr[position_start + length_bucket] += 2.0

    # 6. Character type distribution
    if position_dim > 15:
        digit_count = sum(1 for c in clean_s if c.isdigit())
        alpha_count = len(clean_s) - digit_count

        if digit_count > 0:
            arr[position_start + position_dim - 2] += digit_count / len(clean_s)
        if alpha_count > 0:
            arr[position_start + position_dim - 1] += alpha_count / len(clean_s)

    # Normalize to unit vector for cosine similarity
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm

    return arr


def clean_string(s: str) -> str:
    """Clean a single string by removing non-alphanumeric characters."""
    import re

    return re.sub(r'[^a-zA-Z\d]', '', s).strip().lower()


class FAISSStringMatcher:
    """FAISS-based string matching with stable tie-breaking for improved performance and consistency."""

    def __init__(self, dim: int = 32, k: int = 10, stable_ties: bool = True, tie_threshold: float = 0.001):
        self.dim = dim
        self.k = k
        self.stable_ties = stable_ties
        self.tie_threshold = tie_threshold
        self.index = None
        self.candidate_strings = None
        self.candidate_strings_cleaned = None

    def build_index(self, candidates: list[str]) -> None:
        """Build FAISS index from candidate strings."""
        self.candidate_strings = candidates
        self.candidate_strings_cleaned = [clean_string(s) for s in candidates]

        # Convert strings to vectors
        candidate_vecs = np.vstack([str_to_vec(s, self.dim) for s in self.candidate_strings_cleaned])

        # Create and populate FAISS index
        self.index = faiss.IndexFlatIP(self.dim)  # Inner Product (cosine similarity)
        faiss.normalize_L2(candidate_vecs)  # Normalize for cosine similarity
        self.index.add(candidate_vecs)

    def search_candidates(self, queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Search for candidates using FAISS with optional stable tie-breaking."""
        if self.index is None:
            raise ValueError('Index not built. Call build_index() first.')

        # Clean queries and convert to vectors
        queries_cleaned = [clean_string(q) for q in queries]
        query_vecs = np.vstack([str_to_vec(q, self.dim) for q in queries_cleaned])
        faiss.normalize_L2(query_vecs)

        if not self.stable_ties:
            D, I = self.index.search(query_vecs, self.k)  # noqa: E741
            return D, I

        # For stable tie-breaking, search more candidates
        if len(self.candidate_strings) <= 1000:
            search_k = min(len(self.candidate_strings), max(self.k * 4, 10))
        elif len(self.candidate_strings) <= 10000:
            search_k = min(len(self.candidate_strings), max(self.k * 3, 20))
        else:
            search_k = min(len(self.candidate_strings), max(self.k * 2, 75))

        D, I = self.index.search(query_vecs, search_k)  # noqa: E741

        # Apply stable tie-breaking with fuzzy score priority
        stable_D = []
        stable_I = []

        for i, query in enumerate(queries):
            query_clean = queries_cleaned[i]
            distances = D[i]
            indices = I[i]

            # Create candidates with fuzzy scores for hybrid ranking
            candidates_with_scores = []
            for j, (dist, idx) in enumerate(zip(distances, indices)):
                if idx < len(self.candidate_strings):
                    candidate_clean = self.candidate_strings_cleaned[idx]
                    fuzzy_score = fuzz.ratio(query_clean, candidate_clean)

                    # Calculate hybrid score: combine FAISS similarity and fuzzy score
                    faiss_similarity = 1.0 - dist
                    hybrid_score = 0.6 * (fuzzy_score / 100.0) + 0.4 * faiss_similarity

                    candidates_with_scores.append((dist, idx, self.candidate_strings[idx], fuzzy_score, hybrid_score))

            # Sort by hybrid score (descending), then by candidate name for determinism
            candidates_with_scores.sort(key=lambda x: (-x[4], x[2]))
            top_k = candidates_with_scores[: self.k]

            stable_D.append([item[0] for item in top_k])
            stable_I.append([item[1] for item in top_k])

        return np.array(stable_D), np.array(stable_I)


def vectorized_string_matching_optimized(
    s3: pl.Series,
    s4: pl.Series,
    score_cutoff: float = 90.0,
    length_tolerance: float = 0.25,
    faiss_k: int = 10,
    faiss_dim: int = 64,
    stable_ties: bool = True,
    tie_threshold: float = 0.001,
    adaptive_k: bool = True,
    use_disk_cache: bool = True,
    cache_cleanup_days: int = 7,
    cache_base_dir: str | None = None,
    system_name: str | None = None,
    logger=None,
    debug_query: str | None = None,
) -> pl.DataFrame:
    """
    FAISS-based vectorized string matching with persistent disk caching and system-specific cache directories.

    Args:
        s3: Query strings series
        s4: Candidate strings series
        score_cutoff: Minimum similarity score (0-100)
        length_tolerance: Length tolerance for filtering candidates
        faiss_k: Number of final candidates to return
        faiss_dim: Vector dimension for FAISS encoding
        stable_ties: Whether to use hybrid FAISS+fuzzy scoring
        tie_threshold: Distance threshold for considering candidates as tied
        adaptive_k: Whether to adaptively increase search size for better fuzzy matches
        use_disk_cache: Whether to use disk caching for FAISS index
        cache_cleanup_days: Remove cache files older than this many days
        cache_base_dir: Base directory for cache (from config)
        system_name: System name for system-specific cache subdirectory
        logger: Logger instance for proper logging
        debug_query: If specified, provide detailed debugging for this query string

    Returns:
        DataFrame with columns: Original_String, Matched_String, Similarity_Score
    """

    # Clean up old cache files periodically
    if use_disk_cache and cache_cleanup_days > 0:
        cleanup_old_cache_files(cache_cleanup_days, cache_base_dir, system_name, logger)

    # Filter nulls
    s3_filtered = s3.filter(s3.is_not_null())
    s4_filtered = s4.filter(s4.is_not_null())

    # Convert to lists
    s3_list = s3_filtered.to_list()
    s4_list = s4_filtered.to_list()

    if not s3_list or not s4_list:
        return pl.DataFrame(schema={'Original_String': pl.String, 'Matched_String': pl.String, 'Similarity_Score': pl.String})

    matcher = None

    # Try to load from disk cache
    if use_disk_cache:
        cache_key = create_cache_key(s4_list, faiss_dim, faiss_k, stable_ties, tie_threshold, system_name)
        matcher = load_matcher_from_disk(cache_key, cache_base_dir, system_name, logger)

        # Validate that the loaded matcher matches our requirements
        if matcher is not None:
            if (
                (hasattr(matcher, 'dim') and matcher.dim != faiss_dim)
                or (hasattr(matcher, 'k')
                and matcher.k != faiss_k)
                or (hasattr(matcher, 'candidate_strings')
                and len(matcher.candidate_strings) != len(s4_list))
            ):
                if logger:
                    logger.debug(
                        f"Cached matcher parameters don't match requirements for {system_name or 'default'}, rebuilding..."
                    )
                matcher = None

    # Build new matcher if cache miss or invalid cache
    if matcher is None:
        if logger:
            logger.debug(
                f'Building new FAISS index for {system_name or "default"}: {len(s4_list)} candidates (dim={faiss_dim}, k={faiss_k})'
            )

        t_start = time.time()
        matcher = FAISSStringMatcher(dim=faiss_dim, k=faiss_k, stable_ties=stable_ties, tie_threshold=tie_threshold)
        matcher.build_index(s4_list)

        if logger:
            logger.debug(f'FAISS index built for {system_name or "default"} in {time.time() - t_start:.2f}s')

        # Save to disk cache
        if use_disk_cache:
            save_matcher_to_disk(matcher, cache_key, cache_base_dir, system_name, logger)
    else:
        if logger:
            logger.debug(f'Using cached FAISS index for {system_name or "default"}: {len(s4_list)} candidates')

    # Process queries
    t_search_start = time.time()
    D, I = matcher.search_candidates(s3_list)  # noqa: E741, RUF059

    if logger:
        logger.debug(f'FAISS search completed for {system_name or "default"} in {time.time() - t_search_start:.2f}s')

    # Process results
    results = []
    for i, query_orig in enumerate(s3_list):
        query_clean = clean_string(query_orig)
        candidate_indices = I[i]

        # Get candidate strings
        candidate_strings_orig = [matcher.candidate_strings[idx] for idx in candidate_indices if idx < len(s4_list)]
        candidate_strings_clean = [matcher.candidate_strings_cleaned[idx] for idx in candidate_indices if idx < len(s4_list)]

        if not candidate_strings_clean:
            results.append((query_orig, None, 0.00))
            continue

        # Length-based filtering with adaptive tolerance
        query_length = len(query_clean)
        adaptive_tolerance = length_tolerance
        if query_length <= 6:
            adaptive_tolerance = max(length_tolerance, 0.4)
        elif query_length <= 10:
            adaptive_tolerance = max(length_tolerance, 0.3)

        valid_candidates_orig = []
        valid_candidates_clean = []

        for orig, clean in zip(candidate_strings_orig, candidate_strings_clean):
            candidate_length = len(clean)
            if abs(candidate_length - query_length) <= query_length * adaptive_tolerance:
                valid_candidates_orig.append(orig)
                valid_candidates_clean.append(clean)

        if not valid_candidates_clean:
            results.append((query_orig, None, 0.00))
            continue

        # Calculate fuzzy scores
        scores = [fuzz.ratio(query_clean, cand_clean) for cand_clean in valid_candidates_clean]
        scores = np.array(scores, dtype=np.float32)

        # Find best score
        best_idx = scores.argmax()
        best_score = scores[best_idx] / 100.0

        # Apply score cutoff
        if best_score >= (score_cutoff / 100.0):
            matched_string_orig = valid_candidates_orig[best_idx]
            results.append((query_orig, matched_string_orig, round(best_score, 2)))
        else:
            results.append((query_orig, None, 0.00))

    if logger:
        matches = sum(1 for r in results if r[1] is not None)
        logger.debug(f'FAISS matching completed for {system_name or "default"}: {matches}/{len(results)} matches')

    # Create results DataFrame
    results_cleaned = [(str(r[0]), str(r[1]) if r[1] is not None else None, round(float(r[2]), 2)) for r in results]

    return pl.DataFrame(
        results_cleaned,
        schema={'Original_String': pl.String, 'Matched_String': pl.String, 'Similarity_Score': pl.Float64},
        orient='row',
    )

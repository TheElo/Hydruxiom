"""Vectorizer for 3D Tag Space Visualization.

Builds tag vocabulary and creates TF-IDF sparse vectors for files.
"""

import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from src.utils.query_comperator import ExternalTagScores


class Vectorizer:
    """Converts file tags to TF-IDF sparse vectors with score weights."""

    def __init__(self, min_doc_freq=3, tokenized=False, reverse_vocab=None, drop_universal_tags=False):
        """Initialize the vectorizer.

        Args:
            min_doc_freq: Minimum number of documents a tag must appear in
                to be included in vocabulary (default: 3). Tags appearing in
                fewer documents are filtered out to reduce dimensionality.
            tokenized: Whether tag_data values are integer indices instead of
                strings (default: False). When True, reverse_vocab must be
                provided to resolve indices back to strings for scoring.
            reverse_vocab: List mapping index -> tag string, used only when
                tokenized=True (e.g. from a TagInterner).
            drop_universal_tags: If True, tags appearing in EVERY document
                are excluded from the vocabulary (default: False). These tags
                provide zero discriminative power and are typically visible
                in the user's query field already.
        """
        self.vocabulary = {}  # tag -> index
        self.idf = {}  # tag -> idf score
        self.tag_scores = ExternalTagScores
        self.min_doc_freq = min_doc_freq
        self.tokenized = tokenized
        self.reverse_vocab = reverse_vocab
        self.drop_universal_tags = drop_universal_tags

    def build_vocabulary(self, tag_data):
        """Build vocabulary from all tags in the data.

        Args:
            tag_data: Dictionary mapping file_id to list of tags
                (strings, or integer indices when tokenized=True)
        """
        # Count document frequency for each tag
        doc_freq = defaultdict(int)
        all_tags = set()

        for tags in tag_data.values():
            unique_tags = set(tags)
            all_tags.update(unique_tags)
            for tag in unique_tags:
                doc_freq[tag] += 1

        # Build vocabulary (sorted for consistency), filtering rare tags.
        # In tokenized mode, tags are integer indices from the interner; the
        # vocabulary maps original index -> dense column index so columns stay
        # within the matrix shape (original indices may exceed the filtered size).
        n_documents = len(tag_data)
        
        def _keep(tag):
            if doc_freq[tag] < self.min_doc_freq:
                return False
            if self.drop_universal_tags and doc_freq[tag] >= n_documents:
                return False
            return True
        
        filtered_tags = sorted(tag for tag in all_tags if _keep(tag))
        self.vocabulary = {tag: idx for idx, tag in enumerate(filtered_tags)}
        
        # Calculate IDF scores
        for tag in filtered_tags:
            freq = doc_freq[tag]
            # IDF = log(n_docs / doc_freq) + 1 (smoothed)
            self.idf[tag] = np.log(n_documents / (freq + 1)) + 1

        rare_count = sum(1 for t in all_tags if doc_freq[t] < self.min_doc_freq)
        universal_count = sum(1 for t in all_tags if doc_freq[t] >= n_documents) if self.drop_universal_tags else 0
        filtered_count = len(all_tags) - len(filtered_tags)
        msg = f"Built vocabulary with {len(self.vocabulary)} unique tags "
        msg += f"(filtered out {filtered_count} tags"
        if rare_count:
            msg += f": {rare_count} rare (<{self.min_doc_freq} docs)"
        if universal_count:
            msg += f", {universal_count} universal (100% docs)"
        msg += ")"
        print(msg)

    def create_vectors(self, tag_data):
        """Create TF-IDF sparse vectors for all files.

        Args:
            tag_data: Dictionary mapping file_id to list of tags
                (strings, or integer indices when tokenized=True)

        Returns:
            tuple: (sparse_matrix, file_id_order)
                - sparse_matrix: csr_matrix of shape (n_files, n_tags)
                - file_id_order: List of file IDs in row order
        """
        if not self.vocabulary:
            self.build_vocabulary(tag_data)

        file_ids = list(tag_data.keys())
        n_files = len(file_ids)
        n_tags = len(self.vocabulary)

        # Build sparse matrix
        rows = []
        cols = []
        data = []

        for file_idx, file_id in enumerate(file_ids):
            tags = tag_data.get(file_id, [])
            
            # Count term frequency for this file
            tf_counts = defaultdict(int)
            for tag in tags:
                tf_counts[tag] += 1

            # Create TF-IDF values
            for tag, tf in tf_counts.items():
                if self.tokenized:
                    # Tags are already integer indices from the interner.
                    # Map original index -> dense column index via vocabulary.
                    if tag not in self.vocabulary:
                        continue
                    tag_idx = self.vocabulary[tag]
                    # TF-IDF value
                    tfidf = tf * self.idf.get(tag, 1.0)
                    # Resolve string for ExternalTagScores lookup
                    tag_str = self.reverse_vocab[tag] if self.reverse_vocab and tag < len(self.reverse_vocab) else None
                    score = self.tag_scores.get(tag_str, 0) if tag_str is not None else 0
                    score_multiplier = 1.0 + 0.1 * score
                    final_value = tfidf * score_multiplier
                    rows.append(file_idx)
                    cols.append(tag_idx)
                    data.append(final_value)
                else:
                    if tag in self.vocabulary:
                        tag_idx = self.vocabulary[tag]
                        # TF-IDF value
                        tfidf = tf * self.idf.get(tag, 1.0)
                        # Apply score weight from ExternalTagScores
                        score = self.tag_scores.get(tag, 0)
                        score_multiplier = 1.0 + 0.1 * score
                        final_value = tfidf * score_multiplier
                        rows.append(file_idx)
                        cols.append(tag_idx)
                        data.append(final_value)

        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_files, n_tags)
        )

        print(f"Created sparse matrix: {sparse_matrix.shape}")
        print(f"Non-zero elements: {sparse_matrix.nnz}")
        return sparse_matrix, file_ids

    def get_tag_names(self):
        """Get list of tag names in vocabulary order.

        Returns:
            list: List of tag names indexed by vocabulary index
        """
        if not self.vocabulary:
            return []

        if self.tokenized and self.reverse_vocab:
            # vocabulary maps original index -> dense column index.
            # Return names for the dense columns in order.
            inverse_vocab = {idx: tag for tag, idx in self.vocabulary.items()}
            return [self.reverse_vocab[inverse_vocab[i]] for i in range(len(self.vocabulary))]

        inverse_vocab = {idx: tag for tag, idx in self.vocabulary.items()}
        return [inverse_vocab[i] for i in range(len(self.vocabulary))]

    def get_top_tags(self, top_n=20):
        """Get top N tags by IDF score (most distinctive tags).

        Args:
            top_n: Number of top tags to return

        Returns:
            list: List of (tag, idf_score) tuples
        """
        sorted_tags = sorted(self.idf.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:top_n]

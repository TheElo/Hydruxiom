"""Tag interner for tokenized (integer-index) tag handling.

When the "tokenize" toggle is enabled, tags are converted to integer
indices once at load time and carried through the pipeline as integers.
This avoids storing full tag strings on every TagNode (a large memory
duplication at scale) and replaces repeated string hashing with O(1)
integer lookups.

Strings are only materialised at display boundaries (dominant tags,
node info, JSON export) via the reverse vocabulary.
"""


class TagInterner:
    """Maps tag strings to integer indices and back."""

    def __init__(self):
        self.tag_to_index = {}
        self.index_to_tag = []

    def build(self, tag_data):
        """Build the vocabulary from all tags and tokenize the data.

        Args:
            tag_data: dict mapping file_id -> list of tag strings

        Returns:
            dict: tokenized data mapping file_id -> list of integer indices
        """
        self.tag_to_index = {}
        self.index_to_tag = []
        tokenized = {}
        for fid, tags in tag_data.items():
            indices = []
            for tag in tags:
                if tag not in self.tag_to_index:
                    self.tag_to_index[tag] = len(self.index_to_tag)
                    self.index_to_tag.append(tag)
                indices.append(self.tag_to_index[tag])
            tokenized[fid] = indices
        return tokenized

    def to_strings(self, tokenized_data):
        """Convert tokenized data back to tag strings.

        Args:
            tokenized_data: dict mapping file_id -> list of integer indices

        Returns:
            dict: mapping file_id -> list of tag strings
        """
        strings = {}
        for fid, indices in tokenized_data.items():
            strings[fid] = [self.index_to_tag[i] for i in indices]
        return strings

    def index_to_string(self, idx):
        """Convert a single index back to its tag string (or None)."""
        if 0 <= idx < len(self.index_to_tag):
            return self.index_to_tag[idx]
        return None

    def tokenize_list(self, tags):
        """Tokenize a list of tag strings into indices (reusing vocabulary)."""
        indices = []
        for tag in tags:
            if tag not in self.tag_to_index:
                self.tag_to_index[tag] = len(self.index_to_tag)
                self.index_to_tag.append(tag)
            indices.append(self.tag_to_index[tag])
        return indices

    def strings_to_list(self, indices):
        """Convert a list of indices back to tag strings.

        Indices that are out of bounds (e.g. from a stale interner after a
        failed reload) are silently skipped rather than raising IndexError.
        """
        n = len(self.index_to_tag)
        return [self.index_to_tag[i] for i in indices if 0 <= i < n]

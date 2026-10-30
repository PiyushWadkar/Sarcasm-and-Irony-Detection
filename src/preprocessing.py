"""
Text Preprocessing Pipeline for Sarcasm and Irony Detection.

This module provides text normalization, contraction expansion, hashtag segmentation,
negation-aware stopword filtering, and feature tokenization tailored for
figurative language and irony detection.
"""

import re
import html
import unicodedata
from typing import List, Tuple, Set

# Contraction expansion dictionary
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

# Negations, intensifiers, and contrastive markers critical for irony/sarcasm detection.
# These MUST NOT be stripped out as naive stopwords.
PRESERVED_TOKENS = {
    "not", "no", "nor", "neither", "never", "cannot", "without",
    "but", "however", "although", "though", "yet", "still",
    "so", "too", "very", "much", "really", "totally", "absolutely",
    "always", "again", "great", "love", "thanks", "wow", "yay"
}

# Standard English stopwords (base set from NLTK, adjusted to preserve polarity markers)
BASE_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "under", "until", "up", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves"
}

# Final stopword set excluding preserved irony markers
IRONY_AWARE_STOPWORDS: Set[str] = BASE_STOPWORDS - PRESERVED_TOKENS


def split_hashtag(hashtag: str) -> str:
    """
    Split camelCase or concatenated hashtags into readable tokens.
    Example: '#BestDayEver' -> 'Best Day Ever'
    """
    tag = hashtag.lstrip("#")
    # Split camelCase: 'BestDayEver' -> 'Best Day Ever'
    split_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
    # Split sequences of uppercase before title case: 'USAFirst' -> 'USA First'
    split_words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', split_words)
    return split_words


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    for contraction, expansion in CONTRACTIONS.items():
        # Match word boundaries or standalone tokens
        pattern = re.compile(rf"\b{re.escape(contraction)}\b", re.IGNORECASE)
        text = pattern.sub(expansion, text)
    return text


def clean_text(text: str, preserve_case: bool = False) -> str:
    """
    Core text cleaner designed for irony and sarcasm analysis.
    
    Steps:
    1. Unescape HTML entities (&amp; -> &)
    2. Normalize Unicode characters
    3. Remove URLs
    4. Normalize user mentions (@username -> @user)
    5. Expand contractions (e.g. don't -> do not)
    6. Segment hashtags (#TrafficJam -> Traffic Jam)
    7. Normalize repeated punctuation (!, ?, .) while preserving contrast markers
    8. Condense excess whitespace
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Standardize Twitter user mentions
    text = re.sub(r"@\w+", "@user", text)

    # Expand contractions
    text = expand_contractions(text)

    # Process hashtags (keep content words, remove literal '#')
    text = re.sub(r"#\w+", lambda m: split_hashtag(m.group(0)), text)

    # Normalize repeated punctuation (!, ?, .)
    text = re.sub(r"!{2,}", " !! ", text)
    text = re.sub(r"\?{2,}", " ?? ", text)
    text = re.sub(r"\.{3,}", " ... ", text)
    text = re.sub(r"!\?", " !? ", text)
    text = re.sub(r"\?!", " ?! ", text)

    # Remove noise characters while preserving alphanumeric, spaces, and punctuation
    text = re.sub(r"[^\w\s!?.'@&-]", " ", text)

    if not preserve_case:
        text = text.lower()

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_for_tfidf(text: str) -> str:
    """
    Preprocess text for TF-IDF vectorization:
    - Cleans text
    - Removes non-ironic stopwords while keeping negations and intensifiers
    - Returns normalized space-separated string of tokens
    """
    cleaned = clean_text(text, preserve_case=False)
    tokens = cleaned.split()
    
    # Filter stopwords using irony-aware stopword list
    filtered_tokens = [
        tok for tok in tokens 
        if tok not in IRONY_AWARE_STOPWORDS and (len(tok) > 1 or tok in {"!", "?"})
    ]
    
    return " ".join(filtered_tokens)


def validate_input_text(text: str, max_chars: int = 2000) -> Tuple[bool, str]:
    """
    Validate user input before sending to NLP model.
    Returns (is_valid, error_message).
    """
    if text is None:
        return False, "Input text cannot be null."
    
    trimmed = text.strip()
    if len(trimmed) == 0:
        return False, "Input text is empty. Please enter a sentence or statement to analyze."
    
    if len(trimmed) < 2:
        return False, "Input text is too short. Please provide at least one meaningful word."
    
    if len(trimmed) > max_chars:
        return False, f"Input text exceeds maximum allowed length of {max_chars} characters (current: {len(trimmed)})."
    
    return True, ""

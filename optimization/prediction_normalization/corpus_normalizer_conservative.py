"""
Conservative Kannada Text Normalizer
Only handles:
1. Unicode normalization (combining vs precomposed)
2. Basic whitespace cleanup
3. NO compound splitting
4. NO morphological changes
"""

import unicodedata
import re


class ConservativeKannadaNormalizer:
    def __init__(self):
        """Initialize conservative normalizer with minimal rules."""
        pass
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode composition.
        NFC: Canonical decomposition followed by canonical composition
        """
        return unicodedata.normalize('NFC', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Clean up whitespace issues:
        - Remove leading/trailing spaces
        - Collapse multiple spaces
        - But preserve intentional word boundaries
        """
        # Remove leading/trailing spaces
        text = text.strip()
        
        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def normalize(self, text: str) -> str:
        """
        Apply conservative normalization.
        Only Unicode normalization and whitespace cleanup.
        """
        if not text:
            return text
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Whitespace cleanup
        text = self.normalize_whitespace(text)
        
        return text


def normalize_text_conservative(text: str) -> str:
    """Convenience function for conservative normalization."""
    normalizer = ConservativeKannadaNormalizer()
    return normalizer.normalize(text)


if __name__ == "__main__":
    # Test cases
    normalizer = ConservativeKannadaNormalizer()
    
    test_cases = [
        " ಕೆಲವೇ ನಿಮಿಷಗಳಲ್ಲಿ ",  # Leading/trailing spaces
        "ಭದ್ರ  ನದಿ",  # Multiple spaces
        "ಹಠಕ್ಕೆ",  # Should NOT split to "ಹಠ ಕ್ಕೆ"
        "ಮಾಲ್ನೊಳಗೆ",  # Should NOT split
        "ವಿರುದ್ಧ",  # Keep as-is (not "ವಿರುದ್ದ")
    ]
    
    print("Conservative Normalization Tests:")
    print("=" * 60)
    for original in test_cases:
        normalized = normalizer.normalize(original)
        changed = "✓" if original != normalized else "✗"
        print(f"{changed} '{original}' → '{normalized}'")

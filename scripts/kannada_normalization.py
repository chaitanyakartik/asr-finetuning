"""
Kannada Orthographic Normalization for ASR Evaluation
Handles compound word splitting, spacing, and common orthographic variations
"""

import re
import unicodedata


class KannadaNormalizer:
    """Normalize Kannada text for consistent ASR evaluation"""
    
    def __init__(self):
        # Common compound word patterns in Kannada
        self.compound_patterns = [
            # Time/quantity + noun patterns
            (r'ಕೆಲವರ್ಷ', 'ಕೆಲವು ವರ್ಷ'),
            (r'ಹಲವುವೇಳೆ', 'ಹಲವು ವೇಳೆ'),
            (r'ಹಲವುವರ್ಷ', 'ಹಲವು ವರ್ಷ'),
            (r'ಹಲವುದಿನ', 'ಹಲವು ದಿನ'),
            (r'ಕೆಲವುವೇಳೆ', 'ಕೆಲವು ವೇಳೆ'),
            
            # Adjective + noun patterns
            (r'ಉನ್ನತಶ್ರೇಣಿ', 'ಉನ್ನತ ಶ್ರೇಣಿ'),
            (r'ಪ್ರಮುಖವ್ಯಕ್ತಿ', 'ಪ್ರಮುಖ ವ್ಯಕ್ತಿ'),
            (r'ಮುಖ್ಯಮಂತ್ರಿ', 'ಮುಖ್ಯ ಮಂತ್ರಿ'),
            (r'ಪ್ರಧಾನಮಂತ್ರಿ', 'ಪ್ರಧಾನ ಮಂತ್ರಿ'),
            
            # Common conjunctions and particles
            (r'ಮತ್ತೂ', 'ಮತ್ತು'),
            (r'ಹಾಗೂ', 'ಹಾಗು'),
            (r'ಅಂತೂ', 'ಅಂತು'),
            
            # Sandhi splitting patterns - common compound nouns
            (r'ಕರ್ನಾಟಕದ(\w)', r'ಕರ್ನಾಟಕದ \1'),
            (r'ಬೆಂಗಳೂರಿನ(\w)', r'ಬೆಂಗಳೂರಿನ \1'),
            (r'ಭಾರತದ(\w)', r'ಭಾರತದ \1'),
            
            # Date/time compounds
            (r'ಇಂದುದಿನ', 'ಇಂದು ದಿನ'),
            (r'ನಿನ್ನೆರಾತ್ರಿ', 'ನಿನ್ನೆ ರಾತ್ರಿ'),
            (r'ಇಂದುರಾತ್ರಿ', 'ಇಂದು ರಾತ್ರಿ'),
            
            # Common verb compounds
            (r'ಮಾಡಿದರೆಂದು', 'ಮಾಡಿದರೆ ಎಂದು'),
            (r'ಹೇಳಿದರೆಂದು', 'ಹೇಳಿದರೆ ಎಂದು'),
            (r'ಬಂದರೆಂದು', 'ಬಂದರೆ ಎಂದು'),
        ]
        
        # Reverse patterns for opposite normalization
        self.reverse_patterns = [
            # Join commonly written together
            (r'ಮನೆ\s+ಮನೆಗೆ', 'ಮನೆಮನೆಗೆ'),
            (r'ಒಂದಲ್ಲ\s+ಒಂದ್', 'ಒಂದಲ್ಲ ಒಂದ್'),  # keep spaced
            (r'ಎಲ್ಲ\s+ಎಲ್ಲಾ', 'ಎಲ್ಲಾ'),
        ]
    
    def normalize_unicode(self, text):
        """Normalize Unicode to NFC form"""
        return unicodedata.normalize('NFC', text)
    
    def normalize_spacing(self, text):
        """Normalize spacing around punctuation and words"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([,\.!?;:])\s*', r'\1 ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        return text
    
    def normalize_compounds(self, text):
        """Apply compound word normalization patterns"""
        for pattern, replacement in self.compound_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def normalize_halant_virama(self, text):
        """Normalize halant/virama usage"""
        # Kannada virama is ್ (U+0CCD)
        # Ensure consistent usage
        text = re.sub(r'್\s+', '್', text)  # Remove space after virama
        return text
    
    def normalize_anusvara(self, text):
        """Normalize anusvara variations"""
        # ಂ (U+0C82) vs ಃ (U+0C83)
        # Keep consistent in text
        return text
    
    def normalize_conjuncts(self, text):
        """Normalize consonant conjuncts"""
        # Some conjuncts can be written with explicit virama or ligature
        # Normalize to most common form
        
        # Common variations
        patterns = [
            (r'ತ್ತ್', 'ತ್ತ'),  # Double conjunct normalization
            (r'ನ್ನ್', 'ನ್ನ'),
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def normalize_vowel_signs(self, text):
        """Normalize vowel sign variations"""
        # Handle cases where independent vowels vs vowel signs differ
        # This is language-specific and needs careful handling
        return text
    
    def normalize_numerals(self, text):
        """Normalize number representations"""
        # Convert Kannada numerals to ASCII or vice versa if needed
        # ೦೧೨೩೪೫೬೭೮೯ <-> 0123456789
        
        # For consistency, keep ASCII numerals
        kannada_to_ascii = str.maketrans('೦೧೨೩೪೫೬೭೮೯', '0123456789')
        text = text.translate(kannada_to_ascii)
        
        return text
    
    def normalize_punctuation(self, text):
        """Normalize punctuation marks"""
        # Normalize different dash types
        text = text.replace('–', '-')
        text = text.replace('—', '-')
        text = text.replace('−', '-')
        
        # Normalize quotes
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        text = text.replace("'", "'")
        text = text.replace("'", "'")
        
        return text
    
    def normalize_special_chars(self, text):
        """Handle special characters and symbols"""
        # Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        text = text.replace('\ufeff', '')  # Zero-width no-break space
        
        return text
    
    def normalize(self, text, aggressive=True):
        """
        Apply all normalization steps
        
        Args:
            text: Input Kannada text
            aggressive: If True, apply compound word splitting aggressively
        
        Returns:
            Normalized text
        """
        if not text:
            return text
        
        # 1. Unicode normalization
        text = self.normalize_unicode(text)
        
        # 2. Remove special characters
        text = self.normalize_special_chars(text)
        
        # 3. Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # 4. Normalize numerals
        text = self.normalize_numerals(text)
        
        # 5. Normalize halant/virama
        text = self.normalize_halant_virama(text)
        
        # 6. Normalize conjuncts
        text = self.normalize_conjuncts(text)
        
        # 7. Normalize compound words (if aggressive)
        if aggressive:
            text = self.normalize_compounds(text)
        
        # 8. Final spacing normalization
        text = self.normalize_spacing(text)
        
        return text


def normalize_text(text, normalizer=None, aggressive=True):
    """
    Convenience function for normalizing text
    
    Args:
        text: Input text
        normalizer: KannadaNormalizer instance (creates new if None)
        aggressive: Apply aggressive compound splitting
    
    Returns:
        Normalized text
    """
    if normalizer is None:
        normalizer = KannadaNormalizer()
    
    return normalizer.normalize(text, aggressive=aggressive)


# Example usage
if __name__ == "__main__":
    normalizer = KannadaNormalizer()
    
    # Test cases
    test_cases = [
        "ಕೆಲವರ್ಷ ಹಲವುವೇಳೆ ಉನ್ನತಶ್ರೇಣಿ",
        "ಮನೆಮನೆಗೆ ತಲುಪಿಸುವ",
        "೧೨೩ ಎಂಟು ವರ್ಷ",
        "ಮುಖ್ಯಮಂತ್ರಿ   ಪ್ರಧಾನಮಂತ್ರಿ",
    ]
    
    print("Kannada Normalization Examples:")
    print("=" * 60)
    for text in test_cases:
        normalized = normalizer.normalize(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()

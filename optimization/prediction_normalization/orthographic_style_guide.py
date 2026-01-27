"""
Kannada Orthographic Style Guide Normalizer

Style rules:
1. Compound words: PREFER FUSED (ಉಗಮಸ್ಥಾನ not ಉಗಮ ಸ್ಥಾನ)
2. Emphatic particles: PREFER UNMARKED (ಮೊಮ್ಮಕ್ಕಳು not ಮೊಮ್ಮಕ್ಕಳೂ)
3. Foreign words: PREFER SEPARATED (ಎಸಿ ವೆಂಟ್ not ಎಸಿವೆಂಟ್)
"""

import unicodedata
import re
from typing import List, Tuple


class OrthographicStyleNormalizer:
    """Normalize Kannada text according to orthographic style guide."""
    
    def __init__(self):
        """Initialize style rules."""
        # Rule 1: Common compound words that should be FUSED
        self.compound_fusion_rules = [
            # Postpositions with case markers
            (r'(\S+) ಕ್ಕೆ', r'\1ಕ್ಕೆ'),          # e.g., ಹಠ ಕ್ಕೆ → ಹಠಕ್ಕೆ
            (r'(\S+) ಗೆ', r'\1ಗೆ'),              # e.g., ಶಾಲೆ ಗೆ → ಶಾಲೆಗೆ
            (r'(\S+) ಯಂದು', r'\1ಯಂದು'),        # e.g., ದಿನಾಚರಣೆ ಯಂದು → ದಿನಾಚರಣೆಯಂದು
            (r'(\S+) ವನ್ನು', r'\1ವನ್ನು'),      # e.g., ವಿಷಯ ವನ್ನು → ವಿಷಯವನ್ನು
            (r'(\S+) ಯವರ', r'\1ಯವರ'),          # e.g., ಭಂಡಾರಿ ಯವರ → ಭಂಡಾರಿಯವರ
            (r'(\S+) ಯಾಗಿ', r'\1ಯಾಗಿ'),        # e.g., ಶಾಲೆ ಯಾಗಿ → ಶಾಲೆಯಾಗಿ
            (r'(\S+) ಯಲ್ಲಿ', r'\1ಯಲ್ಲಿ'),      # e.g., ನೆರೆಹೊರೆ ಯಲ್ಲಿ → ನೆರೆಹೊರೆಯಲ್ಲಿ
            (r'(\S+) ದಲ್ಲಿ', r'\1ದಲ್ಲಿ'),      # e.g., ವಡೋದರ ದಲ್ಲಿ → ವಡೋದರದಲ್ಲಿ
            (r'(\S+) ದಲ್ಲಿರುವ', r'\1ದಲ್ಲಿರುವ'), # Compound case
            (r'(\S+) ವನ್ನೂ', r'\1ವನ್ನೂ'),      # e.g., ವಿಷಯ ವನ್ನೂ → ವಿಷಯವನ್ನೂ
            (r'(\S+) ದಿಂದ', r'\1ದಿಂದ'),        # e.g., ವಾರ ದಿಂದ → ವಾರದಿಂದ
            (r'(\S+) ವರೆಗೂ', r'\1ವರೆಗೂ'),      # e.g., ಸಭೆ ಯವರೆಗೂ → ಸಭೆಯವರೆಗೂ
            
            # Common compound nouns (Sanskrit/Kannada compounds)
            (r'ಉಗಮ ಸ್ಥಾನ', r'ಉಗಮಸ್ಥಾನ'),        # Origin-place
            (r'ಗ್ರೇನೇಡ್ ಸ್ಪೋಟ', r'ಗ್ರೇನೇಡ್ಸ್ಪೋಟ'),  # Keep loan word compounds together
            (r'ವಿರುದ್ದ ', r'ವಿರುದ್ಧ '),         # Common spelling variation
            (r'ಊಟ ಕ್ಕೂ', r'ಊಟಕ್ಕೂ'),            # Food + emphatic
            
            # Name compounds
            (r'ರಾಮ ಪ್ರಸಾದ', r'ರಾಮಪ್ರಸಾದ'),      # Names
            (r'ಕೆಳ ಜಾತಿ', r'ಕೆಳಜಾತಿ'),          # Lower caste (compound noun)
            
            # Verb compounds
            (r'ಅಂತ್ಯ ಗೊಂಡು', r'ಅಂತ್ಯಗೊಂಡು'),    # End-become
            (r'ಸಭೆ ಯಾಗಿ', r'ಸಭೆಯಾಗಿ'),          # Assembly-become
            (r'ಬರ ಬರ', r'ಬರಬರ'),                # Reduplicated adverb
            (r'ಹುಟ್ಟು ಕೊಂಡರೆ', r'ಹುಟ್ಟುಕೊಂಡರೆ'), # Born
            (r'ಕಂಡು ಬರುತ್ತದೆ', r'ಕಂಡುಬರುತ್ತದೆ'), # See-come
            (r'ಕಂಡು ಬರುತ್ತವೆ', r'ಕಂಡುಬರುತ್ತವೆ'),
            (r'ಬರೆದು ಕೊಂಡಿದ್ದರು', r'ಬರೆದುಕೊಂಡಿದ್ದರು'), # Write-take
            (r'ಹಂಚಿಕೊಂಡಿದ್ದಾರೆ', r'ಹಂಚಿಕೊಂಡಿದ್ದಾರೆ'), # Share (already fused)
        ]
        
        # Rule 2: Remove emphatic particles (only when they don't change meaning)
        # These are SAFE to remove - they're purely emphatic
        self.emphatic_removal_rules = [
            (r'ಮೊಮ್ಮಕ್ಕಳೂ', r'ಮೊಮ್ಮಕ್ಕಳು'),    # Grandchildren (emphatic ū → u)
            (r'ವನ್ನೂ', r'ವನ್ನು'),              # Object marker emphatic
            (r'ಗಳೂ', r'ಗಳು'),                    # Plural emphatic (when safe)
            # Note: Don't remove ವೇ → only if confirmed emphatic
        ]
        
        # Rule 3: Separate foreign words (English loanwords)
        self.foreign_separation_rules = [
            # English compounds that should have spaces
            (r'ಎಸಿವೆಂಟ್', r'ಎಸಿ ವೆಂಟ್'),        # AC vent
            (r'ಫೇಸ್‌ಬುಕ್', r'ಫೇಸ್ ಬುಕ್'),      # Facebook
            (r'ಬಟನ್ಗಳು', r'ಬಟನ್ ಗಳು'),          # Buttons (keep English stem separate)
            # But NOT these - they're established loanwords:
            # ಬಸ್, ಕಾರ್, ಫೋನ್ (treat as single words)
        ]
        
        # Compile patterns
        self.compound_patterns = [(re.compile(p), r) for p, r in self.compound_fusion_rules]
        self.emphatic_patterns = [(re.compile(p), r) for p, r in self.emphatic_removal_rules]
        self.foreign_patterns = [(re.compile(p), r) for p, r in self.foreign_separation_rules]
    
    def normalize_unicode(self, text: str) -> str:
        """Apply Unicode NFC normalization."""
        return unicodedata.normalize('NFC', text)
    
    def fuse_compounds(self, text: str) -> str:
        """Apply compound fusion rules."""
        for pattern, replacement in self.compound_patterns:
            text = pattern.sub(replacement, text)
        return text
    
    def remove_safe_emphatics(self, text: str) -> str:
        """Remove emphatic particles that don't change meaning."""
        for pattern, replacement in self.emphatic_patterns:
            text = pattern.sub(replacement, text)
        return text
    
    def separate_foreign_words(self, text: str) -> str:
        """Separate English loanword compounds."""
        for pattern, replacement in self.foreign_patterns:
            text = pattern.sub(replacement, text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Clean whitespace."""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def normalize(self, text: str) -> str:
        """Apply all orthographic style rules."""
        if not text:
            return text
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Fuse compound words (PRIORITY 1)
        text = self.fuse_compounds(text)
        
        # Step 3: Remove safe emphatic particles
        text = self.remove_safe_emphatics(text)
        
        # Step 4: Separate foreign words
        text = self.separate_foreign_words(text)
        
        # Step 5: Clean whitespace
        text = self.normalize_whitespace(text)
        
        return text
    
    def get_style_guide_documentation(self) -> str:
        """Return documentation of style rules."""
        doc = """
KANNADA ORTHOGRAPHIC STYLE GUIDE
=================================

1. COMPOUND WORDS - PREFER FUSED
   ✓ ಹಠಕ್ಕೆ (not ಹಠ ಕ್ಕೆ)
   ✓ ಉಗಮಸ್ಥಾನ (not ಉಗಮ ಸ್ಥಾನ)
   ✓ ರಾಮಪ್ರಸಾದ (not ರಾಮ ಪ್ರಸಾದ)
   
   Rationale: Reduces word count variance, follows traditional Kannada orthography

2. EMPHATIC PARTICLES - PREFER UNMARKED
   ✓ ಮೊಮ್ಮಕ್ಕಳು (not ಮೊಮ್ಮಕ್ಕಳೂ)
   ✓ ವಿಷಯವನ್ನು (not ವಿಷಯವನ್ನೂ)
   
   Rationale: Emphatics are prosodic, not semantic - ASR shouldn't be penalized
   Exception: When emphatic changes meaning (contrastive, not yet implemented)

3. FOREIGN WORDS - PREFER SEPARATED
   ✓ ಎಸಿ ವೆಂಟ್ (not ಎಸಿವೆಂಟ್)
   ✓ ಫೇಸ್ ಬುಕ್ (not ಫೇಸ್ಬುಕ್)
   
   Rationale: English compounds retain word boundaries
   Exception: Established loanwords (ಬಸ್, ಕಾರ್, ಫೋನ್)

4. SPELLING STANDARDIZATION
   ✓ ವಿರುದ್ಧ (not ವಿರುದ್ದ)
   
   Rationale: Consistent orthography across datasets

EXCEPTIONS NOT IMPLEMENTED (require semantic analysis):
- Contrastive emphatics (ನಾನೇ vs ನಾನು)
- Disambiguating compounds (context-dependent)
"""
        return doc


if __name__ == "__main__":
    normalizer = OrthographicStyleNormalizer()
    
    # Test cases from actual data
    test_cases = [
        # Compound fusion
        ("ಹಠ ಕ್ಕೆ", "ಹಠಕ್ಕೆ"),
        ("ರಾಮ ಪ್ರಸಾದ", "ರಾಮಪ್ರಸಾದ"),
        ("ಉಗಮ ಸ್ಥಾನ", "ಉಗಮಸ್ಥಾನ"),
        ("ವಿಷಯ ವನ್ನು", "ವಿಷಯವನ್ನು"),
        ("ದಿನಾಚರಣೆ ಯಂದು", "ದಿನಾಚರಣೆಯಂದು"),
        
        # Emphatic removal
        ("ಮೊಮ್ಮಕ್ಕಳೂ", "ಮೊಮ್ಮಕ್ಕಳು"),
        ("ವಿಷಯವನ್ನೂ", "ವಿಷಯವನ್ನು"),
        
        # Foreign word separation
        ("ಎಸಿವೆಂಟ್ಗಳಿವೆ", "ಎಸಿ ವೆಂಟ್ ಗಳಿವೆ"),
        
        # Spelling standardization
        ("ವಿರುದ್ದ", "ವಿರುದ್ಧ"),
    ]
    
    print("ORTHOGRAPHIC STYLE GUIDE NORMALIZATION TESTS")
    print("=" * 70)
    
    for original, expected in test_cases:
        result = normalizer.normalize(original)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{original}' → '{result}' (expected: '{expected}')")
    
    print("\n" + "=" * 70)
    print(normalizer.get_style_guide_documentation())

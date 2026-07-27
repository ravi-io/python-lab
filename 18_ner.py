"""
Program 18: Named Entity Recognition (NER)
Time Complexity: O(n * m) where n=tokens, m=rules/patterns
Space Complexity: O(n)
"""

import re


class SimpleNER:
    """Rule-based Named Entity Recognizer."""

    def __init__(self):
        # Known entity dictionaries
        self.persons = {
            'john', 'jane', 'alice', 'bob', 'charlie', 'david',
            'einstein', 'newton', 'tesla', 'gandhi', 'obama', 'modi'
        }
        self.organizations = {
            'google', 'microsoft', 'apple', 'amazon', 'facebook',
            'nasa', 'who', 'un', 'mit', 'stanford', 'iit', 'ieee'
        }
        self.locations = {
            'india', 'usa', 'london', 'paris', 'tokyo', 'delhi',
            'new york', 'san francisco', 'mumbai', 'bangalore',
            'california', 'europe', 'asia'
        }
        # Title patterns indicating PERSON
        self.titles = {'mr', 'mrs', 'ms', 'dr', 'prof', 'president'}

    def _is_capitalized(self, word):
        """Check if word starts with capital letter."""
        return word[0].isupper() if word else False

    def recognize(self, text):
        """Perform NER on text, return list of (entity, label) tuples."""
        entities = []
        tokens = text.split()
        i = 0

        while i < len(tokens):
            word = tokens[i]
            word_lower = word.lower().strip('.,!?;:')

            # Check for multi-word locations
            if i + 1 < len(tokens):
                two_word = f"{word_lower} {tokens[i+1].lower().strip('.,!?;:')}"
                if two_word in self.locations:
                    entities.append((f"{word} {tokens[i+1].strip('.,!?;:')}", "LOCATION"))
                    i += 2
                    continue

            # Check titles (next word is PERSON)
            if word_lower in self.titles and i + 1 < len(tokens):
                name = tokens[i + 1].strip('.,!?;:')
                entities.append((f"{word} {name}", "PERSON"))
                i += 2
                continue

            # Check known entities
            if word_lower in self.persons:
                entities.append((word.strip('.,!?;:'), "PERSON"))
            elif word_lower in self.organizations:
                entities.append((word.strip('.,!?;:'), "ORGANIZATION"))
            elif word_lower in self.locations:
                entities.append((word.strip('.,!?;:'), "LOCATION"))
            # Date pattern (DD/MM/YYYY or YYYY-MM-DD)
            elif re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', word):
                entities.append((word, "DATE"))
            elif re.match(r'\d{4}-\d{2}-\d{2}', word):
                entities.append((word, "DATE"))
            # Email pattern
            elif re.match(r'[\w.+-]+@[\w-]+\.[\w.-]+', word):
                entities.append((word, "EMAIL"))
            # Money pattern
            elif re.match(r'[\$€£]\d+', word) or re.match(r'\d+[\$€£]', word):
                entities.append((word, "MONEY"))
            # Capitalized word (potential entity) not at start of sentence
            elif self._is_capitalized(word) and i > 0:
                entities.append((word.strip('.,!?;:'), "ENTITY"))

            i += 1

        return entities

    def annotate(self, text):
        """Return text with entity annotations."""
        entities = self.recognize(text)
        annotated = text
        for entity, label in entities:
            annotated = annotated.replace(entity, f"[{entity}/{label}]", 1)
        return annotated


# Driver Code
if __name__ == "__main__":
    print("=== Named Entity Recognition (NER) ===\n")

    ner = SimpleNER()

    test_sentences = [
        "Dr. Einstein worked at NASA in California.",
        "Alice visited Google headquarters in San Francisco.",
        "Mr. Modi met President Obama in India on 15/08/2023.",
        "Send the report to alice@google.com by 2024-01-15.",
        "Microsoft invested $500 in the MIT research project.",
        "Bob and Charlie went to London for the IEEE conference.",
    ]

    for sentence in test_sentences:
        print(f"Input: {sentence}")
        entities = ner.recognize(sentence)
        if entities:
            for entity, label in entities:
                print(f"  [{entity}] -> {label}")
        print(f"Annotated: {ner.annotate(sentence)}\n")

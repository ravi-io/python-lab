"""
Program 14: Chatbot using NLP (Pattern Matching + TF-IDF)
Time Complexity: O(n * m) for matching, n=patterns, m=words
Space Complexity: O(n * m)
"""

import re
import random
from collections import defaultdict
import math


class Chatbot:
    """Simple NLP Chatbot using pattern matching and TF-IDF similarity."""

    def __init__(self, name="KBot"):
        self.name = name
        self.intents = {}
        self.vocab = set()
        self.idf = {}
        self.intent_vectors = {}

    def add_intent(self, intent_name, patterns, responses):
        """Add an intent with patterns and responses."""
        self.intents[intent_name] = {
            'patterns': patterns,
            'responses': responses
        }

    def _tokenize(self, text):
        """Tokenize and normalize text."""
        return re.findall(r'[a-z]+', text.lower())

    def _compute_tf(self, tokens):
        """Compute term frequency."""
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        total = len(tokens)
        return {t: c / total for t, c in tf.items()}

    def train(self):
        """Build TF-IDF vectors for all patterns."""
        all_docs = []
        self.doc_intents = []

        for intent_name, data in self.intents.items():
            for pattern in data['patterns']:
                tokens = self._tokenize(pattern)
                all_docs.append(tokens)
                self.doc_intents.append(intent_name)
                self.vocab.update(tokens)

        # Compute IDF
        n_docs = len(all_docs)
        doc_freq = defaultdict(int)
        for doc in all_docs:
            for word in set(doc):
                doc_freq[word] += 1

        self.idf = {word: math.log(n_docs / (df + 1))
                    for word, df in doc_freq.items()}

        # Compute TF-IDF vectors for each document
        self.doc_vectors = []
        for doc in all_docs:
            tf = self._compute_tf(doc)
            vector = {word: tf.get(word, 0) * self.idf.get(word, 0)
                      for word in self.vocab}
            self.doc_vectors.append(vector)

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in self.vocab)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0
        return dot / (mag1 * mag2)

    def get_response(self, user_input):
        """Find best matching intent and return response."""
        tokens = self._tokenize(user_input)
        if not tokens:
            return "I didn't understand that. Could you rephrase?"

        tf = self._compute_tf(tokens)
        input_vector = {word: tf.get(word, 0) * self.idf.get(word, 0)
                        for word in self.vocab}

        # Find most similar pattern
        best_score = 0
        best_intent = None
        for i, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(input_vector, doc_vec)
            if score > best_score:
                best_score = score
                best_intent = self.doc_intents[i]

        if best_score < 0.1:
            return "I'm not sure I understand. Can you try asking differently?"

        return random.choice(self.intents[best_intent]['responses'])


# Driver Code
if __name__ == "__main__":
    print("=== NLP Chatbot ===\n")

    bot = Chatbot("KBot")

    # Define intents
    bot.add_intent("greeting",
        patterns=["hello", "hi", "hey", "good morning", "good evening"],
        responses=["Hello! How can I help you?", "Hi there!", "Hey! What's up?"]
    )
    bot.add_intent("goodbye",
        patterns=["bye", "goodbye", "see you", "quit", "exit"],
        responses=["Goodbye! Have a great day!", "See you later!", "Bye!"]
    )
    bot.add_intent("name",
        patterns=["what is your name", "who are you", "your name"],
        responses=["I'm KBot, your AI assistant!", "My name is KBot."]
    )
    bot.add_intent("help",
        patterns=["help me", "I need help", "can you help", "assist me"],
        responses=["Sure! I'm here to help. What do you need?",
                   "Of course! What can I assist you with?"]
    )
    bot.add_intent("weather",
        patterns=["weather today", "how is weather", "temperature outside"],
        responses=["I can't check real weather, but I hope it's nice!",
                   "I don't have weather data, but stay prepared!"]
    )
    bot.add_intent("thanks",
        patterns=["thank you", "thanks", "appreciate it"],
        responses=["You're welcome!", "Happy to help!", "Anytime!"]
    )

    bot.train()

    # Interactive chat
    print(f"{bot.name}: Hello! I'm {bot.name}. Type 'quit' to exit.\n")

    test_inputs = [
        "Hi there!",
        "What's your name?",
        "Can you help me with something?",
        "How is the weather?",
        "Thank you so much!",
        "Goodbye!",
    ]

    for user_input in test_inputs:
        response = bot.get_response(user_input)
        print(f"  You: {user_input}")
        print(f"  {bot.name}: {response}\n")

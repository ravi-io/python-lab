# Python Lab - MCA Semester 2 (AI & ML Programs)

## Programs

| # | Topic | File | Time Complexity | Space Complexity |
|---|-------|------|-----------------|------------------|
| 1 | [Uninformed Search (BFS, DFS)](./1_uninformed_search.py) | `1_uninformed_search.py` | O(V + E) | O(V) |
| 2 | [Informed Search (A*)](./2_astar_search.py) | `2_astar_search.py` | O(E log V) | O(V) |
| 3 | [8-Puzzle Problem](./3_eight_puzzle.py) | `3_eight_puzzle.py` | O(b^d) | O(b^d) |
| 4 | [Constraint Satisfaction Problem](./4_csp.py) | `4_csp.py` | O(d^n) | O(n) |
| 5 | [Hill Climbing Algorithm](./5_hill_climbing.py) | `5_hill_climbing.py` | O(n × m) | O(1) |
| 6 | [Genetic Algorithm](./6_genetic_algorithm.py) | `6_genetic_algorithm.py` | O(g × p × n) | O(p × n) |
| 7 | [Decision Tree Learning](./7_decision_tree.py) | `7_decision_tree.py` | O(n × m × d) | O(n × m) |
| 8 | [Naive Bayes Classifier](./8_naive_bayes.py) | `8_naive_bayes.py` | O(n × m) | O(m × k) |
| 9 | [Perceptron Learning](./9_perceptron.py) | `9_perceptron.py` | O(epochs × n × m) | O(m) |
| 10 | [Feedforward Neural Network](./10_feedforward_nn.py) | `10_feedforward_nn.py` | O(epochs × n × L × w) | O(L × w) |
| 11 | [Text Classification (Scikit-learn)](./11_text_classification.py) | `11_text_classification.py` | O(n × m) | O(n × m) |
| 12 | [Sentiment Analysis](./12_sentiment_analysis.py) | `12_sentiment_analysis.py` | O(n × m) | O(V) |
| 13 | [Image Classification (Keras)](./13_image_classification.py) | `13_image_classification.py` | O(epochs × n × L × w) | O(L × w) |
| 14 | [Chatbot using NLP](./14_chatbot.py) | `14_chatbot.py` | O(n × m) | O(n × m) |
| 15 | [Reinforcement Learning (Q-Learning)](./15_q_learning.py) | `15_q_learning.py` | O(episodes × steps) | O(S × A) |
| 16 | [K-Means Clustering](./16_kmeans.py) | `16_kmeans.py` | O(n × k × d × i) | O(n × d) |
| 17 | [Regression Analysis](./17_regression.py) | `17_regression.py` | O(n × m²) | O(n × m) |
| 18 | [Named Entity Recognition](./18_ner.py) | `18_ner.py` | O(n × m) | O(n) |
| 19 | [Decision Boundary Visualization](./19_decision_boundaries.py) | `19_decision_boundaries.py` | O(n × k × grid²) | O(grid²) |
| 20 | [Recommendation System](./20_recommendation.py) | `20_recommendation.py` | O(n × m × k) | O(n × m) |

## How to Run

```bash
python3 <filename>.py
```

Example:
```bash
python3 1_uninformed_search.py
python3 15_q_learning.py
```

## Requirements

### Programs 1–10, 12, 14–18, 20 (No external dependencies)
- Python 3.6+ (standard library only)

### Programs 11, 19 (Scikit-learn)
```bash
pip install scikit-learn numpy matplotlib
```

### Program 13 (Keras/TensorFlow)
```bash
pip install tensorflow
```

## Complexity Legend

| Symbol | Meaning |
|--------|---------|
| V | Vertices/Nodes |
| E | Edges |
| b | Branching factor |
| d | Depth/Dimensions |
| n | Number of samples/variables |
| m | Number of features/neighbors |
| p | Population size |
| g | Generations |
| k | Number of classes/clusters/neighbors |
| L | Number of layers |
| w | Neurons per layer |
| S | Number of states |
| A | Number of actions |
| i | Iterations |
| V | Vocabulary size |

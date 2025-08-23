# Term-Document Matrix and Word Embeddings Tutorial
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
top_k = 30  # Vocabulary size limit

print("=== TERM-DOCUMENT MATRIX AND WORD EMBEDDINGS TUTORIAL ===\n")

# 1. Create 10 small documents with diverse vocabulary
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning algorithms process vast amounts of data efficiently.",
    "Climate change affects ocean temperatures and marine ecosystems globally.",
    "Artificial intelligence revolutionizes healthcare diagnostics and patient care.",
    "Quantum computing promises exponential speedups for complex calculations.",
    "Space exploration missions discover new planets and cosmic phenomena.",
    "Renewable energy sources reduce carbon emissions and environmental impact.",
    "Social media platforms connect billions of users worldwide daily.",
    "Cryptocurrency blockchain technology enables decentralized financial transactions.",
    "Virtual reality creates immersive experiences for education and entertainment."
]

print("=== 1. DOCUMENTS ===")
for i, doc in enumerate(documents):
    print(f"Doc {i}: {doc}")
print()

# 2. Build Term-Document Matrix using CountVectorizer
print("=== 2. BUILDING TERM-DOCUMENT MATRIX ===")
vectorizer = CountVectorizer(
    max_features=top_k,           # Limit to top_k most frequent terms
    lowercase=False,              # Keep original case
    stop_words=None,              # No stopword removal
    token_pattern=r"(?u)\b\w+\b"  # Simple word boundary tokenization
)

# Fit and transform documents
tdm_sparse = vectorizer.fit_transform(documents)
tdm_dense = tdm_sparse.toarray()

print(f"TDM shape: {tdm_dense.shape} (documents × terms)")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
print()

# 3. Display vocabulary and TDM as DataFrame
print("=== 3. VOCABULARY ===")
vocabulary = vectorizer.get_feature_names_out()
print("Selected vocabulary (in order):")
print(list(vocabulary))
print()

print("=== 4. TERM-DOCUMENT MATRIX (DataFrame) ===")
tdm_df = pd.DataFrame(tdm_dense, columns=vocabulary)
tdm_df.index = [f"Doc_{i}" for i in range(len(documents))]
print("TDM DataFrame (first 10 rows):")
print(tdm_df.head(10))
print()

# 4. Compute term frequency table
print("=== 5. TERM FREQUENCY TABLE ===")
term_frequencies = tdm_dense.sum(axis=0)  # Sum across documents
term_freq_dict = dict(zip(vocabulary, term_frequencies))
term_freq_sorted = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)

print("Term frequencies (sorted descending):")
for i, (term, freq) in enumerate(term_freq_sorted[:30]):
    print(f"{i+1:2d}. {term}: {freq}")
print()

# 5. Extract word embeddings from TDM
print("=== 6. WORD EMBEDDINGS FROM TDM ===")
print("Each word's embedding is its column vector in the TDM:")
print("- Each dimension represents a document")
print("- Value = frequency of the word in that document")
print()

# Create word embeddings dictionary
word_embeddings = {}
for i, term in enumerate(vocabulary):
    word_embeddings[term] = tdm_dense[:, i]  # Column vector for this term

# Display embeddings for 5 sample terms
sample_terms = list(vocabulary)[:5]  # First 5 terms
print("Sample word embeddings:")
for term in sample_terms:
    embedding = word_embeddings[term]
    print(f"{term}: {embedding} (shape: {embedding.shape})")
print()

print(f"Total word embeddings extracted: {len(word_embeddings)}")
print(f"Each embedding has {len(documents)} dimensions (one per document)")
print()

# 6. Save outputs to files
print("=== 7. SAVING OUTPUTS ===")

# Save documents to CSV
docs_df = pd.DataFrame({'document_id': range(len(documents)), 'text': documents})
docs_df.to_csv('docs.csv', index=False)
print("✓ Saved documents to 'docs.csv'")

# Save TDM to CSV
tdm_df.to_csv('tdm.csv', index=True)
print("✓ Saved TDM to 'tdm.csv'")

# Save vocabulary to text file
with open('vocab.txt', 'w') as f:
    for term in vocabulary:
        f.write(f"{term}\n")
print("✓ Saved vocabulary to 'vocab.txt'")

print("\n=== SUMMARY ===")
print(f"• Created {len(documents)} documents with diverse vocabulary")
print(f"• Built {tdm_dense.shape[0]}×{tdm_dense.shape[1]} term-document matrix")
print(f"• Extracted {len(word_embeddings)} word embeddings")
print(f"• Each word embedding is a {len(documents)}-dimensional vector")
print("• Word embeddings represent term frequency patterns across documents")
print("• Files saved: docs.csv, tdm.csv, vocab.txt")
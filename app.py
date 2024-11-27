import json
import os
import glob
import faiss
import torch
import openai
from groq import Groq
from typing import List, Dict, Any
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
import ollama  # Import ollama
import numpy as np  

USE_GROQ = True  # Else use Ollama
USE_OPENAI = True  # Else use Ollama


class SearchEmbeddings:
    def __init__(self, model_name: str = 'dunzhang/stella_en_1.5B_v5'):
        """Initialize the search embeddings with the Stella model"""
        # Load Stella embedding model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        self.model.to(self.device)

        # Dictionaries to hold multiple indices and their chunks
        self.indices = {}        # e.g., {'a': faiss_index_a, 'b': faiss_index_b}
        self.chunks_maps = {}    # e.g., {'a': {0: chunk0, 1: chunk1, ...}, 'b': {...}}

    def encode_text(self, text: str) -> np.ndarray:
        """Generate normalized embeddings for a given text using Stella model"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        # Normalize the embedding
        embedding = normalize(embedding)
        return embedding[0]

    def load_embeddings(self, embeddings_dir: str, chunks_dir: str = '.'):
        """Load all FAISS indices and reconstruct their corresponding chunks mappings"""
        # Find all FAISS index files in the embeddings directory
        index_files = glob.glob(os.path.join(embeddings_dir, '*_faiss_index.bin'))
        if not index_files:
            print("❌ No FAISS index files found in the embeddings directory.")
            return

        for index_file in index_files:
            # Extract identifier from filename, e.g., 'a' from 'a_faiss_index.bin'
            base_name = os.path.basename(index_file)
            identifier = base_name.split('_faiss_index.bin')[0]
            try:
                # Load FAISS index
                faiss_index = faiss.read_index(index_file)
                self.indices[identifier] = faiss_index
                print(f"✅ FAISS index '{identifier}' loaded with {faiss_index.ntotal} entries.")
            except Exception as e:
                print(f"❌ Failed to load FAISS index from {index_file}: {e}")
                continue

            # Load corresponding chunks
            chunks = self.load_chunks_in_order(chunks_dir)
            if not chunks:
                print(f"❌ No chunks loaded for index '{identifier}'.")
                continue

            # Build chunks_map with index IDs
            chunks_map = {i: chunk for i, chunk in enumerate(chunks)}
            self.chunks_maps[identifier] = chunks_map
            print(f"✅ Loaded {len(chunks_map)} chunks for index '{identifier}'.")

            # Validate synchronization
            if len(chunks_map) != faiss_index.ntotal:
                print(f"⚠️ Warning: Number of chunks ({len(chunks_map)}) doesn't match FAISS index entries ({faiss_index.ntotal}) for index '{identifier}'. Ensure data consistency.")

    def load_chunks_in_order(self, directory: str) -> List[Dict[str, Any]]:
        """Load chunks from NDJSON files in the specified directory."""
        chunks = []
        ndjson_files = sorted(glob.glob(os.path.join(directory, '*_chunks.ndjson')))
        if not ndjson_files:
            print("❌ No '_chunks.ndjson' files found in the chunks directory. Ensure the directory is correct.")
            return chunks  # Return empty list if no files found
        else:
            print(f"🔍 Found {len(ndjson_files)} '_chunks.ndjson' files in directory: {directory}")

        for file_path in ndjson_files:
            print(f"📖 Loading chunks from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            print(f"⚠️ Skipping invalid JSON in {file_path}")
                            continue
        print(f"✅ Total chunks loaded: {len(chunks)}")
        return chunks

    def refine_query(self, query: str, index_summary: str) -> str:
        """Use an LLM to refine the user's query for better similarity search."""
        print("\n📝 Refining the user's query using LLM...")

        prompt = (
            f"Refine the user's query to align with the conventions found in the provided context. "
            f"Ensure the refined query uses specific naming conventions such as uppercase for error codes, "
            f"maintains structural clarity, and emphasizes the main topics and keywords for retrieving the "
            f"most relevant code snippets and documentation.\n\n"
            f"Context:\n{index_summary}\n\n"
            f"Original Query: {query}\n\n"
            f"ONLY RETURN THE REFINED QUERY:"
        )

        messages = [{'role': 'user', 'content': prompt}]
        try:
            if USE_GROQ:
                import dotenv
                dotenv.load_dotenv()
                client = Groq()
                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=messages,
                    stream=False
                )
                refined_query = completion.choices[0].message.content.strip()
            else:
                response = ollama.chat(model='llama3.1:8b', messages=messages)
                refined_query = response['message']['content'].strip()
            print(f"✅ Refined Query: {refined_query}")
            return refined_query
        except Exception as e:
            print(f"❌ Error refining query: {e}")
            # Fallback to the original query if refinement fails
            return query

    def search_similar_chunks(self, query: str, index_identifier: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks in a specific FAISS index and return their content"""
        faiss_index = self.indices.get(index_identifier)
        chunks_map = self.chunks_maps.get(index_identifier)

        if not faiss_index or not chunks_map:
            print(f"❌ FAISS index or chunks_map for '{index_identifier}' not loaded.")
            return []

        # Encode the query using the same method
        query_vector = self.encode_text(query).astype('float32').reshape(1, -1)

        # Search the index
        D, I = faiss_index.search(query_vector, k)

        # Get the chunks
        similar_chunks = []
        for idx, (distance, index) in enumerate(zip(D[0], I[0]), 1):
            if index < 0 or index >= len(chunks_map):
                continue
            chunk = chunks_map[index]
            similar_chunks.append({
                'rank': idx,
                'similarity': float(distance),
                'file': chunk['metadata']['file'],
                'chunk_number': chunk['metadata']['chunk_number'],
                'code': chunk['code'],
                'explanation': chunk['explanation'],
                'metadata': chunk['metadata'],
                'repository_explanation': chunk.get('repository_explanation', ''),
                'file_summary': chunk.get('file_summary', '')
            })

        return similar_chunks

    def answer_query_with_context(self, query: str, k: int = 3) -> Dict[str, str]:
        """Answer the user's query using the top k similar chunks from each index as context."""
        print(f"\n🔍 Processing query for {len(self.indices)} indices: {query}")
        answers = {}

        for identifier in self.indices.keys():
            print(f"\n--- Processing Index '{identifier}' ---")
            try:
                # Provide a summary of what's in the index
                index_summary = (
                    "The index contains detailed code snippets, explanations, documentation, and guides "
                    "from Magic Labs (Magic.link) embedded wallet SDK, including various programming languages "
                    "and frameworks. This includes complete instructions, examples, and best practices for using "
                    "Magic's SDK to integrate authentication and manage wallets."
                )

                # Refine the query using the LLM
                refined_query = self.refine_query(query, index_summary)

                # Search for similar chunks using the refined query
                similar_chunks = self.search_similar_chunks(refined_query, identifier, k=k)
                print(f"✅ Found {len(similar_chunks)} similar chunks in index '{identifier}' using the refined query.")

                # Print details of each chunk
                for i, chunk in enumerate(similar_chunks, 1):
                    print(f"\nChunk {i}:")
                    print(f"File: {chunk['file']}")
                    print(f"Chunk number: {chunk['chunk_number']}")
                    print(f"Similarity score: {chunk['similarity']:.4f}")
                    print("Code:")
                    print(chunk['code'])
                    print("Explanation:")
                    print(chunk['explanation'])
                    if chunk.get('file_summary'):
                        print("File Summary:")
                        print(chunk['file_summary'])
                    if chunk.get('repository_explanation'):
                        print("Repository Explanation:")
                        print(chunk['repository_explanation'])
                    print("-" * 80)

                if not similar_chunks:
                    answers[identifier] = "No relevant context found to answer the query."
                    continue

                # Combine the contents of the similar chunks
                print("\n📖 Combining chunk contents...")
                context = "\n\n".join(
                    f"File: {chunk['file']}\nCode:\n{chunk['code']}\nExplanation:\n{chunk['explanation']}\nFile Summary:\n{chunk.get('file_summary', '')}\nRepository Explanation:\n{chunk.get('repository_explanation', '')}"
                    for chunk in similar_chunks
                )
                print("✅ Context assembled")

                # Prepare the messages for LLM
                print("\n💬 Preparing prompt...")
                prompt = (
                    "As a Magic Labs SDK expert, answer the following question based on the provided context. "
                    "Use the information from the code snippets, explanations, and documentation to provide a "
                    "comprehensive answer.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Answer:"
                )
                messages = [{'role': 'user', 'content': prompt}]
                print(f"\n📜 Prompt prepared for index '{identifier}'.")

                # Generate response using the selected LLM
                if not USE_OPENAI:
                    # Generate response using Ollama
                    print("\n🤖 Generating response using Ollama...")
                    response = ollama.chat(model='llama3.1:8b', messages=messages)
                    answer = response['message']['content'].strip()
                else:
                    # Generate response using OpenAI with Groq fallback
                    print("\n🤖 Generating response using OpenAI...")
                    import dotenv
                    dotenv.load_dotenv()

                    try:
                        response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=messages
                        )
                        answer = response.choices[0].message.content.strip()
                    except Exception as openai_error:
                        print(f"OpenAI failed: {openai_error}, falling back to Groq...")
                        client = Groq()
                        completion = client.chat.completions.create(
                            model="llama-3.1-70b-versatile",
                            messages=messages,
                            stream=False
                        )
                        answer = completion.choices[0].message.content.strip()

                print("✅ Successfully generated response.")
                answers[identifier] = answer

            except Exception as e:
                print(f"❌ Error processing index '{identifier}': {e}")
                answers[identifier] = "An error occurred while generating the response."

        print("\n✅ All answers generated.")
        return answers


def main():
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Search Embeddings and Answer Query')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_dir = os.path.join(current_dir, 'chunks')  # Adjust if chunks are in a different relative path

    parser.add_argument('--embeddings_dir', type=str, default=current_dir,
                        help='Directory containing embeddings and index files (default: current directory)')
    parser.add_argument('--chunks_dir', type=str, default=chunks_dir,
                        help='Directory containing the _chunks.ndjson files (default: ./chunks)')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top chunks to retrieve for answering (default: 3)')
    parser.add_argument('--model_name', type=str, default='dunzhang/stella_en_1.5B_v5',
                        help='Embedding model name (default: dunzhang/stella_en_1.5B_v5)')
    args = parser.parse_args()

    # Initialize the search
    searcher = SearchEmbeddings(model_name=args.model_name)

    # Load the embeddings and reconstruct chunks mappings
    searcher.load_embeddings(args.embeddings_dir, chunks_dir=args.chunks_dir)

    # Check if any indices were loaded
    if not searcher.indices:
        print("❌ No FAISS indices loaded. Ensure that the embeddings_dir contains *_faiss_index.bin files.")
        return

    # Print summary of loaded indices and chunks
    for identifier, faiss_index in searcher.indices.items():
        chunks_map = searcher.chunks_maps.get(identifier, {})
        print(f"\n📂 Index '{identifier}': {faiss_index.ntotal} entries, {len(chunks_map)} chunks loaded.")

    # Interactive query loop
    while True:
        print("\n=== 🛠 Magic Chatbot Query Interface ===")
        query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
        print(f"\n📥 Received query: {query}")

        if query.lower() == 'exit':
            print("\n🚪 Exiting Magic Chatbot...")
            break

        if not query:
            print("\n⚠️ Please enter a valid query.")
            continue

        try:
            # Answer the query using the context from all indices
            answers = searcher.answer_query_with_context(query, k=args.top_k)
            print("\n✅ Generated responses from context.")

            # Print the answers
            for identifier, answer in answers.items():
                print(f"\n=== ✨ Answer from Index '{identifier}' ===")
                print(answer)
                print("\n" + "=" * 80)

        except Exception as e:
            print(f"\n❌ Error occurred during processing: {e}")
            print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

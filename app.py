import json
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer
import faiss
import torch
from sklearn.preprocessing import normalize
import os
import glob
import ollama  # Import ollama
import argparse
from openai import OpenAI
from groq import Groq

# LLM Provider flag - options: 'ollama', 'openai', 'groq'
LLM_PROVIDER = 'openai'

class SearchEmbeddings:
    def __init__(self, model_name: str = 'dunzhang/stella_en_1.5B_v5'):
        """Initialize the search embeddings with the Stella model"""
        self.llm_provider = LLM_PROVIDER
        
        # Initialize API clients based on provider
        if self.llm_provider == 'openai':
            self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.llm_provider == 'groq':
            self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            
        # Load Stella embedding model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # Device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        self.model.to(self.device)

        # Dictionaries to hold multiple indices and their chunks
        self.indices: Dict[str, faiss.Index] = {}        # e.g., {'a': faiss_index_a, 'b': faiss_index_b}
        self.chunks_maps: Dict[str, Dict[int, Dict[str, Any]]] = {}    # e.g., {'a': {0: chunk0, 1: chunk1, ...}, 'b': {...}}

    def encode_text(self, text: str) -> np.ndarray:
        """Generate normalized embeddings for a given text using Stella model"""
        try:
            print(f"🔍 Encoding text: {text[:100]}...")  # Show first 100 chars

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
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            raise

    def load_embeddings(self, embeddings_dir: str, chunks_dir: str = '.'):
        """Load multiple FAISS indices and their corresponding chunks"""
        try:
            print(f"📂 Looking for FAISS indices in: {embeddings_dir}")
            print(f"📂 Directory exists: {os.path.exists(embeddings_dir)}")
            print(f"📂 Directory contents: {os.listdir(embeddings_dir)}")
            
            faiss_index_files = glob.glob(os.path.join(embeddings_dir, '*_faiss_index.bin'))
            faiss_index_files.extend(glob.glob(os.path.join(embeddings_dir, 'faiss_index.bin')))
            
            print(f"🔍 Found FAISS index files: {faiss_index_files}")
            
            if not faiss_index_files:
                print("❌ No FAISS index files found in the embeddings directory.")
                return

            for index_file in faiss_index_files:
                # Extract identifier from filename, e.g., 'a' from 'a_faiss_index.bin'
                base_name = os.path.basename(index_file)
                identifier = base_name.replace('_faiss_index.bin', '').replace('.faiss_index.bin', '')

                # Load FAISS index
                try:
                    faiss_index = faiss.read_index(index_file)
                    self.indices[identifier] = faiss_index
                    print(f"✅ FAISS index '{identifier}' loaded with {faiss_index.ntotal} entries.")
                except Exception as e:
                    print(f"❌ Failed to load FAISS index from {index_file}: {e}")
                    continue

                # Load corresponding chunks
                # Assuming each index has its own chunks file named like 'a_chunks.ndjson' or 'b_chunks.ndjson'
                chunks_file_pattern = os.path.join(chunks_dir, f"*_chunks.ndjson")
                chunks_files = glob.glob(chunks_file_pattern)
                if not chunks_files:
                    print(f"❌ No chunks file found for index '{identifier}' with pattern '{chunks_file_pattern}'.")
                    continue
                else:
                    print(f"🔍 Found chunks file(s) for index '{identifier}': {chunks_files}")

                # Load chunks from the corresponding file(s)
                chunks = []
                for chunks_file in chunks_files:
                    print(f"📖 Loading chunks from: {chunks_file}")
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    chunk['source_file'] = os.path.basename(chunks_file)  # Add source file info
                                    chunks.append(chunk)
                                except json.JSONDecodeError:
                                    print(f"⚠️ Skipping invalid JSON in {chunks_file}")
                                    continue
                print(f"✅ Total chunks loaded for index '{identifier}': {len(chunks)}")
    
                # Build chunks_map with index IDs
                chunks_map = {i: chunk for i, chunk in enumerate(chunks)}
                self.chunks_maps[identifier] = chunks_map
                print(f"✅ Loaded {len(chunks_map)} chunks for index '{identifier}'.")

                # Validate synchronization
                if len(chunks_map) != faiss_index.ntotal:
                    print(f"⚠️ Warning: Number of chunks ({len(chunks_map)}) doesn't match FAISS index entries ({faiss_index.ntotal}) for index '{identifier}'. Ensure data consistency.")
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise

    def refine_query(self, query: str, index_summary: str) -> str:
        """Use an LLM to refine the user's query for better similarity search."""
        print("\n📝 Refining the user's query using LLM...")

        system_instructions = (
            "You are a highly skilled assistant specialized in refining user queries to enhance the effectiveness of similarity searches. "
            "Your task is to transform the original query into a more precise and detailed version without altering its original intent. "
            "Ensure that the refined query includes all essential keywords and maintains clarity to facilitate accurate retrieval of relevant information."
            "Only return the refined query, no other text or comments."
        )

        user_message = (
            f"Context:\n{index_summary}\n\n"
            f"Original Query: {query}\n\n"
            f"Provide a refined version of the original query that is more specific and clear to improve the quality of search results."
        )

        messages = [
            {'role': 'system', 'content': system_instructions},
            {'role': 'user', 'content': user_message}
        ]

        try:
            if self.llm_provider == 'ollama':
                response = ollama.chat(model='llama3.1:8b', messages=messages)
                refined_query = response['message']['content'].strip()
            elif self.llm_provider == 'openai':
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                refined_query = response.choices[0].message.content.strip()
            elif self.llm_provider == 'groq':
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=messages
                )
                refined_query = response.choices[0].message.content.strip()
            
            print(f"✅ Refined Query: {refined_query}")
            return refined_query
        except Exception as e:
            print(f"❌ Error refining query: {e}")
            # Fallback to the original query if refinement fails
            return query

    def get_github_link(self, chunk_filename: str) -> str:
            """
            Convert a chunk filename to its corresponding GitHub repository URL.
            
            Expected chunk_filename format: 'magiclabs_{repo_name}_chunks.ndjson'
            Example: 'magiclabs_fortmatic-ios-pod_chunks.ndjson' -> 'https://github.com/magiclabs/fortmatic-ios-pod'
            """
            prefix = 'magiclabs_'
            suffix = '_chunks.ndjson'
            if chunk_filename.startswith(prefix) and chunk_filename.endswith(suffix):
                repo_name = chunk_filename[len(prefix):-len(suffix)]
                return f"https://github.com/magiclabs/{repo_name}"
            else:
                return 'Unknown Repository'
            
    def search_similar_chunks(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks across all FAISS indices and aggregate the results"""
        if not self.indices:
            raise ValueError("No FAISS indices loaded. Call load_embeddings first.")

        # Encode the query once
        query_vector = self.encode_text(query).astype('float32').reshape(1, -1)

        aggregated_chunks = []

        for identifier, faiss_index in self.indices.items():
            print(f"\n🔍 Searching in index '{identifier}'...")
            D, I = faiss_index.search(query_vector, k)

            chunks_map = self.chunks_maps.get(identifier, {})
            for idx, (distance, index) in enumerate(zip(D[0], I[0]), 1):
                if index < 0 or index >= len(chunks_map):
                    continue
                chunk = chunks_map[index]
                aggregated_chunks.append({
                    'rank': idx,
                    'similarity': float(distance),
                    'file': chunk['metadata']['file'],
                    'chunk_number': chunk['metadata']['chunk_number'],
                    'code': chunk['code'],
                    'explanation': chunk['explanation'],
                    'metadata': chunk['metadata'],
                    'repository_explanation': chunk.get('repository_explanation', ''),
                    'file_summary': chunk.get('file_summary', ''),
                    'index': identifier,            # To know which index it came from
                    'source_file': chunk.get('source_file', 'Unknown')  # To know which NDJSON file it came from
                })

        # Sort aggregated_chunks based on similarity (assuming higher similarity is better)
        aggregated_chunks = sorted(aggregated_chunks, key=lambda x: x['similarity'], reverse=True)

        # Keep only top_k across all indices
        top_k_chunks = aggregated_chunks[:k]

        return top_k_chunks

    def answer_query_with_context(self, query: str, k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Answer the user's query using the top k similar chunks as context."""
        print(f"\n🔍 Refining the query and searching for top {k} similar chunks for query: {query}")
        try:
            # Provide a summary of what's in the indices
            index_summary = (
                "The indices contain detailed code snippets, explanations, documentation, and guides "
                "from Magic Labs (Magic.link) embedded wallet SDK, including various programming languages "
                "and frameworks. This includes complete instructions, examples, and best practices for using "
                "Magic's SDK to integrate authentication and manage wallets."
            )

            # Refine the query using the LLM
            refined_query = self.refine_query(query, index_summary)

            # Search for similar chunks using the refined query
            similar_chunks = self.search_similar_chunks(refined_query, k=k)
            print(f"✅ Found {len(similar_chunks)} similar chunks using the refined query.")

            # Print details of each chunk
            '''
            for i, chunk in enumerate(similar_chunks, 1):
                print(f"\nChunk {i}:")
                print(f"Index: {chunk['index']}")
                print(f"Source File: {chunk['source_file']}")
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
            '''

            if not similar_chunks:
                return "No relevant context found to answer the query.", []

            # Combine the contents of the similar chunks
            print("\n📖 Combining chunk contents...")
            context = "\n\n".join(
                f"Index: {chunk['index']}\nFile: {chunk['file']}\nCode:\n{chunk['code']}\nExplanation:\n{chunk['explanation']}\nFile Summary:\n{chunk.get('file_summary', '')}\nRepository Explanation:\n{chunk.get('repository_explanation', '')}"
                for chunk in similar_chunks
            )
            print("✅ Context assembled")

            # Prepare the messages for Ollama with a detailed system prompt
            print(f"\n💬 Preparing prompt for {self.llm_provider}...")
            system_prompt = (
                "You are an expert in Magic Labs SDK with extensive knowledge of its functionalities and integrations. "
                "Your task is to provide clear, detailed, and accurate answers to user queries using the provided context. "
                "Ensure that your responses are comprehensive, referencing specific sections of the context where applicable, "
                "and maintain a professional and informative tone."
            )
            user_prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            #print(f"\n\n📜 System Prompt:\n{system_prompt}")
            #print(f"\n\n📜 User Prompt:\n{user_prompt}")

            # Generate a response using Ollama
            print(f"\n🤖 Generating response using {self.llm_provider}...")
            if self.llm_provider == 'ollama':
                response = ollama.chat(model='llama3.1:8b', messages=messages)
                answer = response['message']['content'].strip()
            elif self.llm_provider == 'openai':
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                answer = response.choices[0].message.content.strip()
            elif self.llm_provider == 'groq':
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=messages
                )
                answer = response.choices[0].message.content.strip()
            
            print("✅ Successfully generated response")
        except Exception as e:
            print(f"❌ Error: {e}")
            answer = "An error occurred while generating the response."
            similar_chunks = []

        print("\n✅ Returning answer...")
        return answer, similar_chunks


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Search Embeddings and Answer Query')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_dir = os.path.join(current_dir, 'chunks')
    embeddings_dir = os.path.join(current_dir, 'embeddings')

    parser.add_argument('--embeddings_dir', type=str, default=embeddings_dir,
                        help='Directory containing embeddings and index files (default: ./embeddings)')
    parser.add_argument('--chunks_dir', type=str, default=chunks_dir,
                        help='Directory containing the _chunks.ndjson files (default: ./chunks)')
    parser.add_argument('--top_k', type=int, default=4,
                        help='Number of top chunks to retrieve for answering (default: 3)')
    parser.add_argument('--model_name', type=str, default='dunzhang/stella_en_1.5B_v5',
                        help='Embedding model name (default: dunzhang/stella_en_1.5B_v5)')
    args = parser.parse_args()

    # Initialize the search
    searcher = SearchEmbeddings(model_name=args.model_name)

    # Load the embeddings and reconstruct chunks mappings
    searcher.load_embeddings(args.embeddings_dir, chunks_dir=args.chunks_dir)

    # Verify data loading
    total_indices = len(searcher.indices)
    total_chunks = sum(len(chunks_map) for chunks_map in searcher.chunks_maps.values())
    print(f"\nNumber of FAISS indices loaded: {total_indices}")
    print(f"Total number of chunks loaded: {total_chunks}")

    if total_indices == 0 or total_chunks == 0:
        print("❌ No data loaded. Ensure the embeddings and chunks mapping are correctly set up.")
        return

    # Loop to allow multiple queries
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
            # Answer the query using the context
            answer, similar_chunks = searcher.answer_query_with_context(query, k=args.top_k)
            print("\n✅ Generated response from context.")
            # Print the answer
            print("\n=== ✨ Answer ===")
            print(answer)
            print("\n=== 📝 Source Details ===")
            for i, chunk in enumerate(similar_chunks, 1):
                github_link = searcher.get_github_link(chunk['source_file'])
                print(f"\nChunk {i}:")
                print(f"  Index: {chunk['index']}")
                print(f"  Source File: {chunk['source_file']}")
                print(f"  File: {chunk['file']}")
                print(f"  Chunk Number: {chunk['chunk_number']}")
                print(f"  GitHub Link: {github_link}")
                print("-" * 40)

            print("\n" + "=" * 80)

        except Exception as e:
            print(f"\n❌ Error occurred during processing: {e}")
            print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

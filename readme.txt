# AI-Powered Documentation Search

A powerful search tool that uses AI embeddings to search through documentation and provide contextual answers using various LLM providers.

## Features

- Semantic search using AI embeddings
- Multiple LLM provider support (OpenAI, Groq, Ollama)
- Contextual answers from documentation
- GitHub link generation for source references
- Configurable search parameters
- Error handling and graceful fallbacks

## Prerequisites

- Python 3.x
- Access to one of the supported LLM providers:
  - OpenAI API key
  - Groq API key
  - Ollama local installation

## Installation

1. Clone the repository:

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
env
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here

## Configuration

- Set the default LLM provider in `app.py`:
    ```python
    LLM_PROVIDER = 'openai'  # or 'groq' or 'ollama'
    ```

## Usage

You can interact with the tool in two ways:

### 1. Interactive Mode
Run the script without a query.txt file:
```bash
python app.py --llm-provider openai --top-k 3
```
This will start an interactive session where you can type queries and receive responses in real-time.

### 2. Query File Mode
Create a `query.txt` file in the root directory with your query:
```bash
echo "How do I implement Magic SDK authentication?" > query.txt
python app.py
```
The tool will:
- Automatically detect and read the query from query.txt
- Process the query and generate a response
- Display the answer along with relevant source references
- Exit after processing the query

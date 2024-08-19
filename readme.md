# Azure ML RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot specialized in Azure Machine Learning. It combines the power of large language models with a custom knowledge base to provide accurate and context-aware responses to queries about Azure ML.

## Features

- **RAG-based Responses**: Utilizes up-to-date Azure ML documentation for accurate answers.
- **Conversational Memory**: Maintains context throughout the conversation for more coherent interactions.
- **GPU-Accelerated**: Optimized for performance on machines with GPU capabilities.
- **User-Friendly Interface**: Built with Streamlit for an intuitive chat experience.

## Tech Stack

- **Python**: Core programming language
- **TensorFlow & TensorFlow Hub**: For generating embeddings
- **ChromaDB**: Vector database for efficient information retrieval
- **Anthropic's Claude API**: Large language model for natural language processing
- **Streamlit**: Web application framework for the user interface

## Prerequisites

- Python 3.10+
- Conda (recommended for environment management)
- GPU support (optional, but recommended for better performance)

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/azure-ml-rag-chatbot.git
   cd azure-ml-rag-chatbot
   ```

2. Create and activate a Conda environment:

   ```
   conda create -n rag-chatbot python=3.10
   conda activate rag-chatbot
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   Create a `.env` file in the project root and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

1. Prepare your Azure ML documentation:
   Place your PDF documentation in the `data` directory.

2. Build the knowledge base:

   ```
   python fill_db.py
   ```

3. Run the Streamlit app:

   ```
   streamlit run chatbot.py
   ```

4. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Project Structure

- `fill_db.py`: Script to process the PDF and create the vector database.
- `chatbot.py`: Main application file with the Streamlit interface and chatbot logic.
- `requirements.txt`: List of Python dependencies.
- `data/`: Directory to store Azure ML documentation PDFs.
- `chroma_db/`: Directory where the ChromaDB files are stored.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Anthropic for providing the Claude API
- Microsoft Azure for their comprehensive ML documentation

## Contact

[Mouaad AGOURRAM] - [email](mouaad.agourram@outlook.fr)

Project Link: [https://github.com/your-username/azure-ml-rag-chatbot](https://github.com/your-username/azure-ml-rag-chatbot)

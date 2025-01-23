import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from dotenv import load_dotenv 

# Updated Langchain imports
from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import

# Other imports
import google.generativeai as genai
import serpapi

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessorConfig:
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ('\n\n', '\n', '.', ',')

class RAGProcessor:
    def __init__(self, config: Optional[DocumentProcessorConfig] = None):
        """Initialize the RAG processor with configuration"""
        load_dotenv()
        self.config = config or DocumentProcessorConfig()
        self._setup_credentials()
        self.model = self._initialize_model()
        self.vector_store = None
        self.embeddings = self._initialize_embeddings()

    def _setup_credentials(self) -> None:
        """Setup API credentials"""
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        if not os.getenv("GEMINI_API_KEY"):
            raise EnvironmentError("GEMINI_API_KEY not found in environment")

    def _validate_environment(self) -> None:
        """Validate required environment variables"""
        required_vars = ["SERPAPI_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    def _initialize_model(self) -> Any:
        """Initialize AI model with fallback strategy"""
        try:
            return genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}. Falling back to OpenAI.")
            try:
                return OpenAI(temperature=0.7)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize both Gemini and OpenAI: {e}")

    def _initialize_embeddings(self) -> Any:
        """Initialize embeddings with fallback strategy"""
        try:
            logger.info("Attempting to initialize Gemini embeddings...")
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini embeddings: {e}")
            logger.info("Falling back to OpenAI embeddings...")
            if not os.getenv("OPENAI_API_KEY"):
                raise EnvironmentError("Neither Gemini nor OpenAI credentials available")
            return OpenAIEmbeddings()

    async def process_documents(self, 
                              urls: Optional[List[str]] = None, 
                              pdf_paths: Optional[List[str]] = None) -> bool:
        """Process documents from multiple sources"""
        docs = []
        
        if urls:
            docs.extend(await self._process_urls(urls))
        
        if pdf_paths:
            docs.extend(await self._process_pdfs(pdf_paths))
            
        if not docs:
            logger.warning("No documents were processed")
            return False
            
        return await self._create_vector_store(docs)

    async def _process_urls(self, urls: List[str]) -> List[Any]:
        """Process documents from URLs"""
        try:
            loader = UnstructuredURLLoader(urls=urls)
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading URLs: {e}")
            return []

    async def _process_pdfs(self, pdf_paths: List[str]) -> List[Any]:
        """Process PDF documents"""
        docs = []
        for path in pdf_paths:
            full_path = Path("./documents") / path  # Look in documents subdirectory
            if not full_path.exists():
                logger.warning(f"PDF path does not exist: {full_path}")
                continue
            try:
                loader = PyPDFLoader(str(full_path))
                docs.extend(loader.load_and_split())
            except Exception as e:
                logger.error(f"Error loading PDF {full_path}: {e}")
        return docs

    async def _create_vector_store(self, docs: List[Any]) -> bool:
        """Create vector store from processed documents"""
        if not docs:
            logger.error("No documents to process")
            return False

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators
            )
            splits = text_splitter.split_documents(docs)
            
            if not splits:
                logger.error("No text splits generated")
                return False

            logger.info(f"Creating vector store with {len(splits)} text splits")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            return True

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False

    async def ask_question(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Query the vector store and generate an answer"""
        if not self.vector_store:
            return {"error": "No documents have been processed yet"}

        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Generate answer using the model
            context = "\n".join(doc.page_content for doc in docs)
            prompt = f"""Based on the following context, answer the question.
            Context: {context}
            
            Question: {query}
            
            Answer:"""
            
            response = self.model.generate_content(prompt)
            
            result = {
                "answer": response.text,
                "sources": [doc.metadata for doc in docs],
                "confidence": 0.8  # Add actual confidence scoring
            }
            
            # If no good answer, fall back to web search
            if len(response.text) < 50 or "I don't know" in response.text.lower():
                web_results = await self._web_search(query)
                result["web_results"] = web_results
                
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {"error": str(e)}

    async def _web_search(self, query: str) -> List[Dict[str, str]]:
        """Fallback web search when no good answer is found"""
        try:
            search = serpapi.GoogleSearch({
                "q": query,
                "api_key": os.getenv("SERPAPI_API_KEY")
            })
            results = search.get_dict()
            
            return [{
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "link": result.get("link")
            } for result in results.get("organic_results", [])]
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []

async def main():
    try:
        processor = RAGProcessor()
        
        urls = []
        url = input("Enter URL of document (or press enter to skip): ")
        if url:
            urls.append(url)
        
        pdf_paths = []
        pdf_path = input("Enter PDF filename from documents folder (or press enter to skip): ")
        if pdf_path:
            pdf_paths.append(pdf_path)
        
        if not urls and not pdf_paths:
            print("No documents provided for processing")
            return
            
        success = await processor.process_documents(urls=urls, pdf_paths=pdf_paths)
        if not success:
            print("Failed to process documents")
            return
            
        while True:
            question = input("Ask a question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
                
            result = await processor.ask_question(question)
            print(f"\nAnswer: {result['answer']}")
            if 'error' not in result:
                print("\nSources:")
                for source in result['sources']:
                    print(f"- {source}")
            print("\n" + "="*50 + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
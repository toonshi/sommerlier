from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

INDEX_NAME = "studyrag"  # Define index name

# Initialize Pinecone client
pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))

def query_knowledge_base(query: str, use_gpt_knowledge: bool = True) -> str:
    """
    Query the knowledge base using the provided question
    
    Args:
        query (str): The user's question
        use_gpt_knowledge (bool): Whether to use GPT knowledge or stick to document context
    
    Returns:
        str: The response to the user's question
    """
    try:
        # Initialize OpenAI components
        embeddings = OpenAIEmbeddings()
        llm = ChatOpenAI(temperature=0.7)
        
        # Initialize Pinecone vector store with OpenAI embeddings
        vectorstore = Pinecone.from_existing_index(
            INDEX_NAME,
            embeddings
        )
        
        # Search for relevant documents in Pinecone
        docs = vectorstore.similarity_search(query, k=3)
        
        # If no documents are found, return a message
        if not docs:
            return "Sorry, I couldn't find any relevant information."

        # Prepare context from documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Prepare the prompt based on mode (using context and/or GPT knowledge)
        if use_gpt_knowledge:
            prompt = f"""Answer the following question using both the provided context and your general knowledge. 
            If you use information outside the context, please indicate this clearly.
            
            Context: {context}
            
            Question: {query}"""
        else:
            prompt = f"""Answer the following question using ONLY the information from the provided context. 
            If the context doesn't contain enough information to answer the question fully, please say so.
            
            Context: {context}
            
            Question: {query}"""
        
        # Get response from OpenAI
        response = llm.predict(prompt)
        
        return response
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

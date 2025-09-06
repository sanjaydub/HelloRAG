from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# wiki = WikipediaLoader()

def get_wiki_content(topic):
    wiki = WikipediaLoader(query=topic, load_max_docs=1)
    documents = wiki.lazy_load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

def create_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store 

def main(topic, query):
    # topic = "Amitabh Bachchan"
    texts = get_wiki_content(topic)
    vector_store = create_vector_store(texts)
    
    # Example query
    # query = "How many kids Amitabh has?"
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Create a prompt template
    prompt = PromptTemplate(
        template="Use the following pieces of context to answer the question at the end. "
                 "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
                 "{context}\n\n"
                 "Question: {question}\n"
                 "Answer:",
        input_variables=["context", "question"]
    )

    

    final_prompt = prompt.invoke({"context": context_text, "question": query})

    # Initialize ChatOpenAI
    # Get the response from the model
    
    chat = ChatOpenAI(temperature=0)
    response = chat.invoke(final_prompt)
    
    print(response.content) 

if __name__ == "__main__":
    # topic = "Web scraping"
    # texts = get_wiki_content(topic)
    # print(f"Retrieved {len(texts)} text chunks from Wikipedia on the topic '{topic}'.")
    # print(texts)
    main("Tata Consultancy Services", "who is the founder of TCS?")

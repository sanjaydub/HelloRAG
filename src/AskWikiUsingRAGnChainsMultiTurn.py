from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import AskWikiUsingRAG

load_dotenv()  # Load environment variables from .env file

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

context_topic = input("Please enter the Wikipedia topic you want to discuss : ")

print(f"===== Step 1: Fetching wiki content for topic: {context_topic} =====")
docs=AskWikiUsingRAG.get_wiki_content(context_topic)
print(f"===== Step 2: Fetched {len(docs)} documents from Wikipedia =====")
print("===== Step 3: Creating vector store =====")
vector_store=AskWikiUsingRAG.create_vector_store(docs)
print("===== Step 4: Creating retriever ===== ")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


retriever_chain = RunnableParallel(
    {"question": RunnablePassthrough(),
     "context": retriever | RunnableLambda(format_docs)}
)



# Create a prompt template
print("===== Step 5: Creating prompt template and augmenting with retriever ===== \n")
prompt = PromptTemplate(
    template="Use the following pieces of context to answer the question at the end. "
                "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
                "{context}\n\n"
                "Question: {question}\n"
                "Answer:",
    input_variables=["context", "question"]
)

parser = StrOutputParser()
llm = ChatOpenAI(temperature=0)
chain = retriever_chain | prompt | llm | parser
while True:
    # Get user input from the console
    input_user = input("user query: ")
    if input_user.strip().lower() == "exit":
        print("Exiting chat.")
        break
    response = chain.invoke(input_user)
    print("AI: " + response +"\n")



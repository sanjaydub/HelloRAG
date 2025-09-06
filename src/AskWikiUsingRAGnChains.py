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

docs=AskWikiUsingRAG.get_wiki_content("Infosys")
vector_store=AskWikiUsingRAG.create_vector_store(docs)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


retriever_chain = RunnableParallel(
    {"question": RunnablePassthrough(),
     "context": retriever | RunnableLambda(format_docs)}
)



# Create a prompt template
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
response=chain.invoke("Who is the CEO of Infosys?")
print(response)

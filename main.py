from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import  GooglePalm
load_dotenv()
from  langchain.document_loaders.csv_loader import CSVLoader
import os
from huggingface_hub import whoami
from huggingface_hub import login

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
#to remove halicunaiton....we use prompttemplates
from langchain.prompts import PromptTemplate



os.environ["HF_TOKEN"] = "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"
login(token=os.environ.get("HF_TOKEN"))
print(whoami())

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ['GOOGLE_API_KEY']

llm = GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'],temperature=0)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)







embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_db_file_path="faiss_index"
def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt', encoding='latin-1')
    data = loader.load()
    db = FAISS.from_documents(data, embeddings)
    db.save_local(vector_db_file_path)


if __name__ =="__main__":
    create_vector_db()



retriever = db.as_retriever()
rdocs = retriever.get_relevant_documents("How about job placement support?")
rdocs


promt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that 'don't know'
CONTEXT: {context} 

QUESTION: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(
    template=promt_template, input_variables=["context", "question"]
)



chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": PROMPT}
                                    )
chain("do you provide any java script course?")
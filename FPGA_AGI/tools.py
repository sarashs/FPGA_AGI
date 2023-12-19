import os
from FPGA_AGI.chains import DesignAndEvaluationChain
from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import BaseLLM

from langchain.chat_models import ChatOpenAI
from langchain.tools import HumanInputRun

from langchain.vectorstores import Chroma
#from langchain.docstore import InMemoryDocstore

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import chromadb
from langchain.tools.base import BaseTool
from langchain.chains import RetrievalQAWithSourcesChain
import warnings

# Define your embedding model
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
# Initialize the vectorstore as empty
#vectorstore = Chroma("langchain_store", embeddings)

### RAG tool

if os.path.isdir('knowledge_base'):
    persistent_client = chromadb.PersistentClient(path="./knowledge_base")
    pdfsearch = Chroma(client=persistent_client, embedding_function=embeddings, collection_name= "xilinx_manuals")
else:
    # Load documents
    loader = DirectoryLoader('.', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    pdfsearch = Chroma.from_documents(texts, embeddings, collection_name= "xilinx_manuals", persist_directory="./knowledge_base") 
retriever1 = pdfsearch.as_retriever(search_kwargs={"k": 3})
retriever1.search_type = "mmr"
search_chain = RetrievalQAWithSourcesChain.from_llm(llm, retriever=retriever1, return_source_documents=True,)
document_search_tool = Tool(
        name="Doc_Search",
        func=search_chain,
        description="useful for when you need to look up information in xilinx device manuals/datasheets, communications protocols, fpga textbooks, and digital design textbooks and other technical information."
        " The input to this tool is your key phrase that you are searching. You need to try a few different keyword combinations to get the best result."
    )
### Web search tool
search = SerpAPIWrapper()
web_search_tool = Tool(
        name = "Web_Search",
        func= search.run,
        description="useful for when you need to perform an internet search to answer questions or investigate online resources."
    )

### Human tool
def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)

human_input_tool = HumanInputRun(input_func=get_input)
#human_input_tool.description = "You can ask a human for guidance when you think you got stuck or you are not sure what to do next."
#" The input should be a specific question for the human."

### File writing tool

def save_file(input_dict: str):
    assert isinstance(input_dict, str), "input is not a string"
    colon_index = input_dict.find(">>>")
    # Extract file_name and file
    key = input_dict[:colon_index].strip()
    value = input_dict[colon_index + 1:].strip()
    # Trim whitespace
    file_name = key.strip()
    file = value.strip()
    directory = "solution"
    path = os.path.join(os.getcwd(), directory)
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = file_name
    file_path = os.path.join(path, file_name)
    try:
        with open(file_path, 'a+') as f:
            f.write(file)
    except:
        warnings.warn("The input format to the Save tool has an issue. Saving eveyrhing.")
        with open(os.path.join(path, 'design_tool_problematic.json'), 'a+') as f:
            f.write(file)
file_save_tool = Tool(name = "Save",
        func=save_file,
        description="useful for when you need to save a file. The input to this tool is a string with the following format: File_name_and_extension >>> content \n Example: \"output.json >>> {{key: value}}\" \n file extentions can include .v .cpp .sv .hdl .md"
    )   

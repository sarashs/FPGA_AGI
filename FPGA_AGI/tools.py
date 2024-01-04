import os
from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.tools import HumanInputRun

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
#from langchain.docstore import InMemoryDocstore

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import chromadb
from langchain.tools.base import BaseTool
from langchain.chains import RetrievalQAWithSourcesChain, LLMMathChain
import warnings
from langchain.utilities import PythonREPL

from FPGA_AGI.utils import extract_codes_from_string

# Define your embedding model
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0)
# Initialize the vectorstore as empty
#vectorstore = Chroma("langchain_store", embeddings)

### RAG tool

def load_sample_code_files(file_ending):
    documents = []
    directory = '.'
    for filename in os.listdir(directory):
        if filename.endswith(file_ending):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(Document(page_content=filename + "\n" + content, metadata={"source": filename}))
    return documents

if os.path.isdir('knowledge_base'):
    persistent_client = chromadb.PersistentClient(path="./knowledge_base")
    pdfsearch = Chroma(client=persistent_client, embedding_function=embeddings, collection_name= "knowledge_base")
else:
    # Load documents
    loader = DirectoryLoader('.', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    pdfsearch = Chroma.from_documents(texts, embeddings, collection_name= "knowledge_base", persist_directory="./knowledge_base")
    # load axi verilog samples 
    documents = load_sample_code_files('.v')
    pdfsearch.add_documents(documents)
    # System verilog samples from harris book
    documents = load_sample_code_files('.sv')
    pdfsearch.add_documents(documents)
retriever1 = pdfsearch.as_retriever(search_kwargs={"k": 3})
retriever1.search_type = "mmr"
search_chain = RetrievalQAWithSourcesChain.from_llm(llm, retriever=retriever1, return_source_documents=True,)
document_search_tool = Tool(
        name="Doc_Search",
        func=search_chain,
        description="useful for when you need to search for information, procedures, methods within documents."
        " The input to this tool is your key phrase that you are searching. You need to try a few different keyword combinations to get the best result."
    )

code_search_tool = Tool(
        name="Code_Search",
        func=search_chain,
        description="useful for when you need to look up sample codes."
        " The input to this tool is your key phrase that you are searching. You need to try a few different keyword combinations to get the best result."
    )

### Math tool
python_repl = PythonREPL()

def python_run(input_dict: str):
    # This is to take care of markdown formatting
    return python_repl.run(extract_codes_from_string(input_dict)) 

llm_math_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. You can use this tool for math computations of any kind in python. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_run,
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
def DUD(input_dict: str):
    return f'your thought is: {input_dict}'
think_again_tool = Tool(name = "Think",
                        func=DUD,
                        description="useful when you need to think more. This tool returns your own thought that you pass as input to the tool and cannot answer a question.")

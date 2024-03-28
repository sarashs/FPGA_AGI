import io
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from contextlib import redirect_stdout
from FPGA_AGI.utils import extract_codes_from_string

### SerpAPI websearch tool

search = SerpAPIWrapper()

@tool
def search_web(keywords: str):
    """A web search tool."""
    ret = search.run(keywords)
    return ret

### Python run tool

@tool
def python_run(input_dict: str):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`."""
    if 'print' in input_dict:
        pass
    else:
        return "Your code is not printing the results."
    global_scope = {"__builtins__": __builtins__}
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            exec(extract_codes_from_string(input_dict), global_scope)
        return output_buffer.getvalue()
    except Exception as e:
        return str(e)
    finally:
        output_buffer.close()

### Thought tool

@tool
def Thought(thought: str):
    """A thought happes via a function call to this function where the thought is passed to this function."""
    return f'Your thought is: {thought}'

### RAG tool



#human_input_tool.description = "You can ask a human for guidance when you think you got stuck or you are not sure what to do next."
#" The input should be a specific question for the human."


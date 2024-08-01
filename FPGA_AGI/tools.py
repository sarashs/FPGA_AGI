import io
import requests
from bs4 import BeautifulSoup
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools.tavily_search import TavilySearchResults
#from langchain_community.utilities import SerpAPIWrapper
from contextlib import redirect_stdout
from typing import Annotated
from FPGA_AGI.utils import extract_codes_from_string
from serpapi import GoogleSearch
import os
from FPGA_AGI.parameters import MAX_WEBSEARCH_RESULTS#, SERPAPI_PARAMS



### SerpAPI websearch tool

#search = SerpAPIWrapper()
"""
@tool
def search_web(keywords: Annotated[str, "The keywords you want to search on the web"]) -> list:
    SERPAPI_PARAMS["q"] = keywords
    search = GoogleSearch(SERPAPI_PARAMS)
    ret = search.get_dict()
    docs = []
    if not "organic_results" in ret.keys():
        print(SERPAPI_PARAMS)
        print(ret['error'])
        return docs
    for item in ret["organic_results"]:
        doc = {'content': None}
        # Fetch the webpage content
        if 'link' in item.keys():
            url = item["link"]
            try:
                response = requests.get(url)
            except:
                pass

            # Ensure the request was successful
            if response.status_code == 200:
                
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    page_text = ' '.join(soup.stripped_strings)
                    doc['content'] = page_text
                    docs.append(doc)
                except:
                    pass
            else:
                print(f"Failed to fetch the URL: {response.status_code}")
        if len(docs) == MAX_WEBSEARCH_RESULTS:
            return docs
    return docs
"""
search_web = TavilySearchResults()

### Python run tool

@tool
def python_run(code: Annotated[str, "The python code to execute to generate your results."]) -> str:
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`."""
    if 'print' in code:
        pass
    else:
        return "Your code is not printing the results."
    global_scope = {"__builtins__": __builtins__}
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer):
            exec(extract_codes_from_string(code), global_scope)
        return output_buffer.getvalue()
    except Exception as e:
        return str(e)
    finally:
        output_buffer.close()

### Thought tool

@tool
def Thought(thought: Annotated[str, "The thought that you came up with."]) -> str:
    """A thought happes via a function call to this function. You must pass your thought as an argument."""
    return f'Your thought is: {thought}'

### RAG tool



#human_input_tool.description = "You can ask a human for guidance when you think you got stuck or you are not sure what to do next."
#" The input should be a specific question for the human."


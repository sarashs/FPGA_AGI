#from typing import Dict, List, Optional, Any
#from pydantic import BaseModel, Field
from langchain import OpenAI, LLMChain, PromptTemplate
#from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.llms import BaseLLM
from FPGA_AGI.prompts import prompt_manager

class TestBenchCreationChain(LLMChain):
    """Chain to generate testbench."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = prompt_manager("TestBenchCreationChain").prompt
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=prompt_manager("TestBenchCreationChain").input_vars,
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
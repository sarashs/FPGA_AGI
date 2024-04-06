from langchain_core.runnables import Runnable
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_functions import create_structured_output_runnable
from typing import Dict, List, Optional, Any, Union
from FPGA_AGI.prompts import requirement_prompt, webextraction_cleaner_prompt

class Requirements(BaseModel):
    """Project requirements"""

    goals: List[str] = Field(
        description="List of goals based on the project's objectives"
    )

    requirements: List[str] = Field(
        description="List of requirements including all technical specifications and instructions provided"
    )

    lang: str = Field(
        description="HDL/HLS language to be used"
    )

#requirement_runnable = create_structured_output_runnable(
#    Requirements, bigllm, requirement_prompt
#)

class RequirementChain(Runnable):
    @classmethod
    def from_llm(cls, llm):
        requirement_runnable = create_structured_output_runnable(
            Requirements, llm, requirement_prompt
        )
        return requirement_runnable


class CleanedWeb(BaseModel):
    """Project requirements"""

    cleaned: str = Field(
        description="Extracted web page after clean-up"
    )

class WebsearchCleaner(Runnable):
    @classmethod
    def from_llm(cls, llm):
        requirement_runnable = create_structured_output_runnable(
            CleanedWeb, llm, webextraction_cleaner_prompt
        )
        return requirement_runnable

if __name__=='__main__':
    pass
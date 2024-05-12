from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, BaseChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing import Dict, List, Optional, Any, Union
from langchain.output_parsers import PydanticToolsParser
from FPGA_AGI.prompts import requirement_prompt, webextraction_cleaner_prompt

# requirement chain
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

# web cleaner chain
class CleanedWeb(BaseModel):
    """Extracted and cleaned web pages"""

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

# planner chain

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Planner(Runnable):
    @classmethod
    def from_llm_and_prompt(cls, llm, prompt):
        planner_runnable = create_structured_output_runnable(
            Plan, llm, prompt
        )
        return planner_runnable
    
# literature review chain
    
class Review(BaseModel):
    """Literature review structure"""
    #goals: List[str] = Field(description="list of design goals.")
    #requirements: List[str] = Field(description="list of design requirements.")
    #overview: str = Field(description="An overview of common specifications and architectures.") *   **Overview: A general overview of the report.
    methodology: str = Field(description="complete and comprehensive description of foundational theories, algorithms, existing solution and case studies, common technical challenges, effective strategies to mitigate any challenges.")
    implementation: str = Field(description="complete description of the choice of implementation technique including generic and hardware specific optimization techniques. This should be complete enough so that a hardware designer can design a solution based on it.")

class LiteratureReview(Runnable):
    @classmethod
    def from_llm_and_prompt(cls, llm, prompt):
        lit_rev_runnable = create_structured_output_runnable(
            Review, llm, prompt
        )
        return lit_rev_runnable
    
if __name__=='__main__':
    pass
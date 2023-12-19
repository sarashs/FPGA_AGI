import os
import re
from typing import Dict, List, Optional, Any, Union

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.prompts import StringPromptTemplate

from FPGA_AGI.tools import *

class RequirementAgentExecutor(AgentExecutor):
    @classmethod
    def from_llm_and_tools(cls, llm, tools, verbose=True):
        prefix = """You are an FPGA design engineer who will provide a comprehensive explanation of the key functions, goals, and specific requirements necessary for the following design project.

        1. Identifying essential features such as the deployment platform relevant to the design.
        2. Outlining interface requirements and communication protocols involved.
        3. Employ iterative 'Doc_search' for Comprehensive Understanding:
        - After reviewing the initial search results, refine your search terms to fill in any gaps or address specific project aspects.
        - Perform additional "Doc_search" iterations with these new terms.
        - Continue this iterative process of refining and searching until you have a complete understanding of the project requirements and constraints.

        Do not include any generic statements in your response. Make sure to extensively get human help.
        Before generating a final answer get human feedback. If human is not satisfied with the results, you will go through another iteration of generating the Goals, Requirements and constraints.

        Your final answer should be in the following format:
        - Goals
        ...
        - Requirements
        ...
        - constraints
        ...
        """
        suffix = """Question: {objective}
        {agent_scratchpad}"""
        prompt = ZeroShotAgent.create_prompt(
            tools, 
            prefix=prefix, 
            suffix=suffix, 
            input_variables=["objective", "agent_scratchpad"]
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        return cls.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose, handle_parsing_errors=True)

class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class ModuleAgentExecutor(AgentExecutor):
    @classmethod
    def from_llm_and_tools(cls, llm, tools, verbose=True):
        module_agent_template = """Generate a list of necessary modules and their descriptions for the FPGA based hardware design project.
        Include a top module which contains all of the other modules in it. Return the results in an itemized markdown format.

        You have access to the following tools:

        {tools}

        Use the following format:

        Goals: the main goal(s) of the design
        Requirements: design requirements
        Constraints: design constraints
        Thought: you should think about satisfying Goals and requirements
        Thought: you should further refine your thought
        Thought: you should think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Thought/Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question must be a JSON list of modules and descriptions with the following format for each module

        "Module_Name": {{
            "ports": ["specific inputs and outputs, including bit width"],
            "description": "detailed description of the module function",
            "connections": ["specific other modules it must connect to"],
            "hdl_language": "hdl language to be used",
        }}

        Goals: 
        {Goals}
        Requirements: 
        {Requirements}
        Constraints: 
        {Constraints}
        {agent_scratchpad}"""

        prompt = CustomPromptTemplate(
            template=module_agent_template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["Goals", "Requirements", "Constraints", "intermediate_steps"]
        )

        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        return cls.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
    
class HdlAgentExecutor(AgentExecutor):
    @classmethod
    def from_llm_and_tools(cls, llm, tools, verbose=True):
        hdl_agent_template = """You are an FPGA hardware engineer and you will code the module given after "Module". You will write fully functioning code not code examples or templates. 
        The final solution you prepare must compile into a synthesizable FPGA solution. It is of utmost important that you fully implement the module and do not leave anything to be coded later.

        Some guidelines:
        - DO NOT LEAVE THE PERFIPHERAL LOGIC TO THE USER AND FULLY DESIGN IT.
        - When using document search, you might have to use the tool multiple times and with various search terms in order to get better resutls.
        - Leave sufficient amount of comments in the code to enable further development and debugging.

        You have access to the following tools:

        {tools}

        Use the following format:

        Goals: the main goal(s) of the design
        Requirements: design requirements
        Constraints: design constraints
        Module list: list of modules you will build
        Module Codes: HDL/HLS code for the modules that you have already built
        Module: The module that you are currently building
        Thought: you should think about finding more about the coding style or example codes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: You write the HDL/HLS code. the final code of the module must be JSON with the following format. Do not return any comments or extra formatting.

        "Module_Name": {{
            "ports": ["specific inputs and outputs, including bit width"],
            "description": "detailed description of the module function",
            "connections": ["specific other modules it must connect to"],
            "hdl_language": "hdl language to be used",
            "code": "Synthesizable HLS/HDL code in line with the Goals, Requirements and Constraints. This code must be fully implemented and no aspect of it should be left to the user."
        }}

        Goals: 
        {Goals}
        Requirements: 
        {Requirements}
        Constraints: 
        {Constraints}
        Module list:
        {module_list}
        Module Codes:
        {codes}
        Module:
        {module}

        {agent_scratchpad}"""

        prompt = CustomPromptTemplate(
            template=hdl_agent_template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["Goals", "Requirements", "Constraints", "module_list", "codes", "module", "intermediate_steps"]
        )

        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        return cls.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
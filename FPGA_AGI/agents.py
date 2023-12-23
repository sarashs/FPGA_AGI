import re
from typing import Dict, List, Optional, Any, Union

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.prompts import StringPromptTemplate
import warnings

from FPGA_AGI.tools import *
from FPGA_AGI.utils import *
import FPGA_AGI.prompts as prompts

class RequirementAgentExecutor(AgentExecutor):
    @classmethod
    def from_llm_and_tools(cls, llm, tools, verbose=True):
        prefix = prompts.prefix2
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
        Final Answer: the final answer to the original input question must be a list of JSON dicts of modules and descriptions with the following format for each module

        {{  "Module_Name": "name of the module",
            "ports": ["specific inputs and outputs, including bit width"],
            "description": "detailed description of the module function",
            "connections": ["specific other modules it must connect to"],
            "hdl_language": "hdl language to be used"
        }}

        Do not return any extra comments, words or formatting.

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
        return cls.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose, handle_parsing_errors=True)
    
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

        {{  "Module_Name": "name of the module",
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
        return cls.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose, handle_parsing_errors=True)

class FPGA_AGI(BaseModel):
    verbose: bool = False
    solution_num: int = 0
    requirement_agent_executor: RequirementAgentExecutor = None
    module_agent_executor: ModuleAgentExecutor = None
    hdl_agent_executor: HdlAgentExecutor = None
    requirement: str = ''
    project_details: ProjectDetails = None
    module_list: List = []
    codes: List = []
    module_list_str: List = []

    def __init__(self, **data):
        super().__init__(**data)
        if (self.requirement_agent_executor is None) or (self.module_agent_executor is None) or (self.hdl_agent_executor is None):
            warnings.warn("RequirementAgentExecutor, ModuleAgentExecutor and HdlAgentExecutor are not set. Please use the from_llm class method to properly initialize it.", UserWarning)

    @classmethod
    def from_llm(cls, llm, verbose=False):
        tools = [
            web_search_tool,
            document_search_tool,
            #human_input_tool
        ]
        requirement_agent_executor = RequirementAgentExecutor.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        tools = [
            web_search_tool,
            document_search_tool,
        ]
        module_agent_executor = ModuleAgentExecutor.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        tools = [
            #web_search_tool,
            document_search_tool,
        ]
        hdl_agent_executor = HdlAgentExecutor.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        return cls(verbose=verbose, requirement_agent_executor=requirement_agent_executor, module_agent_executor=module_agent_executor, hdl_agent_executor=hdl_agent_executor)

    def _run(self, objective, action_type):
        if action_type == 'full':
            self.requirement = self.requirement_agent_executor.run(objective)
            self.project_details = extract_project_details(self.requirement)
        elif action_type == 'module':
            assert type(objective) == ProjectDetails, "objective needs to be of ProjectDetails type for module level design"
            self.project_details = objective
        if action_type == 'hdl':
            try:
                self.module_list = [json.loads(objective)]
                self.project_details = ProjectDetails(goals="design the specified module",requirements="no specific requirements",constraints="no specific constraints")
            except json.JSONDecodeError as e:
                raise FormatError
        else:
            self.module_list = self.module_agent_executor.run(Goals=self.project_details.goals, Requirements=self.project_details.requirements, Constraints=self.project_details.constraints)
            self.module_list = extract_json_from_string(self.module_list)
        self.codes = []
        self.module_list_str = [str(item) for item in self.module_list]
        #current_modules_verified = json.loads(current_modules_verified)
        for module in self.module_list:
            code = self.hdl_agent_executor.run(
                Goals=self.project_details.goals,
                Requirements=self.project_details.requirements,
                Constraints=self.project_details.constraints,
                module_list='\n'.join(self.module_list_str),
                codes='\n'.join(self.codes),
                module=str(module)
                )
            self.codes.append(code)
        save_solution(self.codes, solution_num=self.solution_num)
        self.project_details.save_to_file(solution_num=self.solution_num)
        self.solution_num += 1

    def __call__(self, objective: Any, action_type: str = 'full'):
        if (self.requirement_agent_executor is None) or (self.module_agent_executor is None) or (self.hdl_agent_executor is None):
            raise Exception("RequirementAgentExecutor, ModuleAgentExecutor and HdlAgentExecutor are not set. Please use the from_llm class method to properly initialize it.")
        if not (action_type in ['full', 'module', 'hdl']):
            raise ValueError("Invalid action type specified.")
        self._run(objective, action_type)
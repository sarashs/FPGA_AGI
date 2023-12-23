import re
from typing import Dict, List, Optional, Any, Union
from time import sleep

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.prompts import StringPromptTemplate
import warnings
import shutil

from FPGA_AGI.tools import *
from FPGA_AGI.utils import *
from FPGA_AGI.prompts import *
from FPGA_AGI.chains import TestBenchCreationChain

class RequirementAgentExecutor(AgentExecutor):
    @classmethod
    def from_llm_and_tools(cls, llm, tools, verbose=True):
        prefix = prompt_manager("RequirementAgentExecutor").prompt
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
        module_agent_template = prompt_manager("ModuleAgentExecutor").prompt

        prompt = CustomPromptTemplate(
            template=module_agent_template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=prompt_manager("ModuleAgentExecutor").input_vars
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
        hdl_agent_template = prompt_manager("HdlAgentExecutor").prompt

        prompt = CustomPromptTemplate(
            template=hdl_agent_template,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=prompt_manager("HdlAgentExecutor").input_vars
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
    project_details: ProjectDetails = None
    test_bench_creator: TestBenchCreationChain = None
    module_list: List = []
    codes: List = []
    test_benches: List = []
    module_list_str: List = []

    def __init__(self, **data):
        super().__init__(**data)
        if (self.requirement_agent_executor is None) or (self.module_agent_executor is None) or (self.hdl_agent_executor is None):
            warnings.warn("RequirementAgentExecutor, ModuleAgentExecutor and HdlAgentExecutor are not set. Please use the from_llm class method to properly initialize it.", UserWarning)
        dir_path = f'./solution_{self.solution_num}/'
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    @classmethod
    def from_llm(cls, llm, verbose=False):
        tools = [
            #web_search_tool,
            think_again_tool,
            document_search_tool,
            #human_input_tool
        ]
        requirement_agent_executor = RequirementAgentExecutor.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        tools = [
            #web_search_tool,
            think_again_tool,
            document_search_tool,
        ]
        module_agent_executor = ModuleAgentExecutor.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        tools = [
            #web_search_tool,
            document_search_tool,
        ]
        hdl_agent_executor = HdlAgentExecutor.from_llm_and_tools(llm=llm, tools=tools, verbose=verbose)
        test_bench_creator = TestBenchCreationChain.from_llm(llm=llm, verbose=verbose)
        return cls(verbose=verbose, requirement_agent_executor=requirement_agent_executor, module_agent_executor=module_agent_executor, hdl_agent_executor=hdl_agent_executor, test_bench_creator=test_bench_creator)

    def _run(self, objective, action_type):
        if action_type == 'full':
            requirement = self.requirement_agent_executor.run(objective)
            self.project_details = extract_project_details(requirement)
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
        self.test_benches = []
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
            sleep(10)
            tb = self.test_bench_creator.run(
                Goals=self.project_details.goals,
                Requirements=self.project_details.requirements,
                Constraints=self.project_details.constraints,
                module_list='\n'.join(self.module_list_str),
                module=code                
            )
            self.test_benches.append(tb)
            self.codes.append(code)
        save_solution(self.codes, solution_num=self.solution_num)
        save_solution(self.test_benches, solution_num=self.solution_num)
        self.project_details.save_to_file(solution_num=self.solution_num)
        self.solution_num += 1
        dir_path = f'./solution_{self.solution_num}/'
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    def __call__(self, objective: Any, action_type: str = 'full'):
        if (self.requirement_agent_executor is None) or (self.module_agent_executor is None) or (self.hdl_agent_executor is None):
            raise Exception("RequirementAgentExecutor, ModuleAgentExecutor and HdlAgentExecutor are not set. Please use the from_llm class method to properly initialize it.")
        if not (action_type in ['full', 'module', 'hdl']):
            raise ValueError("Invalid action type specified.")
        self._run(objective, action_type)
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class PromptVersion:
    prompt: str
    input_vars: List[str]

@dataclass
class PromptManager:
    prompts: Dict[str, List[PromptVersion]] = field(default_factory=dict)

    def add_prompt(self, prompt_name: str, prompt_text: str, input_vars: List[str]):
        if prompt_name not in self.prompts:
            self.prompts[prompt_name] = []
        self.prompts[prompt_name].append(PromptVersion(prompt_text, input_vars))

    def __call__(self, prompt_name: str, version: int = None) -> PromptVersion:
        if prompt_name not in self.prompts:
            raise ValueError(f"No prompt found with the name '{prompt_name}'.")
        
        if version is not None:
            if version < 1 or version > len(self.prompts[prompt_name]):
                raise ValueError(f"Invalid version for prompt '{prompt_name}'. Valid versions are 1 to {len(self.prompts[prompt_name])}.")
            return self.prompts[prompt_name][version - 1]
        else:
            return self.prompts[prompt_name][-1]

prompt_manager = PromptManager()
########
# Agents
########
# Requirement Agent
prompt_manager.add_prompt("RequirementAgentExecutor", """You are an FPGA design engineer who will provide a comprehensive explanation of the key functions, goals, and specific requirements necessary for the following design project.
                        1. Identifying essential features such as the deployment platform relevant to the design.
                        2. Outlining interface requirements and communication protocols involved.
                        3. Employ iterative 'Doc_search' for Comprehensive Understanding:
                        - After reviewing the initial search results, refine your search terms to fill in any gaps or address specific project aspects.
                        - Perform additional "Doc_search" iterations with these new terms.
                        - Continue this iterative process of refining and searching until you have a complete understanding of the project requirements and constraints.
                        4. Your job is not to write any code. Your focus is on Goals, Requirements, constraints
                        5. The requirements must include the implementation hdl/hls language.
                        6. All of the technical information provided to you via the input must be included within the Goals, Requirements and constraints.
                        Do not include any generic statements in your response. Make sure to extensively get human help.
                        Before generating a final answer get human feedback. If human is not satisfied with the results, you will go through another iteration of generating the Goals, Requirements and constraints.
                        Your final answer should be in the following format:
                        - Goals
                        ...
                        - Requirements
                        ...
                        - constraints
                        ...
                        """, [])

prompt_manager.add_prompt("RequirementAgentExecutor", """You are an FPGA design engineer tasked with providing a comprehensive explanation of the key functions, goals, and specific requirements for a design project. Your responsibilities include:

                        1.  **Broad Overview**: Provide a high-level overview of the project, encompassing essential aspects like the deployment platform and key functionalities.
                        2.  **Technical Details**: Ensure that all technical information provided, including any specific features, instructions, or protocols, is thoroughly incorporated into the project's goals, requirements, and constraints.
                        3.  **Focus on Planning**: Concentrate on defining the project's objectives and specifications. You are not responsible for writing any code, but rather for outlining the necessary steps and considerations for successful implementation.
                        4.  **Requirements Specification**: Detail the requirements, including any relevant hardware description languages or high-level synthesis languages, that will be utilized for the FPGA design.
                        5.  **Comprehensive Coverage**: Avoid generic statements and ensure that your response comprehensively covers the technical aspects provided in the input.
                        6.  **Self-Contained Response**: Ensure that your response is complete and standalone, suitable for input into another module without the need for external references. If you are refering to something you must include it in the output.

                        Your response should be structured as follows:

                        *   **Goals**: \[List of goals based on the project's objectives\]
                        *   **Requirements**: \[Detailed requirements, including all technical specifications and instructions provided\]
                        *   **Constraints**: \[List of constraints, considering the project's limitations and challenges\]""", [])

# Module Generation Agent
prompt_manager.add_prompt("ModuleAgentExecutor", """Generate a list of necessary modules and their descriptions for the FPGA based hardware design project.
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
                        {agent_scratchpad}""", ["Goals", "Requirements", "Constraints", "intermediate_steps"])

# HDL Generation Agent 
prompt_manager.add_prompt("HdlAgentExecutor", """You are an FPGA hardware engineer and you will code the module given after "Module". You will write fully functioning code not code examples or templates. 
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

                        {agent_scratchpad}""", ["Goals", "Requirements", "Constraints", "module_list", "codes", "module", "intermediate_steps"])

prompt_manager.add_prompt("HdlAgentExecutor_v2", """You are an FPGA hardware engineer and you will code the module given after "Module". You will write fully functioning code not code examples or templates. 
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
                        Thought: you MUST always look for code examples and potential readily available solutions through an Action
                        Action: the action to take, should be one of [{tool_names}]
                        Action Input: the input to the action
                        Observation: the result of the action
                        ... (this Thought/Action/Action Input/Observation can repeat N times)
                        Thought: I now know the final answer
                        Final Answer: You write the HDL/HLS code. Synthesizable HLS/HDL code in line with the Goals, Requirements and Constraints. This code must be fully implemented and no aspect of it should be left to the user. Do not return any comments or extra formatting.

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

                        {agent_scratchpad}""", ["Goals", "Requirements", "Constraints", "module_list", "codes", "module", "intermediate_steps"])
########
# Chains
########
# Test bench generation chain
prompt_manager.add_prompt("TestBenchCreationChain", """You are an FPGA hardware engineer and you will write a test-bench the module given after "Module". 

                        Some guidelines:
                        - Your test benches are written in the language that is used for coding the actual code.
                        - Your test bench must cover the function of the module you are testing
                        - Your testbench module name must be "original_module-tb". basically adding "-tb" to the original module's name

                        You are provided with the following:

                        Goals: the main goal(s) of the design
                        Requirements: design requirements
                        Constraints: design constraints
                        Module list: list of modules you will write test-benches for
                        Module: The module that you are currently building along with its code

                        Final Answer: You write the HDL/HLS code. the final code of the module must be JSON with the following format. Do not return any comments or extra formatting.

                        {{  "Module_Name": "test-bench module name",
                            "hdl_language": "hdl language to be used",
                            "code": "Test benchcode in line with the module code. All corner cases must be covered."
                        }}

                        Goals: 
                        {Goals}
                        Requirements: 
                        {Requirements}
                        Constraints: 
                        {Constraints}
                        Module list:
                        {module_list}
                        Module:
                        {module}""", ["Goals", "Requirements", "Constraints", "module_list", "module"])
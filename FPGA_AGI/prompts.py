from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, FunctionMessage

requirement_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are very senior FPGA design engineer.
            Your goal is to expand the input objective into a requirement document for a design. You should also take into account the context provided.
            The design document consists of three items: "Goals", "Requirements", "HDL/HLS language".
            Keep the following in mind:
            1.  **Goal extraction**: Extracting the main goals based on the objective
            2.  **Technical Details**: Ensure that all technical information provided, including any specific features, instructions, or protocols, is thoroughly incorporated into the project's goals, requirements, and constraints.
            3.  **Focus on Planning**: Concentrate on defining the project's objectives and specifications. You are not responsible for writing any code, but rather for outlining the necessary steps and considerations for successful implementation.
            4.  **Focus on HDL/HLS design**: Come up with a step by step process for hardware description languages or high-level synthesis project. You are not responsible for anything other than logic design.
            5.  **Avoid generic specifications**: Avoid generic statements and ensure that your response comprehensively covers the technical aspects provided in the input or gathered through searches.
            6.  **Avoid non-functional specifications**: Do not include anything that is not related to HDL/HLS coding of the module. In other words, avoid time-lines, reliaility, scalability and such
            7.  **Self-Contained Response**: Ensure that your response is complete and standalone, suitable for input into another module without the need for external references. If you are refering to something you must include it in the output.
""",
        ),
        ("user", "Objective:\n {objective} \n\n Context: \n {context}"),
    ]
)

webextraction_cleaner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Your task is to clean the output of a web extractor given after "Extraction". \
            That is to remove the nonsensical strings as well as excessive line breaks and etc. \
            Your job is not to summarize the content.\
            You must retain all of the equations, formulas, algorithms etc. Do not summarize.""",
        ),
        ("user", "Extraction:\n {extraction}"),
    ]
)
#####

hierarchical_agent_prompt = ChatPromptTemplate.from_messages(
    [SystemMessage(content="""You are an FPGA design engineer whose purpose is to design the architecture graph of a HDL hardware project. Your design will be used by a HDL/HLS coder to write the modules.
            The HDL/HSL coder agent only knows how to code. It cannot perform any computations or do any independent designs. It needs all the technical information necessary to code the modules. \
            - You will receive guidelines from the researcher agent on what you need to design.
            - You are deciding on the module names, description, ports, what modules each module is connected to.
            - You also include anything that may be necessary in order for the downstream HDL/HSL coder in the description. This includes any search results or computations performed by other agents that are used for coding a particular module. This is a very important requirement.
            - If necessary, you can expand your knowledge on the subject matter via the search tool before committing to a response.
            - You are not responsible for designing any test benches.
            - If you are defining a top module or any other hierarchy, you must mention that in the module description.
            - If multiple instances of the same module are needed for your design then you include multiple instances of that module and subscribt the name either with numbers or letters.
            - If you have performed or otherwise have access to any computation that is needed or can help with coding a particular module, include that within the module notes.
            - Do not forget that your actions take place via a function call,

            Use the following format:

            Thought: You should think of an action. You do this by calling the Though tool/function. This is the only way to think.
            Action: You take an action through calling the search_web tool.
            ... (this Thought/Action can repeat 3 times)
            Response: You should use the HierarchicalResponse tool to format your response. Do not return your final response without using the HierarchicalResponse tool"""),
            MessagesPlaceholder(variable_name="messages"),
]
)

module_design_agent_prompt = ChatPromptTemplate.from_messages(
    [SystemMessage(content="""You are an FPGA design engineer whose purpose is to code the modules along with test cases for an HDL/HLS hardware project.
            You are being called in an iterative fashion and at each stage you are provided with the whole design architecture in json format as well as the module you will be designing at the moment.
            - You are responsible for writitng complete synthasizable code.
            - You are not allowed to leave any placeholders in the code. You must write complete synthesizable code.
            - You are not going to ask the user to write the code but rather you gather information and write codes on your own.
            - You can use the tools provided to you if you need to search or compute anything.
            - You must think or take actions via a function calls,

            Use the following format:

            Thought: You should think of an action. You do this by calling the Thought tool/function. This is the only way to think.
            Action: You take an action through calling one of the search_web or python_run tools.
            ... (this Thought/Action can repeat 3 times)
            Response: You Must use the CodeModuleResponse tool to format your response. Do not return your final response without using the CodeModuleResponse tool"""),
            MessagesPlaceholder(variable_name="messages"),
]
)

# The following prompt prepares message for the hirearchical design agent based only on goals and requirements
hierarchical_agent_prompt_human = HumanMessagePromptTemplate.from_template("""Design the architecture graph for the following goals and requirements. 
        Goals:
        {goals}
        
        Requirements:
        {requirements}
        """)
# The following prompt prepares message for the module design agent
module_agent_prompt_human = HumanMessagePromptTemplate.from_template(
    """Write the HLS/HDL code in verilog for the following desgin. Note that the design consists of modules with\
    input/output and connecting modules already designed for you. Your task is to build the modules consistently with the modules that you have already build and with the overal desing.\
    note also that the note section of each module provides you with necessary information, guidelines and other helpful elements to perform your design.
    Remember to write complete synthesizable module code without placeholders.

    Hierarchical design:
    {hierarchical_design}
                                                                 
    Modules built so far:
    {modules_built}
    
    Current Module:
    {current_module}

    you must always use the CodeModuleResponse tool for your final response.
    """
    )
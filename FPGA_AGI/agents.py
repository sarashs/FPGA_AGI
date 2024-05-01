try:
    from FPGA_AGI.tools import search_web, python_run, Thought
    from FPGA_AGI.prompts import hierarchical_agent_prompt, module_design_agent_prompt, hierarchical_agent_evaluator, hierarchical_agent_update_prompt, module_evaluate_agent_prompt
    from FPGA_AGI.parameters import RECURSION_LIMIT
    from FPGA_AGI.chains import WebsearchCleaner, Planner, LiteratureReview
    import FPGA_AGI.utils as utils
except ModuleNotFoundError:
    from prompts import hierarchical_agent_prompt, module_design_agent_prompt, hierarchical_agent_evaluator, hierarchical_agent_update_prompt, module_evaluate_agent_prompt
    from parameters import RECURSION_LIMIT
    from chains import WebsearchCleaner, Planner, LiteratureReview
    from tools import search_web, python_run, Thought
    import utils
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, BaseChatPromptTemplate
import json
from langgraph.prebuilt import ToolExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool, format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
import operator
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, FunctionMessage
#from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate, BaseChatPromptTemplate
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
import os

class Module(BaseModel):
    """module definition"""
    name: str = Field(description="Name of the module.")
    description: str = Field(description="Module description including detailed explanation of what the module does and how to achieve it. Think of it as a code equivalent of the module without coding it.")
    connections: List[str] = Field(description="List of the modules connecting to this module (be it via input or output).")
    ports: List[str] = Field(description="List of input output ports inlcuding clocks, reset etc.")
    module_template: str = Field(description="Outline of the HDL/HLS code with placeholders and enough comments to be completed by a coder. The outline must include module names and ports. The placeholders must be comments starting with PLACEHOLDER:")

class HierarchicalResponse(BaseModel):
    """Final response to the user"""
    graph: List[Module] = Field(
        description="""List of modules"""
        )

### System level evaluator
class SystemEvaluator(BaseModel):
    """system design evaluation"""
    coding_language: bool = Field(description="Whether the coding language is incorrect.")
    connections: bool = Field(description="Whether the connections between modules are inconsistent or the input/outputs are connected improperly.")
    ports: bool = Field(description="Whether the ports and interfaces are defined incorrectly or there are any missing ports")
    excessive: bool = Field(description="Whether the design has any excessive and/or superflous modules.")
    missing: bool = Field(description="Whether the design is missing any modules.")
    template: bool = Field(description="Whether the template code correctly identifies all of the place holders and correctly includes the module ports.")
    overal: bool = Field(description="Whether the current system is not going to be able to satisfy all of the design goals and most of its requirements.")
    description: str = Field(description="If any of the above is true, describe which module it is and how it should be corrected, otherwise NA.")

### Module level evaluator
class ModuleEvaluator(BaseModel):
    """module evaluation"""
    coding_language: bool = Field(description="Whether the coding language is incorrect.")
    adherence: bool = Field(description="Whether the code adheres to HDL/HLS language components. In particular, if the language is HLS C++, the code must be in HLS C++ and not just regular C++.")
    placeholders: bool = Field(description="Whether the code has any placeholders or otherwise missing components from a complete synthesizable code in this module.")
    optimizations: bool = Field(description="Whether the code is optimized in line with the goals and requirements. For HLS this is achieved via pragmas")
    feedback: str = Field(description="If any of the above is true, describe what is needed to be done, otherwise NA.")

### Module Design agent
class CodeModuleResponse(BaseModel):
    """Final response to the user"""
    name: str = Field(description="Name of the module.")
    description: str = Field(description="Brief module description.")
    connections: List[str] = Field(description="List of the modules connecting to this module.")
    ports: List[str] = Field(description="List of input output ports inlcuding clocks, reset etc.")
    module_code: str = Field(description="Complete working synthesizable HDL/HLS code without any placeholders.")
    test_bench_code: str = Field(description="Complete behavioral test for the module must be written in the same HDL/HLS language. the testbench module name should be module name underline tb")

class FinalDesignGraph(BaseModel):
    """Final Design Graph"""
    graph: List[CodeModuleResponse] = Field(
        description="""List of modules"""
        )

class GenericAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class GenericToolCallingAgent(object):
    """This agent is a generic tool calling agent. The list of tools, prompt template and resopnse format decides what the agent actually does."""
    def __init__(self, model: ChatOpenAI,
                 prompt: BaseChatPromptTemplate=hierarchical_agent_prompt,
                 tools: List=[], response_format: BaseModel=HierarchicalResponse):
        tools.append(Thought)
        self.tool_executor = ToolExecutor(tools)
        self.response_format = response_format
        functions = [convert_to_openai_function(t) for t in tools]
        functions.append(convert_to_openai_function(response_format))
        self.model = prompt | model.bind_functions(functions)
        # Define a new graph
        self.workflow = StateGraph(GenericAgentState)

        # Define the two nodes we will cycle between
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("action", self.call_tool)

        self.workflow.set_entry_point("agent")

        # We now add a conditional edge
        self.workflow.add_conditional_edges(
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            {
                # If `tools`, then we call the tool node.
                "continue": "action",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        self.workflow.add_edge("action", "agent")

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        self.app = self.workflow.compile()

    def should_continue(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if "function_call" not in last_message.additional_kwargs:
            print("---Error---")
            print(last_message)
            print("The message must be a function call")
            return "end"
        # Otherwise if there is, we need to check what type of function call it is
        elif last_message.additional_kwargs["function_call"]["name"] == self.response_format.__name__:
            return "end"
        # Otherwise we continue
        else:
            return "continue"

    # Define the function that calls the model
    def call_model(self, state):
        #messages = state["messages"]
        response = self.model.invoke(state)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the function to execute tools
    def call_tool(self, state):
        messages = state["messages"]
        # Based on the continue condition
        # we know the last message involves a function call
        last_message = messages[-1]
        # We construct an ToolInvocation from the function_call
        print(last_message.additional_kwargs["function_call"]["name"])
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            ),
        )
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)
        # We return a list, because this will get added to the existing list
        #if last_message.additional_kwargs["function_call"]["name"] != 'Thought':
        return {"messages": [function_message]}
        #else:
        #    return

    def invoke(self, messages):

        output = self.app.invoke(messages, {"recursion_limit": RECURSION_LIMIT})
        print(output)
        out = json.loads(output['messages'][-1].additional_kwargs["function_call"]["arguments"])
        out = self.response_format.parse_obj(out)
        return out

    def stream(self, messages):
        
        for output in self.app.stream(messages, {"recursion_limit": RECURSION_LIMIT}):
        # stream() yields dictionaries with output keyed by node name
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
                print("\n---\n")  
        out = json.loads(output['__end__']['messages'][-1].additional_kwargs["function_call"]["arguments"])
        out = self.response_format.parse_obj(out)
        return out   
### 

class ResearcherResponse(BaseModel):
    """Final response to the user"""
    web_results: str = Field(description="""Useful results from the web""")
    document_results: str = Field(description="""Useful results from the document database""")
    code_output: str = Field(description="""Any code execution results that may be useful for the design""")
    solution_approach: str = Field(description="""Description of the solution approach""")

class EngineerAgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
        messages: The commumications between the agents
        sender: The agent who is sending the message
    """
    keys: Dict[str, any]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class Engineer(object):
    """This agent performs research on the design."""

    def __init__(self, model: ChatOpenAI, retriever: Any, language: str = 'verilog', solution_num: int = 0):

        self.solution_num = solution_num
        self.retriever = retriever
        self.model = model
        self.webcleaner = WebsearchCleaner.from_llm(llm=model)
        self.language = language # or "vhdl", "systemverilog", "cpp"
        self.input_context = None
        self.requirements = None
        self.goals = None
        self.lit_search_results_ = []
        self.lit_review_results = None
        self.hierarchical_solution_result = None

        #### lit review questions Agent

        # prompt **For each of the following general topics, break them down into at least 3 sub-topics and generate queries for them:**
        planner_prompt = ChatPromptTemplate.from_messages(
            [(
                    "system",
                    """
**Objective:** You are programmed as a hardware engineering literature review agent. Your purpose is to autonomously generate a step-by-step list of web search queries that will aid in gathering both comprehensive and relevant information about hardware design, adaptable across various FPGA platforms without being overly specific to any single device.

**Follow these instructions when generating the queries:**

*   **Focus on Practicality and Broad Applicability:** Ensure each search query is practical and likely to result in useful findings. Avoid queries that are too narrow or device-specific which may not yield significant search results.
*   **Sequential and Thematic Structure:** Organize questions to start from broader concepts and architectures, gradually narrowing down to specific challenges and solutions relevant to a wide range of FPGA platforms.
*   **Contextually Rich and Insightful Inquiries:** Avoid overly broad or vague topics that do not facilitate actionable insights. The list of questions should involve individual tasks, that if searched on the will yield specific results. Do not add any superfluous questions.
*   **Use of Technical Terminology with Caution:** While technical terms should be used to enhance query relevance, ensure they are not used to create highly specific questions that are unlikely to be answered by available literature.
*   **Clear and Structured Format:** Queries should be clear and direct, with each starting with "search:" followed by a practical, result-oriented question. End with a "report:" task to synthesize findings into a comprehensive literature review.

**Only perform one query for each topic:**

1.  **General Overview:** Start with an overview of common specifications and architectures, related to the project goal. avoiding overly specific details related to any single model or board.
2.  **Existing Solutions and Case Studies:** Investigate a range of implementations and case studies focusing on digital signal processing tasks like FFT on FPGAs, ensuring a variety of platforms are considered.
3.  **Foundational Theories:** Delve into the theories and methodologies underpinning FPGA applications.
4.  **Common Technical Challenges:** Identify and explore common technical challenges associated with FPGA implementations, discussing broadly applicable solutions.
5.  **Optimization and Implementation Techniques:**Identify effective strategies and techniques for optimizing FPGA designs, applicable across different types of hardware.
6.  **Hardware specific Optimization :** Conclude with effective strategies and techniques for optimizing FPGA designs for the specific hardware (if a specific platform is provided to you).

**Final Task:**

*   **report:** Synthesize all information into a structured and comprehensive literature review that is informative and applicable to hardware designers working with various FPGA platforms.

Example of a Specific Query Formation:

search: "pipelining of risc-v devices."
"""
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # This is the input prompt for when the object is invoked or streamed
        self.planner_agent_prompt_human = HumanMessagePromptTemplate.from_template("""Design the literature review set of questions for the following goals and requirements. Be considerate of the user input context.
                    goals:
                    {goals}
                    requirements:
                    {requirements}
                    user input context:
                    {input_context}""")

        self.planner_chain = Planner.from_llm_and_prompt(llm=model, prompt=planner_prompt)

        #### literature review Agent

        # prompt
        lit_review_prompt = ChatPromptTemplate.from_messages(
            [(
                    "system",
                    """
**Objective:** You are a hardware engineering literature review agent. Your purpose is to generate a literature review based on a set of goals, requirements, user input context and queries and results provided to you. 
You must write this literature review document as thoroughly as possible.

**Follow these instructions when generating the report:**

*   **Methodology: Completely describe any methods, algorithms and theoretical background of what will be implemented. Just naming or mentioning the method is not sufficient. You need to explain them to the best of your ability. This section is often more than 500 words.** 
*   **implementation: For this section, you will write about an implementation strategy. You must write detailed description of your chosen implementation strategy and why it is better than other strategies and more aligned with the goals/requirements. Try to base this section on the search results if possibe. This section is often more than 500 words.**
*   Do not write meaningless or vague sections but rather write a to the point complete technical document. Do not write superfluous statemetns.
*   Do not write anything about documentation and testing or anything outside of what is needed for a design engineer to write HDL/HLS code.
"""
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # This is the input prompt for when the object is invoked or streamed
        self.lit_review_agent_prompt_human = HumanMessagePromptTemplate.from_template("""Prepare the literature review for the following list of queries and results given goals, requirements and input context given by the user.
goals:
{goals}
requirements:
{requirements}
user input context:
{input_context}
Queries and results:
{queries_and_results}""")

        self.lit_review_chain = LiteratureReview.from_llm_and_prompt(llm=model, prompt=lit_review_prompt)

    #### Research Agent
        class decision(BaseModel):
            """Decision regarding the next step of the process"""

            decision: str = Field(description="Decision 'search' or 'compute' or 'solution'", default='search')
            search: str = Field(description="If decision search, query to be searched else, NA", default='NA')
            compute: str = Field(description="If decision compute, description of what needs to be computed else, NA", default='NA')
        
        # Tool
        decision_tool_oai = convert_to_openai_tool(decision)
        # LLM with tool and enforce invocation
        research_with_tool = self.model.bind(
            tools=[decision_tool_oai],
            tool_choice={"type": "function", "function": {"name": "decision"}},
        )
        
        # Parser
        research_parser_tool = PydanticToolsParser(tools=[decision])

        # prompt
        research_prompt = ChatPromptTemplate.from_messages(
            [(
                    "system","""
                    You are a hardware enigneer whose job is to collect all of the information needed to create a new solution.\
                    You are collaborating with some assistants. In particular you are following the plans generated by the planner assistant.\
                    Following the plans provided by the planner and the current step of the plan you come up with what needs to be searched or computed and then making a decision to use 
                    the search assistant, the compute assisstant or the final solution excerpt generator.\
                    You ask your questions or perform your computations with the help of the assisstant agents one at a time. You generate an all inclusive search or compute quary based on the current stage of the plan.\
                    The Plan might get updated throughout the operation by the planner.\
                    You have access to the following assisstants: \
                    search assisstant: This assisstant is going to perform document and internet searches.\
                    compute assisstant: This assisstant is going to generate a python code based on your request, run it and share the results with you.\
                    solution assisstant: This assisstant is going to generate the final excerpt based on the interaction you had with the other two agents."""
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        # Chain
        self.research_agent = (research_prompt
            | research_with_tool
            | research_parser_tool
        )

    #### Document Grading Agent
        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # Tool
        grade_tool_oai = convert_to_openai_tool(grade)

        # LLM with tool and enforce invocation
        document_grading_llm_with_tool = self.model.bind(
            tools=[grade_tool_oai],
            tool_choice={"type": "function", "function": {"name": "grade"}},
        )

        # Parser
        document_grading_parser_tool = PydanticToolsParser(tools=[grade])

        # Prompt
        document_grading_prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            """,
            input_variables=["context", "question"],
        )

        # Chain
        self.document_grading_agent = document_grading_prompt | document_grading_llm_with_tool | document_grading_parser_tool

    #### Compute Agent

        compute_agent_system_prompt = """You are a hardware engineer helping a senior hardware engineer with their computational needs. In order to do that you use python.
        Work autonomously according to your specialty, using the tools available to you.
        You must write a code to compute what is asked of you in python to answer the question.
        Do not answer based on your own knowledge/memory. instead write a python code that computes the answer, run it, and observe the results.
        You must print the results in your response no matter how long or large they might be. 
        You must completely print the results. Do not leave any place holders. Do not use ... 
        Do not ask for clarification. You also do not refuse to answer the question.
        retrun your response as some explaination of what the response is about plus the actual response.
        Your other team members (and other teams) will collaborate with you with their own specialties."""
        compute_agent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    compute_agent_system_prompt,
                ),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent_with_tool = create_openai_tools_agent(self.model, [python_run], compute_agent_prompt)
        self.compute_agent = AgentExecutor(agent=agent_with_tool, tools=[python_run], verbose=True)
        

    #### Hierarchical Design Agent
        self.hierarchical_design_agent = GenericToolCallingAgent(model=model, tools=[search_web])

    #### Hierarchical Design Evaluator Agent
        self.hierarchical_design_evaluator_agent = GenericToolCallingAgent(
            response_format=SystemEvaluator,
            prompt=hierarchical_agent_evaluator,
            model=model,
            tools=[]
            )

    #### Hierarchical Redesign Agent
        self.hierarchical_redesign_agent = GenericToolCallingAgent(
            prompt=hierarchical_agent_update_prompt,
            model=model,
            tools=[]
            )

    #### Module Design Agent
        self.module_design_agent = GenericToolCallingAgent(
            model=model, prompt=module_design_agent_prompt,
            tools=[search_web, python_run], response_format=CodeModuleResponse
            )

    #### Module Evaluator agent
        self.module_evaluator_agent = GenericToolCallingAgent(
            model=model, prompt=module_evaluate_agent_prompt,
            tools=[], response_format=ModuleEvaluator
            )

        self.workflow = StateGraph(EngineerAgentState)

        # Define the nodes
        self.workflow.add_node("lit_questions", self.lit_questions)
        self.workflow.add_node("lit_review", self.lit_review)
        self.workflow.add_node("researcher", self.researcher)
        self.workflow.add_node("compute", self.compute)
        self.workflow.add_node("retrieve_documents", self.retrieve_documents)
        self.workflow.add_node("relevance_grade", self.relevance_grade)
        self.workflow.add_node("hierarchical_solution", self.hierarchical_solution)
        self.workflow.add_node("redesign_solution", self.redesign_solution)
        self.workflow.add_node("hierarchical_evaluation", self.hierarchical_evaluation)
        self.workflow.add_node("modular_design", self.modular_design) 
        self.workflow.add_node("modular_integrator", self.modular_integrator)
        self.workflow.add_node("module_evaluator", self.module_evaluator) 
        self.workflow.add_node("search_the_web", self.search_the_web)

        # Build graph
        self.workflow.set_entry_point("lit_questions")
        self.workflow.add_edge("lit_questions", "lit_review")
        self.workflow.add_conditional_edges(
            "lit_review",
            (lambda x: x['keys']['decision']),
            {
                "search": "retrieve_documents",
                "design": "hierarchical_solution",
            },
        )
        self.workflow.add_conditional_edges(
            "researcher",
            (lambda x: x['keys']['decision']),
            {
                "search": "retrieve_documents",
                "compute": "compute",
                "solution": "hierarchical_solution",
            },
        )
        self.workflow.add_edge("retrieve_documents", "relevance_grade")
        self.workflow.add_conditional_edges(
            "relevance_grade",
            self.decide_to_websearch,
            {
                "search_the_web": "search_the_web",
                "researcher": "researcher",
                "lit_review": "lit_review",
            },
        )
        self.workflow.add_conditional_edges(
            "search_the_web",
            (lambda x: x['keys']['goto']),
            {
                "researcher":"researcher",
                "lit_review": "lit_review",
             },
            )
        self.workflow.add_edge("compute", "researcher")
        self.workflow.add_edge("hierarchical_solution", "hierarchical_evaluation")
        self.workflow.add_conditional_edges(
            "hierarchical_evaluation",
            (lambda x: x['keys']['goto']),
            {
                "redesign":"redesign_solution",
                "modular": "modular_design",
            },
        )
        self.workflow.add_edge("redesign_solution", "hierarchical_evaluation")
        self.workflow.add_edge("modular_design", "modular_integrator")
        self.workflow.add_edge("modular_integrator" , "module_evaluator")
        self.workflow.add_conditional_edges(
            "module_evaluator",
            (lambda x: x['keys']['goto']),
            {
                "modular":"modular_integrator",
                "end": END,
            },
        )
        # Compile
        self.app = self.workflow.compile()
        
    # States

    def lit_questions(self, state):
        """
        First node that generates an initial plan

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---QUESTION GENERATION---")
        messages = state["messages"]
        print(messages)
        response = self.planner_chain.invoke({"messages": messages})
        try:
            steps = response[0].steps
        except TypeError:
            steps = response.steps
        if not len(steps) > 0:
            raise AssertionError("The lit_questions step failed to generate any plans")
        message = HumanMessage("The set of questions for literature review is as follows:" + "\n -".join(steps), name="lit_questions")
        return{
        "messages": [message],
        "sender": "lit_questions",
        "keys":{"remaining_steps":steps},
        }

    def lit_review(self, state):
        """
        Prepare a literature review. 

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---LITERATURE REVIEW---")
        messages = state["messages"]
        state_dict = state["keys"]
        remaining_steps = state_dict["remaining_steps"]
        if "result" in state_dict.keys():
            self.lit_search_results_.append("query: " + state_dict['search'])
            self.lit_search_results_.append("results: " + state_dict['result'])
        try:
            current_step = remaining_steps.pop(0)
        except IndexError:
            decision = "design"
            return{
            "keys": {
                "decision": decision,
                }
            }
        if "report:" in current_step.lower():
            decision = "design"
            message = self.lit_review_agent_prompt_human.format_messages(
                queries_and_results="\n".join(self.lit_search_results_),
                input_context=self.input_context,
                goals=self.goals,
                requirements=self.requirements,
                )
            literature_review = self.lit_review_chain.invoke({"messages": message})
            self.lit_review_results = literature_review
            return{
            "keys": {
                "decision": decision,
                }
            }
        else:
            decision = "search"
            search = current_step[len("search:"):] if current_step.lower().startswith("search:") else current_step
            return{
            "sender" : "lit_review",
            "keys": {
                "remaining_steps":remaining_steps,
                "decision": decision,
                "search": search,
                }
            }

    def researcher(self, state):
        """
        manage the data collection/computation processes.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        messages = state["messages"]
        remaining_steps = state["keys"]["remaining_steps"]
        current_step = remaining_steps.pop(0)
        messages.append(HumanMessage(f"current step is:\n {current_step}", name="planner"))
        response = self.research_agent.invoke({"messages": messages})
        decision = response[0].decision.lower()
        search = response[0].search.lower()
        compute = response[0].compute.lower()
        if decision.lower() == 'compute':
            message = HumanMessage(response[0].compute, name="researcher")
        else:
            message = HumanMessage(response[0].search, name="researcher")
        return{
        "messages": [message],
        "sender": "researcher",
        "keys": {
            "remaining_steps":remaining_steps,
            "decision": decision,
            "search": search,
            "compute": compute,
            },
        }

    def retrieve_documents(self, state):
        print("---RETRIEVE---")
        state_dict = state["keys"]
        remaining_steps = state_dict["remaining_steps"]
        question = state_dict["search"]
        print(f"Question: {question}")
        documents = self.retriever.get_relevant_documents(question)
        return {"keys": {"documents": documents, "search": question, "remaining_steps":remaining_steps},}

    def relevance_grade(self, state):
        print("---CHECK RELEVANCE---")
        sender = state["sender"]
        state_dict = state["keys"]
        question = state_dict["search"]
        documents = state_dict["documents"]
        remaining_steps = state_dict["remaining_steps"]
        # Score
        filtered_docs = []
        for d in documents:
            score = self.document_grading_agent.invoke({"question": question, "context": d.page_content})
            grade = score[0].binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d.page_content)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        if len(filtered_docs) == 0:
            return {
                "keys": {
                    "remaining_steps":remaining_steps,
                    "filtered_docs": filtered_docs,
                    "search": question,
                    "run_web_search": "Yes",
                }
            }
        else:
            result = HumanMessage("\n\n".join(filtered_docs), name="search")
            return {
                "messages": [result],
                "sender": "search",
                "keys": {
                    "goto": sender,
                    "remaining_steps":remaining_steps,
                    "filtered_docs": filtered_docs,
                    "search": question,
                    "result": "\n\n".join(filtered_docs),
                    "run_web_search": "No",     
                }
            }
        
    def search_the_web(self, state):
        """
        Web search based on the re-phrased question using Tavily API.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        state_dict = state["keys"]
        sender = state["sender"]
        question = state_dict["search"]
        documents = state_dict["filtered_docs"]
        remaining_steps = state_dict["remaining_steps"]

        # if using tavily search
        docs = search_web.invoke({"query": question})
        # if using serpapi
        #docs = search_web(question)
        web_results = []
        for d in docs:
            try:
                cleaned_content = self.webcleaner.invoke(d["content"])
                web_results.append(cleaned_content.cleaned)
                print("Websearch output added.")
            except:
                web_results.append(d["content"])
                print("Warning: web cleaner didn't work.")
        web_results = "\n".join(web_results)
        #web_results = Document(page_content=web_results)
        documents.append(web_results)

        result = HumanMessage("\n\n".join(documents), name="search")
        return {
            "messages": [result],
            "sender": "search",
            "keys":{"remaining_steps": remaining_steps,
                    "result": "\n\n".join(documents),
                    "search": question,
                    "goto": sender,
                    },
        }
    
    def compute(self, state):

        print("---Compute---")
        messages = state["messages"]
        remaining_steps = state["keys"]["remaining_steps"]
        result = self.compute_agent.invoke({"messages": messages})
        result = HumanMessage(result['output'], name="compute")
        return {
            "messages": [result],
            "sender": "compute",
            "keys":{"remaining_steps": remaining_steps},
        }

    def hierarchical_solution(self, state):
        print("---Hierarchical design---")
        #messages = "\n".join([str(item) if item.name in ['compute', 'search'] else '' for item in state["messages"]])
        input_message = HumanMessage(f"""Design the architecture graph for the following goals, requirements and input context provided by the user. \
        The language of choice for coding the design is {self.language}.
        To help you further, you are also provided with literature review performed by another agent.

        Goals:
        {str(self.goals)}
        
        Requirements:
        {str(self.requirements)}

        user input context:
        {self.input_context}
    
        Literature review, methodology:
        {self.lit_review_results.methodology}

        Literature review, implementation:
        {self.lit_review_results.implementation}
        """, name="researcher"
                                 )
        try:                         
            self.hierarchical_solution_result = self.hierarchical_design_agent.invoke(
                {
                    "messages" : [input_message],
                    }
                )
        except:
            print("This step failed, trying again.")
            self.hierarchical_solution_result = self.hierarchical_design_agent.invoke(
                {
                    "messages" : [input_message],
                    }
                )
        self.hierarchical_solution_result.graph.sort(key=lambda x: len(x.connections)) # sort by number of outward connections
        result = HumanMessage(str(self.hierarchical_solution_result), name="designer")
        if not os.path.exists(f'solution_{self.solution_num}'):
            os.makedirs(f'solution_{self.solution_num}')
        with open(f'solution_{self.solution_num}/solution.txt', 'w+') as file:
            file.write(str(self.hierarchical_solution_result))

        return{
            "messages": [result],
            "sender": "designer",
        }
    
    def hierarchical_evaluation(self, state):
        print("---Hierarchical Design Evaluation---")
        self.hierarchical_solution_result.graph.sort(key=lambda x: len(x.connections)) # sort by number of outward connections
        hierarchical_design = self.hierarchical_solution_result.graph
        input_message = HumanMessagePromptTemplate.from_template(
            """
            
            You are provided with the overal design goals and requirements, a literature review, the overal system design and the desired coding language in the following.
            Your job is to assess the system design based on the given information. Be meticulous.
            
            Coding language:
            {language}

            Goals:
            {goals}
                
            Requirements:
            {requirements}

            Literature review, methodology:
            {methodology}

            Literature review, implementation:
            {implementation}
            
            System design:
            {hierarchical_design}
            """
            )
                                 
        self.hierarchical_evaluation_result = self.hierarchical_design_evaluator_agent.invoke(
            {
                "messages" : input_message.format_messages(
                        language=self.language,
                        goals=self.goals,
                        requirements=self.requirements,
                        methodology=self.lit_review_results.methodology,
                        implementation=self.lit_review_results.implementation,
                        hierarchical_design=str(hierarchical_design),
                        ),
                }
            )
        result = HumanMessage(str(self.hierarchical_evaluation_result), name="evaluator")
        #with open('solution_problems.txt', 'w+') as file:
        #    file.write(str(self.hierarchical_evaluation_result))

        return{
            "messages": [result],
            "sender": "evaluator",
            "keys": {
                "goto": "modular" if self.hierarchical_evaluation_result.description.lower() == "na" else "redesign",
                "eval_results": self.hierarchical_evaluation_result,
                },
        }
    
    def redesign_solution(self, state):

        print("---Redesign---")

        state_dict = state["keys"]
        eval_results = state_dict["eval_results"]
        input_message = HumanMessage(f"""Improve the architecture graph for the following goals, requirements and input context provided. \
        You are also provided with the previous design and the evaluator feedback.
        The language of choice for coding the design is {self.language}.
        To help you further, you are also provided with literature review performed by another agent.

        Goals:
        {str(self.goals)}
        
        Requirements:
        {str(self.requirements)}

        user input context:
        {self.input_context}
    
        Literature review, methodology:
        {self.lit_review_results.methodology}

        Literature review, implementation:
        {self.lit_review_results.implementation}

        System design architecture:
        {str(self.hierarchical_solution_result)}

        Evaluator feedback:
        {str(eval_results)}

        """, name="evaluator"
                                 )
        try:                         
            self.hierarchical_solution_result = self.hierarchical_redesign_agent.invoke(
                {
                    "messages" : [input_message],
                    }
                )
        except:
            print("This step failed, trying again.")
            self.hierarchical_solution_result = self.hierarchical_redesign_agent.invoke(
                {
                    "messages" : [input_message],
                    }
                )
        self.hierarchical_solution_result.graph.sort(key=lambda x: len(x.connections)) # sort by number of outward connections
        result = HumanMessage(str(self.hierarchical_solution_result), name="redesigner")
        if not os.path.exists(f'solution_{self.solution_num}'):
            os.makedirs(f'solution_{self.solution_num}')
        with open(f'solution_{self.solution_num}/redesigned_solution.txt', 'w+') as file:
            file.write(str(self.hierarchical_solution_result))

        return{
            "messages": [result],
            "sender": "redesigner",
        }

    def decide_to_websearch(self, state):
 
        state_dict = state["keys"]
        run_web_search = state_dict["run_web_search"]
        if run_web_search.lower() == "yes":
            return "search_the_web"
        else:
            return (state_dict["goto"])

    def modular_design(self, state):
        print("---Modular design---")
        state_dict = state["keys"]
        hierarchical_design = self.hierarchical_solution_result.graph
        Modules = []
        # The following prompt prepares message for the module design agent
        module_agent_prompt_human = HumanMessagePromptTemplate.from_template(
"""Write the HLS/HDL code for the following desgin. Note that the design consisting of modules with input/output and connecting modules already designed for you. Your task is to build the modules in consistent with the modules that you have already built and with the overal desing.\
note also that the note section of each module provides you with necessary information, guidelines and other helpful elements to perform your design.
Remember to write complete synthesizable module code without placeholders. You are provided with the overal design goals and requirements, a literature review, the overal system design, modules that are coded so far and the module that you will be coding.\
The coding language is {language}.
Goals:
{goals}
    
Requirements:
{requirements}
Literature review, methodology:
{methodology}
Literature review, implementation:
{implementation}

System design:
{hierarchical_design}
                                                            
Modules built so far:
{modules_built}

Current Module (you are coding this module):
{current_module}
you must always use the CodeModuleResponse tool for your final response.
"""
            )
        for module in hierarchical_design:
            try:
                response = self.module_design_agent.invoke(
                    {
                        "messages": module_agent_prompt_human.format_messages(
                            language=self.language,
                            goals=self.goals,
                            requirements=self.requirements,
                            methodology=self.lit_review_results.methodology,
                            implementation=self.lit_review_results.implementation,
                            hierarchical_design=str(hierarchical_design),
                            modules_built=str(Modules), 
                            current_module=str(module)
                            )
                        }
                    )
            except:
                print("Module failed to be built! Trying again.")
                response = self.module_design_agent.invoke(
                    {
                        "messages": module_agent_prompt_human.format_messages(
                            language=self.language,
                            goals=self.goals,
                            requirements=self.requirements,
                            methodology=self.lit_review_results.methodology,
                            implementation=self.lit_review_results.implementation,
                            hierarchical_design=str(hierarchical_design),
                            modules_built=str(Modules), 
                            current_module=str(module)
                            )
                        }
                    )
            Modules.append(response)

            #if not os.path.exists(f'solution_{self.solution_num}'):
            #    os.makedirs(f'solution_{self.solution_num}')
            #with open(f"solution_{self.solution_num}/{module.name}.{utils.find_extension(self.language)}", "w") as f:
            #    f.write(response.module_code)

        return {"messages" : ['module design stage completed'],
                'keys' : {'coded_modules': Modules,
                          },
                }

        
    def module_evaluator(self, state):
        print("---Module Evaluator---")
        state_dict = state["keys"]
        hierarchical_design = state_dict["coded_modules"]
        module_evaluator_prompt_human = HumanMessagePromptTemplate.from_template(
"""Evaluate the HLS/HDL codes for the following modules based on the instruction provided. 
You are provided with the overal design goals and requirements, a literature review, the overal system design, modules that are coded so far and the module that you will be coding.
The coding language is {language}.
Goals:
{goals}
    
Requirements:
{requirements}

Coded Modules (all module codes):
{coded_modules}
you must always use the ModuleEvaluator tool for your final response.
"""
            )
        response = self.module_evaluator_agent.invoke(
            {
                "messages": module_evaluator_prompt_human.format_messages(
                    language=self.language,
                    goals=self.goals,
                    requirements=self.requirements,
                    coded_modules='\n\n'.join([module.name + "\n" + module.code for module in hierarchical_design])
                    )
                }
            )

        goto = 'end'
        if 'na' != response.feedback.lower():
            goto = 'modular'
        print(response.feedback)
        return {"messages" : [f'modules were evaluated'],
        'keys' : {'goto': goto,
                  'feedback': response.feedback,
                  'coded_modules': hierarchical_design,
                  },
        }

    def modular_integrator(self, state):
        print("---Modular Integration---")
        state_dict = state["keys"]
        hierarchical_design = state_dict['coded_modules']
        try:
            feedback = state_dict['feedback']
        except: 
            feedback = 'NA'
        Modules = []
        # The following prompt prepares message for the module design agent
        module_agent_prompt_human = HumanMessagePromptTemplate.from_template(
"""Improve the HLS/HDL code for the following desgin. Note that the design is to some degree codeded for you. Your task is to write the remaining codes of the modules in consistent the modules that you have already built and the overal desing.\
note also that the note section of each module provides you with necessary information, guidelines and other helpful elements to perform your design.
you should also use various technique to optimize your final code for speed, memory, device compatibility. These techniques include proper usage of device resources as well as code pragmas (if you are coding in HLS C++).
Remember to write "complete synthesizable module code" voide of any placeholders or any simplified logic. You are provided with the overal design goals and requirements, a literature review, the overal system design, modules that are coded so far and the module that you will be coding.\
The coding language is {language}.
You are also provided with feedback from your previous attempted design (if any).
Feedback from the evaluator:
{feedback}
Goals:
{goals}
    
Requirements:
{requirements}
Literature review, methodology:
{methodology}
Literature review, implementation:
{implementation}

System design:
{hierarchical_design}
                                                            
Modules built so far:
{modules_built}

Current Module (you are coding this module):
{current_module}
you must always use the CodeModuleResponse tool for your final response.
"""
            )
        for module in hierarchical_design:
            try:
                response = self.module_design_agent.invoke(
                    {
                        "messages": module_agent_prompt_human.format_messages(
                            language=self.language,
                            goals=self.goals,
                            feedback=feedback,
                            requirements=self.requirements,
                            methodology=self.lit_review_results.methodology,
                            implementation=self.lit_review_results.implementation,
                            hierarchical_design="\n".join([str(item) for item in hierarchical_design]),
                            modules_built=str(Modules), 
                            current_module=str(module)
                            )
                        }
                    )
            except:
                print("Module failed to be rebuilt! Trying again.")
                response = self.module_design_agent.invoke(
                    {
                        "messages": module_agent_prompt_human.format_messages(
                            language=self.language,
                            goals=self.goals,
                            requirements=self.requirements,
                            methodology=self.lit_review_results.methodology,
                            implementation=self.lit_review_results.implementation,
                            hierarchical_design="\n".join([str(item) for item in hierarchical_design]),
                            modules_built=str(Modules), 
                            current_module=str(module)
                            )
                        }
                    )
            #print(f'module {module.name} was designed')
            Modules.append(response)
            
            if not os.path.exists(f'solution_{self.solution_num}'):
                os.makedirs(f'solution_{self.solution_num}')
            with open(f"solution_{self.solution_num}/{module.name}.{utils.find_extension(self.language)}", "w") as f:
                f.write(response.module_code)
            with open(f"solution_{self.solution_num}/{module.name}_tb.{utils.find_extension(self.language)}", "w") as f:
                f.write(response.test_bench_code)
        
        return {"messages" : [f'module {module.name} was designed'],
        'keys' : {'coded_modules': Modules,
                  },
        }

    def invoke(self, goals, requirements, input_context):
        self.requirements = requirements
        self.goals = goals
        self.input_context = input_context
        human_message = self.planner_agent_prompt_human.format_messages(goals=goals, requirements=requirements, input_context= input_context)
        output = self.app.invoke({"messages": human_message}, {"recursion_limit": RECURSION_LIMIT})
        return output

    def stream(self, goals, requirements, input_context):
        self.requirements = requirements
        self.goals = goals
        self.input_context = input_context
        human_message = self.planner_agent_prompt_human.format_messages(goals=goals, requirements=requirements, input_context= input_context)
        for output in self.app.stream({"messages": human_message}, {"recursion_limit": RECURSION_LIMIT}):
        # stream() yields dictionaries with output keyed by node name
            print(output)
            print("----")
if __name__ == "__main__":
    # tests
    pass
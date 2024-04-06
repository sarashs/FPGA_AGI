try:
    from FPGA_AGI.tools import search_web, python_run, Thought
except ModuleNotFoundError:
    from tools import search_web, python_run, Thought

try:
    from FPGA_AGI.prompts import hierarchical_agent_prompt
except ModuleNotFoundError:
    from prompts import hierarchical_agent_prompt

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
import json
from langgraph.prebuilt import ToolExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool, format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
import operator
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.schema import Document
from operator import itemgetter
from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain_community.tools.tavily_search import TavilySearchResults
from pprint import pprint

class Module(BaseModel):
    """module definition"""
    name: str = Field(description="Name of the module.")
    description: str = Field(description="Module description.")
    connections: List[str] = Field(description="List of the modules connecting to this module.")
    ports: List[str] = Field(description="List of input output ports inlcuding clocks, reset etc.")
    notes: str = Field(description="Any points, notes, information or computations necessary for the implementation of module.")

class HierarchicalResponse(BaseModel):
    """Final response to the user"""
    graph: List[Module] = Field(
        description="""List of modules"""
        )

class HierarchicalAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
class HierarchicalDesignAgent(object):
    """This agent performs a hierarchical design of the desired logic."""
#TODO: we will have to merge the tool node into the compute node
    def __init__(self, model: ChatOpenAI, tools: List=[search_web]):
        tools.append(Thought)
        self.tool_executor = ToolExecutor(tools)
        functions = [convert_to_openai_function(t) for t in tools]
        functions.append(convert_to_openai_function(HierarchicalResponse))
        self.model = model.bind_functions(functions)
        # Define a new graph
        self.workflow = StateGraph(HierarchicalAgentState)

        # Define the two nodes we will cycle between
        self.workflow.add_node("agent", self.call_model)
        self.workflow.add_node("action", self.call_tool)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.workflow.set_entry_point("agent")

        # We now add a conditional edge
        self.workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
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
            return "end"
        # Otherwise if there is, we need to check what type of function call it is
        elif last_message.additional_kwargs["function_call"]["name"] == "HierarchicalResponse":
            return "end"
        # Otherwise we continue
        else:
            return "continue"

    # Define the function that calls the model
    def call_model(self, state):
        messages = state["messages"]
        response = self.model.invoke(messages)
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
        if last_message.additional_kwargs["function_call"]["name"] != 'Thought':
            return {"messages": [function_message]}
        else:
            return

    def invoke(self, goals, requirements):
        inputs = {'messages': hierarchical_agent_prompt.format_prompt(goals=goals, requirements='\n'.join(requirements)).to_messages()}
        output = self.app.invoke(inputs)
        out = json.loads(output['messages'][-1].additional_kwargs["function_call"]["arguments"])
        out = HierarchicalResponse.parse_obj(out)
        return out

    def stream(self, goals, requirements):
        inputs = {'messages': hierarchical_agent_prompt.format_prompt(goals=goals, requirements='\n'.join(requirements)).to_messages()}
        for output in self.app.stream(inputs):
        # stream() yields dictionaries with output keyed by node name
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
                print("\n---\n")

class ResearcherResponse(BaseModel):
    """Final response to the user"""
    web_results: str = Field(description="""Useful results from the web""")
    document_results: str = Field(description="""Useful results from the document database""")
    code_output: str = Field(description="""Any code execution results that may be useful for the design""")
    solution_approach: str = Field(description="""Description of the solution approach""")

class ResearcherAgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
        remaining_steps: A list consisting of the step by step plan.
        messages: The commumications between the agents
        sender: The agent who is sending the message
    """
    keys: Dict[str, any]
    remaining_steps: List[str]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class Researcher(object):
    """This agent performs research on the design."""

    def __init__(self, model: ChatOpenAI, retriever: Any):

        self.retriever = retriever
        self.model = model

        #### Plan Agent
        class Plan(BaseModel):
            """Plan to follow in future"""

            steps: List[str] = Field(
                description="different steps to follow, should be in sorted order"
            )
            
        # Tool
        plan_tool_oai = convert_to_openai_tool(Plan)

        # Parser
        plan_parser_tool = PydanticToolsParser(tools=[Plan])

        # LLM with tool and enforce invocation
        planner_with_tool = self.model.bind(
            tools=[plan_tool_oai],
            tool_choice={"type": "function", "function": {"name": "Plan"}},
        )

        # prompt
        planner_prompt = ChatPromptTemplate.from_messages(
            [(
                    "system",
                    """You are a hardware design engineer and your purpose is to come up with a step by step plan that will enable the design engineers to actually design a hardware based on the set of goals, requirements and the context given by the user. \
                    This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. You are not responsible for simulations and testing. You are only responsible for design. \
                    Your steps should consist of:
                    - search when something needs to be searched via the internet or corpus of data. \
                    - computes when something needs to be computed via writing a python code. This should include parameters, functions etc that would be coded into the final design. For example generating the values for a lookup table implementation of a non-linear function is a compute step.\
                    - solution when we have all of the necessary information for enerating the final design. \
                    The result of the final step should be "solution: writing the final design in the form of an HDL/HLS code." Make sure that each step has all the information needed - do not skip steps."""
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        # This is the input prompt for when the object is invoked or streamed
        self.planner_agent_prompt_human = HumanMessagePromptTemplate.from_template("""You are planning the procedure for the following. Your goal is to collect all the data and perform all the computation necessary for another hardware engineer to build the design seamlessly.
                    goals:
                    {goals}
                    requirements:
                    {requirements}
                    user input context:
                    {input_context}""")
        # Chain
        self.planner_agent = (planner_prompt
            | planner_with_tool
            | plan_parser_tool
        )
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
        self.compute_agent = AgentExecutor(agent=agent_with_tool, tools=[python_run])

        
        self.workflow = StateGraph(ResearcherAgentState)

        # Define the nodes
        self.workflow.add_node("planner", self.planner)
        self.workflow.add_node("researcher", self.researcher)
        self.workflow.add_node("compute", self.compute)
        self.workflow.add_node("retrieve_documents", self.retrieve_documents)
        self.workflow.add_node("relevance_grade", self.relevance_grade)
        self.workflow.add_node("generate_excerpts", self.generate_excerpts) 
        self.workflow.add_node("search_web", self.search_web)

        # Build graph
        self.workflow.set_entry_point("planner")
        self.workflow.add_edge("planner", "researcher")
        self.workflow.add_conditional_edges(
            "researcher",
            (lambda x: x['keys']['decision']),
            {
                "search": "retrieve_documents",
                "compute": "compute",
                "solution": "generate_excerpts",
            },
        )
        self.workflow.add_edge("retrieve_documents", "relevance_grade")
        self.workflow.add_conditional_edges(
            "relevance_grade",
            self.decide_to_websearch,
            {
                "search_web": "search_web",
                "researcher": "researcher",
            },
        )
        self.workflow.add_edge("search_web", "researcher")
        self.workflow.add_edge("compute", "researcher")
        self.workflow.add_edge("generate_excerpts", END)

        # Compile
        self.app = self.workflow.compile()
        
    # States

    def planner(self, state):
        """
        First node that generates an initial plan

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        messages = state["messages"]
        print(messages)
        response = self.planner_agent.invoke({"messages": messages})
        steps = response[0].steps
        if not len(steps) > 0:
            raise AssertionError("The planner step failed to generate any plans")
        message = HumanMessage("The plan as per the planner agent is as follows:" + "\n -".join(steps), name="planner")
        return{
        "messages": [message],
        "sender": "planner",
        "keys": {
                 "remaining_steps":steps,
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
        "messages": [HumanMessage(f"current step is:\n {current_step}"), message],
        "sender": "researcher",
        "keys": {
            "decision": decision,
            "search": search,
            "compute": compute,
            "remaining_steps":remaining_steps,
            }
        }

    def retrieve_documents(self, state):
        print("---RETRIEVE---")
        state_dict = state["keys"]
        remaining_steps = state_dict["remaining_steps"]
        question = state_dict["search"]
        documents = self.retriever.get_relevant_documents(question)
        return {"keys": {"documents": documents, "search": question, "remaining_steps": remaining_steps}}

    def relevance_grade(self, state):
        print("---CHECK RELEVANCE---")
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
                    "filtered_docs": filtered_docs,
                    "search": question,
                    "run_web_search": "Yes",
                    "remaining_steps": remaining_steps,
                }
            }
        else:
            result = HumanMessage("\n\n".join(filtered_docs), name="search")
            return {
                "messages": [result],
                "sender": "search",
                "keys": {
                    "filtered_docs": filtered_docs,
                    "search": question,
                    "run_web_search": "No",
                    "remaining_steps": remaining_steps,
                }
            }
        
    def search_web(self, state):
        """
        Web search based on the re-phrased question using Tavily API.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        state_dict = state["keys"]
        question = state_dict["search"]
        documents = state_dict["filtered_docs"]
        remaining_steps = state_dict["remaining_steps"]

        tool = TavilySearchResults()
        docs = tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        #web_results = Document(page_content=web_results)
        documents.append(web_results)

        result = HumanMessage("\n\n".join(documents), name="search")
        return {
            "messages": [result],
            "sender": "search",
            "remaining_steps": remaining_steps,
        }
    
    def compute(self, state):

        print("---Compute---")
        state_dict = state["keys"]
        messages = state["messages"]
        remaining_steps = state_dict["remaining_steps"]
        result = self.compute_agent.invoke({"messages": messages})
        result = HumanMessage(result['output'], name="compute")
        return {
            "messages": [result],
            "sender": "compute",
            "remaining_steps": remaining_steps,
        }

    def generate_excerpts(self, state):
        print(state["messages"])

    def decide_to_websearch(self, state):
 
        state_dict = state["keys"]
        run_web_search = state_dict["run_web_search"]
        if run_web_search.lower() == "yes":
            return "search_web"
        else:
            return "researcher"


    def invoke(self, goals, requirements, input_context):
        human_message = self.planner_agent_prompt_human.format_messages(goals=goals, requirements=requirements, input_context= input_context)
        output = self.app.invoke({"messages": human_message}, {"recursion_limit": 150})
        return output

    def stream(self, goals, requirements, input_context):
        human_message = self.planner_agent_prompt_human.format_messages(goals=goals, requirements=requirements, input_context= input_context)
        for output in self.app.stream({"messages": human_message}, {"recursion_limit": 150}):
        # stream() yields dictionaries with output keyed by node name
            print(output)
            print("----")
if __name__ == "__main__":
    from pprint import pprint   

    #gpt-4-0125-preview
    model = ChatOpenAI(model='gpt-4-1106-preview', temperature=0, streaming=True)

    tools = [search_web]

    agent = HierarchicalDesignAgent(model, tools)

    hierarchical_response = agent.invoke('Design an 8-bit RISC V processor using SystemVerilog.', ['1. The processor must be designed based on the RISC V instruction set and '
                                            'should follow a Harvard-type data path structure.',
                                            '2. Implement memory access instructions including Load Word (LD) and Store '
                                            'Word (ST) with the specified operations.',
                                            '3. Implement data processing instructions including Add (ADD), Subtract '
                                            '(SUB), Invert (INV), Logical Shift Left (LSL), Logical Shift Right (LSR), '
                                            'Bitwise AND (AND), Bitwise OR (OR), and Set on Less Than (SLT) with the '
                                            'specified operations.',
                                            '4. Implement control flow instructions including Branch on Equal (BEQ), '
                                            'Branch on Not Equal (BNE), and Jump (JMP) with the specified operations.',
                                            '5. Design the processor control unit to generate appropriate control signals '
                                            'for each instruction type.',
                                            '6. Design the ALU control unit to generate the correct ALU operation based '
                                            'on the ALUOp signal and the opcode.',
                                            '7. Implement instruction memory, data memory, register file, ALU, ALU '
                                            'control unit, datapath unit, and control unit modules in SystemVerilog.',
                                            '8. Ensure the processor supports a 16-bit instruction format and operates on '
                                            '8-bit data widths.',
                                            '9. The processor must be capable of executing the provided instruction set '
                                            'with correct control and data flow for each instruction type.',
                                            '10. Verify the processor design using a testbench that simulates various '
                                            'instruction executions and validates the functionality of the processor.']
                                            )
    print(hierarchical_response)
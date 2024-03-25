try:
    from FPGA_AGI.tools import search_web, python_run, Thought
except ModuleNotFoundError:
    from tools import search_web, python_run, Thought

try:
    from FPGA_AGI.prompts import hierarchical_agent_prompt
except ModuleNotFoundError:
    from prompts import hierarchical_agent_prompt

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
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.schema import Document
from operator import itemgetter

from langchain_community.tools.tavily_search import TavilySearchResults

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
        messages: The commumications between the agents
        sender: The agent who is sending the message
    """
    keys: Dict[str, any]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

class ResearcherAgent(object):
    """This agent performs research on the design."""

    def __init__(self, model: ChatOpenAI, retriever: Any):

        self.retriever = retriever
        self.model = model
        self.workflow = StateGraph(ResearcherAgentState)

        # Define the nodes
        self.workflow.add_node("researcher", self.researcher)
        self.workflow.add_node("retrieve_documents", self.retrieve_documents)
        self.workflow.add_node("relevance_grade", self.relevance_grade)
        self.workflow.add_node("evaluate_results", self.relevance_grade)
        self.workflow.add_node("generate_excerpts", self.generate_excerpts) 
        self.workflow.add_node("search_web", self.search_web)

        # Build graph
        self.workflow.set_entry_point("researcher")
        self.workflow.add_edge("researcher", "retrieve_documents")
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
        self.workflow.add_conditional_edges(
            "evaluate_results",
            self.decide_to_generate,
            {
                "researcher": "researcher",
                "generate_excerpts": "generate_excerpts",
            },
        )
        self.workflow.add_edge("evaluate_results", "generate")
        self.workflow.add_edge("generate_excerpts", END)

        # Compile
        self.app = self.workflow.compile() 

    # States 
    def researcher(self, state):
        """
        manage the data collection/computation processes. This agent is in fact a router.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        
        goals = state['keys']['goals']
        requirements = state['keys']['requirements']
        context = state['keys']['context']
        
        class decision(BaseModel):
            """Decision regarding the next step of the process"""

            decision: str = Field(description="Decision 'search' or 'compute' or 'solution'", default='search')
            search: str = Field(description="If decision search, query to be searched else, NA", default='NA')
            compute: str = Field(description="If decision compute, description of what needs to be computed else, NA", default='NA')
        
        # Tool
        decision_tool_oai = convert_to_openai_tool(decision)
        # LLM with tool and enforce invocation
        llm_with_tool = self.model.bind(
            tools=[decision_tool_oai],
            tool_choice={"type": "function", "function": {"name": "decision"}},
        )
        
        # Parser
        parser_tool = PydanticToolsParser(tools=[decision])

        # prompt
        prompt = PromptTemplate.from_messages(
            [SystemMessage(
                    "system","""
                    You are a hardware engneer whose job is to collect all of the information needed to create a new solution.
                    You are collaborating with some assistants.
                    Your job consists of coming up with what needs to be searched or computed and then making a decision to use 
                    the search assistant, the compute assisstant or the final solution excerpt generator.
                    You ask your questions or perform your computations one at a time.
                    You are researching the necessary information for the following,
                    goals:
                    {goals}
                    requirements:
                    {requirements}
                    user input context:
                    {context}
                    You have access to the following assisstants: 
                    search assisstant: This assisstant is going to perform document and internet searches.
                    compute assisstant: This assisstant is going to generate a python code based on your request, run it and share the results with you.
                    solution assisstant: This assisstant is going to generate the final excerpt based on the interaction you had with the other two agents."""
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        #prompt = prompt.partial(goals=goals, requirements=requirements, context=context)
        #prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        
        # Chain
        chain = (
            {
                "context": itemgetter("context"),
                "goals": itemgetter("goals"),
                "requirements": itemgetter("requirements"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        code_solution = chain.invoke(
            {"question": question, "generation": str(code_solution[0]), "error": error}
        )

    def retrieve_documents(self, state):
        print("---RETRIEVE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = self.retriever.get_relevant_documents(question)
        return {"keys": {"documents": documents, "question": question}}

    def relevance_grade(self, state):
        print("---CHECK RELEVANCE---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""

            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # Tool
        grade_tool_oai = convert_to_openai_tool(grade)

        # LLM with tool and enforce invocation
        llm_with_tool = self.model.bind(
            tools=[grade_tool_oai],
            tool_choice={"type": "function", "function": {"name": "grade"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[grade])

        # Prompt
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # Chain
        chain = prompt | llm_with_tool | parser_tool

        # Score
        filtered_docs = []
        search = "No"  # Default do not opt for web search to supplement retrieval
        for d in documents:
            score = chain.invoke({"question": question, "context": d.page_content})
            grade = score[0].binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        if len(filtered_docs) == 0:
            search = "Yes"

        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "run_web_search": search,
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
        question = state_dict["question"]
        documents = state_dict["documents"]

        tool = TavilySearchResults()
        docs = tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"keys": {"documents": documents, "question": question}}
    
    def compute(self, state):

        print("---Compute---")
        state_dict = state["keys"]
        code = state_dict["code"]


    def evaluate_results(self, state):
        pass

    def generate_excerpts(self, state):
        pass

    #conditions
    def decide_to_generate(self, state):
        pass

    def decide_to_websearch(self, state):
        pass

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

"""

        functions = [format_tool_to_openai_function(t) for t in tools]

        prompt = PromptTemplate.from_messages(
            [SystemMessage(
                    "system",
                    You are a helpful AI assistant, collaborating with other assistants.
                    Use the provided tools to progress towards answering the question.
                    Execute what you can to make progress.
                    If you or any of the other assistants have the final answer or deliverable,
                    prefix your response with FINAL ANSWER so the team knows to stop.
                    You have access to the following tools: {tool_names}.,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        chain = prompt | self.model.bind_functions(functions)
"""
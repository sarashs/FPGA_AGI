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
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Sequence, List, Dict
import operator
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, FunctionMessage

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
    """

    keys: Dict[str, any]

class ResearcherAgent(object):
    """This agent performs research on the design."""

    def __init__(self, model: ChatOpenAI, tools: List=[search_web]):
        self.workflow = StateGraph(ResearcherAgentState)

        # Define the nodes
        self.workflow.add_node("generate_query", self.generate_query)
        self.workflow.add_node("retrieve_documents", self.retrieve_documents)
        self.workflow.add_node("relevance_grade", self.relevance_grade)
        self.workflow.add_node("decide_to_generate", self.decide_to_generate)
        self.workflow.add_node("generate_excerpts", self.generate_excerpts) 
        self.workflow.add_node("search_web", self.search_web)

        # Build graph
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "web_search")
        self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("generate", END)

        # Compile
        app = self.workflow.compile() 

    def generate_query(self, state):
        pass

    def retrieve_documents(self, state):
        pass

    def relevance_grade(self, state):
        pass

    def search_web(self, state):
        pass

    def decide_to_generate(self, state):
        pass

    def generate_excerpts(self, state):
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
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
from typing import TypedDict, Annotated, Sequence, List
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

    def prompt(self, goals, requirements):
        return ({"messages": [SystemMessage(content=f"""You are an FPGA design engineer whose purpose is to design the architecture graph of a HDL hardware project.
            You are deciding on the module names, description, ports, what modules each module is connected to. You also include notes on anything that may be necessary in order for the downstream logic designers.
            - You must expand your knowledge on the subject matter via the seach tool before committing to a response.
            - You are not responsible for designing a test bench.
            - If you are defining a top module or any other hierarchy, you must mention that in the module description.

            Use the following format:

            Thought: You should think of an action. You do this by calling the Though tool/function. This is the only way to think.
            Action: the action to take, should be one of the functions you have access to.
            ... (this Thought/Action can repeat 3 times)
            Response: You should use the HierarchicalResponse tool to format your response. Do not return your final response without using the HierarchicalResponse tool"""),
            HumanMessage(content=f"""Design the architecture graph for the following goals and requirements.

                Goals:
                {goals}
                
                Requirements:
                {requirements}
                """)]})

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
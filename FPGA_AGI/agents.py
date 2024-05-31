try:
    from FPGA_AGI.tools import search_web, python_run, Thought
    from FPGA_AGI.prompts import *
    from FPGA_AGI.parameters import RECURSION_LIMIT
    from FPGA_AGI.chains import WebsearchCleaner, Planner, LiteratureReview
    import FPGA_AGI.utils as utils
except ModuleNotFoundError:
    from prompts import *
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

from langchain_openai import ChatOpenAI
#accurate_model = ChatOpenAI(model='gpt-4', temperature=0)

class Module(BaseModel):
    """module definition"""
    name: str = Field(description="Name of the module.")
    description: str = Field(description="Module description including detailed explanation of what the module does and how to achieve it. Think of it as a code equivalent of the module without coding it.")
    connections: List[str] = Field(description="List of the modules connecting to this module (be it via input or output).")
    ports: List[str] = Field(description="List of input output ports inlcuding clocks, reset etc.")
    module_template: str = Field(description="Outline of the xilinx HLS C++ code with placeholders and enough comments to be completed by a coder. The placeholders must be comments starting with PLACEHOLDER:")

class SystemDesign(BaseModel):
    """system design"""
    graph: List[Module] = Field(
        description="""List of modules"""
        )
    
class SystemEvaluator(BaseModel):
    """system design evaluation"""
    coding_language: str = Field(description="NA if coding language is HLS C++ otherwise explain the problem. Note that this is not regular C++ but one that is used by xilinx high level synthesis")
    functionality: str = Field(description="NA If the design achieves design goals and requirements, otherwise explain.")
    connections: str = Field(description="NA If the connections between modules are consistent or the input/outputs are connected properly, otherwise explain. THIS IS VERY IMPORTANT")
    excessive: str = Field(description="NA if the design is free of any excessive and/or superflous modules otherwise explain.")
    missing: str = Field(description="NA if the design is complete and is not missing any modules otherwise explain. In particular every design must have a module (HLS C++ function) named main which will be the main function in HLS C++.")
    template: str = Field(description="NA if the template code correctly identifies all of the place holders and correctly includes the module ports otherwise explain.")
    fail: bool = Field(description="true if the design fails in any of the coding_language, functionality, connections, excessive, missing and template otherwise false.")

### Module level evaluator
class ModuleEvaluator(BaseModel):
    """module evaluation"""
    coding_language: str = Field(description="NA if coding language is HLS C++ otherwise explain the problem. Note that this is not regular C++ but one that is used by xilinx high level synthesis")
    functionality: str = Field(description="NA if the design achieves design goals and requirements, otherwise explain.")
    connections: str = Field(description="NA if all the necessary connections are made between the modules, otherwise explain. This includes, all the necessary signals coming out or going into the module")
    interfaces: str = Field(description="NA if all the ports/wires/regs and their types and widths across different modules match, otherwise explain.")
    syntax: str = Field(description="NA if the code adheres to xilinx HLS c++ language components, otherwise explain.")
    placeholders: str = Field(description="NA if the code has any placeholders or otherwise missing components from a complete synthesizable code in this module, otherwise explain.")
    optimizations: str = Field(description="NA if the code is optimized in line with the goals and requirements, otherwise explain. For HLS C++ this is achieved via pragmas.")
    fail: bool = Field(description="true if the design fails any of the coding_language, functionality, connections, interfaces is false, port_type, syntax, placeholders or optimizations, else false.")

### Module Design agent
class CodeModuleResponse(BaseModel):
    """Final response to the user"""
    name: str = Field(description="Name of the module.")
    description: str = Field(description="Brief module description.")
    connections: List[str] = Field(description="List of the modules connecting to this module.")
    ports: List[str] = Field(description="List of input output ports inlcuding clocks, reset etc.")
    module_code: str = Field(description="Complete working synthesizable xilinx HLS C++ module code without any placeholders.")
    header_file: str = Field(description="Complete header file associated with the module code. Every module must have a header file in order to be included in the top module.")
    test_bench_code: str = Field(description="Complete behavioral test for the module must be written in the HLS C++ language. the testbench module name should be module name underline tb. It must have a main function that returns int.")

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
                 tools: List=[], response_format: BaseModel=SystemDesign):
        tools.append(Thought)
        self.tool_executor = ToolExecutor(tools)
        self.response_format = response_format
        functions = [convert_to_openai_function(t) for t in tools]
        functions.append(convert_to_openai_function(response_format))
        self.model = prompt | model.bind_functions(functions)
        self._failure_count = 0
        self._max_failurecount = 2
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
                "agent": "agent",
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
            if self._failure_count > self._max_failurecount:
                return 'end'
            else:
                self._failure_count += 1
                print(f"trying again: {self._failure_count}th out of {self._max_failurecount}")
                state["messages"].append(HumanMessage("There is an error in your output. Your output must be a function call as explained to you via the system message. Please respond via a function call and do not respond in any other form.", name="Moderator"))
                return 'agent'
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
        out = json.loads(output['messages'][-1].additional_kwargs["function_call"]["arguments"])
        print(out)
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

    def __init__(self, model: ChatOpenAI, evaluator_model: ChatOpenAI, retriever: Any, solution_num: int = 0):

        self.solution_num = solution_num
        self.retriever = retriever
        self.model = model
        self.evaluator_model = evaluator_model
        self.webcleaner = WebsearchCleaner.from_llm(llm=model)
        self.language = "HLS C++" # or "vhdl", "systemverilog", "cpp"
        self.input_context = None
        self.requirements = None
        self.goals = None
        self.lit_search_results_ = []
        self.lit_review_results = None
        self.hierarchical_solution_result = None

        #### lit review questions Agent
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
        # This is the input prompt for when the object is invoked or streamed
        self.lit_review_agent_prompt_human = HumanMessagePromptTemplate.from_template("""Prepare the document for the following list of queries and results given goals, requirements and input context given by the user.
                                                                                        The main component you are using for your wiretup is the queries and results. everything else is provided as context.
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
        agent_with_tool = create_openai_tools_agent(self.model, [python_run], compute_agent_prompt)
        self.compute_agent = AgentExecutor(agent=agent_with_tool, tools=[python_run], verbose=True)
        

    #### Hierarchical Design Agent
        self.hierarchical_design_agent = GenericToolCallingAgent(model=model, tools=[])

    #### Hierarchical Design Evaluator Agent
        self.hierarchical_design_evaluator_agent = GenericToolCallingAgent(
            response_format=SystemEvaluator,
            prompt=hierarchical_agent_evaluator,
            model=evaluator_model, #model,
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
            model=evaluator_model, prompt=module_evaluate_agent_prompt,
            tools=[], response_format=ModuleEvaluator
            )
        
        if "3.5" in model.model_name:
            self.module_design_agent._max_failurecount = 6
            self.hierarchical_design_agent._max_failurecount = 6

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

        if (self.lit_review_results):
            assert self.lit_review_results.methodology
            assert self.lit_review_results.implementation
            print("---Literature review already provided. Skipping this step.---")
            return{
            "keys": {
                "decision": "design",
                }
            }

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
        hierarchical_solution_human = HumanMessage(f"""Design the architecture graph for the following goals, requirements and input context provided by the user. \
        The language of choice for coding the design is Xilinx HLS C++.
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
        #try:                         
        self.hierarchical_solution_result = self.hierarchical_design_agent.invoke(
            {
                "messages" : [hierarchical_solution_human],
                }
            )
        #except:
        #    print("This step failed, trying again.")
        #    self.hierarchical_solution_result = self.hierarchical_design_agent.invoke(
        #        {
        #            "messages" : [hierarchical_solution_human],
        #            }
        #        )
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
        hierarchical_evaluation_human = HumanMessagePromptTemplate.from_template(
            """
            
            You are provided with the overal design goals and requirements, a literature review, the overal system design and the desired coding language in the following.
            Your job is to assess the system design based on the given information. Be meticulous.
            The design is coded in Xilinx HLS C++.

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
                "messages" : hierarchical_evaluation_human.format_messages(
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
                "goto": "redesign" if self.hierarchical_evaluation_result.fail else "modular",
                "eval_results": self.hierarchical_evaluation_result,
                },
        }
    
    def redesign_solution(self, state):

        print("---Redesign---")

        state_dict = state["keys"]
        eval_results = state_dict["eval_results"]
        redesign_human = HumanMessage(f"""Improve the architecture graph for the following goals, requirements and input context provided. \
        You are also provided with the previous design and the evaluator feedback. This design is in Xilinx HLS C++.
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
                    "messages" : [redesign_human],
                    }
                )
        except:
            print("This step failed, trying again.")
            self.hierarchical_solution_result = self.hierarchical_redesign_agent.invoke(
                {
                    "messages" : [redesign_human],
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
        for module in hierarchical_design:
            response = self.module_design_agent.invoke(
                {
                    "messages": modular_design_human_prompt.format_messages(
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
"""Evaluate the HLS C++ codes for the following modules based on the instruction provided. 
You are provided with the overal design goals and requirements, a literature review, the overal system design, modules that are coded so far and the module that you will be coding.
The coding language is Xilinx HLS C++.
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
                    coded_modules='\n\n'.join([module.name + "\n" + module.module_code for module in hierarchical_design])
                    )
                }
            )

        goto = 'end'
        if response.fail:
            goto = 'modular'
        return {"messages" : [response],
        'keys' : {'goto': goto,
                  'feedback': response,
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
The coding language is Xilinx HLS C++.
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
you must always use the CodeModuleResponse tool for your final response. Every thing you do is through function calls.
"""
            )
        for module in hierarchical_design:
            response = self.module_design_agent.invoke(
                {
                    "messages": module_agent_prompt_human.format_messages(
                        goals=self.goals,
                        feedback=str(feedback),
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
            if response.header_file != "NA":
                with open(f"solution_{self.solution_num}/{module.name}.h", "w") as f:
                    f.write(response.header_file)               
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
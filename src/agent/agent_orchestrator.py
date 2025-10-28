"""
Agent Orchestrator for Autonomous DTI Analysis.
Manages the LangChain agent and coordinates tool execution.
"""

import os
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import logging

from .agentic_tools import AGENTIC_TOOLS

logger = logging.getLogger(__name__)

import streamlit as st
from typing import List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from .agentic_tools import DrugInput, GeneInput, LiteratureInput
from ..core_processing import CoreProcessor
from ..utils.exceptions import DataFetchException, ModelException, ValidationException

class DTIAgentOrchestrator:
    """
    Manages the agent's state, tools, and execution logic.
    State is held within the class instance, not globally.
    """
    
    def __init__(self, core_processor: CoreProcessor, config: dict, api_key: str):
        self.core_processor = core_processor
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            model=config['agent']['llm_model'],
            google_api_key=api_key,
            temperature=0.0
        )
        self.prompt_template = self._load_prompt_template()
        
        # --- Instance-level State ---
        # This replaces the global 'agent_state' dictionary
        self.drug_name: Optional[str] = None
        self.gene_name: Optional[str] = None
        self.smiles_string: Optional[str] = None
        self.protein_sequence: Optional[str] = None
        self.prediction_result: Optional[dict] = None
        self.literature_context: Optional[str] = None
        # --- End State ---
        
        self.tools = self.get_tools()
        self.agent_executor = self.create_agent_executor()

    def _load_prompt_template(self) -> PromptTemplate:
        # (Your existing prompt loading logic)
        # For brevity, assuming it's loaded from config or a file
        template = """
        You are a Clinical Decision Support System agent.
        Your goal is to answer questions about drug-gene interactions.
        
        You must follow these steps:
        1.  Identify the drug and gene from the user's query.
        2.  Use `set_drug` to set the drug. This will fetch its molecular structure (SMILES).
        3.  Use `set_gene` to set the gene. This will fetch its protein sequence.
        4.  Once BOTH are set, use `get_dti_prediction` to predict the interaction.
        5.  If the user asks for context, use `search_literature` *after* setting the drug and gene.
        6.  Formulate a final answer based on the results.

        Tools:
        {tools}

        Conversation:
        {chat_history}
        
        User Query:
        {input}

        Agent Scratchpad:
        {agent_scratchpad}
        """
        return PromptTemplate.from_template(template)

    def get_tools(self) -> List:
        """Returns all agent tools as methods of this class."""
        return [
            self.set_drug,
            self.set_gene,
            self.get_dti_prediction,
            self.search_literature
        ]

    def create_agent_executor(self) -> AgentExecutor:
        """Creates the LangChain agent executor."""
        agent = create_react_agent(self.llm, self.tools, self.prompt_template)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str, chat_history: List) -> str:
        """Run the agent with the user query and chat history."""
        # Reset state for this run if it's a new query?
        # Or maintain state? For now, we maintain state.
        try:
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history
            })
            return response['output']
        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            return "I'm sorry, I encountered an error. Please try rephrasing your query."

    # --- AGENT TOOLS (Now as class methods) ---

    @tool(args_schema=DrugInput)
    def set_drug(self, drug_name: str) -> str:
        """
        Sets the drug for analysis. Fetches and validates the drug's
        SMILES string. This MUST be called before prediction.
        """
        try:
            self.drug_name = drug_name
            self.smiles_string = self.core_processor.get_smiles_sync(drug_name)
            return f"Drug set to {drug_name}. SMILES string successfully fetched and validated."
        except (ValidationException, DataFetchException) as e:
            self.drug_name = None
            self.smiles_string = None
            return f"Error: {e.message}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    @tool(args_schema=GeneInput)
    def set_gene(self, gene_name: str) -> str:
        """
        Sets the gene for analysis. Fetches and validates the gene's
        protein sequence. This MUST be called before prediction.
        """
        try:
            self.gene_name = gene_name
            self.protein_sequence = self.core_processor.get_sequence_sync(gene_name)
            return f"Gene set to {gene_name}. Protein sequence successfully fetched and validated."
        except (ValidationException, DataFetchException) as e:
            self.gene_name = None
            self.protein_sequence = None
            return f"Error: {e.message}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    @tool
    def get_dti_prediction(self) -> str:
        """
        Runs the DTI prediction model using the currently set drug and gene.
        You MUST call set_drug and set_gene successfully before using this tool.
        """
        if not self.smiles_string or not self.drug_name:
            return "Error: Drug not set. Please use set_drug first."
        if not self.protein_sequence or not self.gene_name:
            return "Error: Gene not set. Please use set_gene first."

        try:
            result = self.core_processor.run_model(self.smiles_string, self.protein_sequence)
            self.prediction_result = result # Save full result
            
            # Format a clean response
            prob = result['probability'] * 100
            pred_class = "Binds" if result['prediction'] == 1 else "Does not bind"
            
            return (
                f"Prediction for {self.drug_name} and {self.gene_name}:\n"
                f"- Classification: **{pred_class}**\n"
                f"- Binding Probability: **{prob:.2f}%**"
            )
        except (ModelException, FeaturizationException) as e:
            return f"Error during model prediction: {e.message}"
        except Exception as e:
            return f"An unexpected error occurred during prediction: {e}"

    @tool(args_schema=LiteratureInput)
    def search_literature(self, query: str) -> str:
        """
        Searches PubMed for literature relevant to the user's query.
        Best used *after* setting a drug and gene to provide context.
        """
        if not self.drug_name or not self.gene_name:
            st.warning("Running literature search without drug/gene context.")
            
        try:
            articles = self.core_processor.get_literature_sync(query, max_results=3)
            if not articles:
                return f"No relevant articles found for: '{query}'"
            
            context = "Here are the most relevant findings:\n\n"
            for i, art in enumerate(articles, 1):
                context += f"**{i}. {art['title']}** (PMID: {art['pmid']})\n"
                context += f"{art['abstract']}\n\n"
            
            self.literature_context = context
            return context
        except Exception as e:
            return f"Error searching literature: {e}"

class DTIAgentOrchestrator:
    """
    High-level orchestrator for the agentic DTI prediction system.
    """
    
    def __init__(self, config: Dict, api_key: Optional[str] = None):
        self.config = config
        self.agent_config = config['agent']
        
        # Set up API key
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY must be provided or set in environment")
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        logger.info("DTI Agent Orchestrator initialized")
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the language model."""
        if self.agent_config['llm_provider'] == 'google':
            return ChatGoogleGenerativeAI(
                model=self.agent_config['llm_model'],
                temperature=self.agent_config['temperature']
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.agent_config['llm_provider']}")
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with tools."""
        
        # Define the agent prompt
        prompt_template = """You are an expert AI assistant specializing in drug-gene interaction analysis,
acting as a Clinical Decision Support System (CDSS). Your role is to help researchers and clinicians
understand potential interactions between drugs and genes/proteins.

You have access to the following tools to perform comprehensive analyses:

{tools}

**CRITICAL INSTRUCTIONS:**
1. ALWAYS use tools in the correct sequence:
   - First: fetch_molecular_data_tool
   - Second: featurize_drug_protein_pair_tool
   - Third: predict_interaction_tool
   - Fourth (optional): explain_prediction_tool
   - Fifth (optional): search_literature_tool
   - Last (optional): synthesize_clinical_report_tool

2. When explaining predictions, focus on:
   - Molecular mechanisms (specific atoms, residues)
   - Biological plausibility
   - Clinical relevance

3. When citing literature, always include:
   - Paper titles
   - PMIDs
   - Brief relevance summary

4. For clinical reports:
   - Be clear about confidence levels
   - Provide actionable recommendations
   - Include appropriate disclaimers

**IMPORTANT:** Always acknowledge uncertainties and limitations. This system is for
research support, not definitive clinical diagnosis.

Begin your analysis!

User Query: {input}

Thought Process:
{agent_scratchpad}
"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=AGENTIC_TOOLS,
            prompt=prompt
        )
        
        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=AGENTIC_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=self.agent_config['max_iterations'],
            early_stopping_method="generate"
        )
        
        return executor
    
    def analyze_interaction(
        self,
        query: str,
        include_literature: bool = True,
        include_explanation: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete interaction analysis based on a natural language query.
        
        Args:
            query: Natural language description of the analysis request
            include_literature: Whether to search PubMed for supporting papers
            include_explanation: Whether to generate detailed molecular explanations
            generate_report: Whether to generate a clinical report
        
        Returns:
            Dictionary containing agent response and all gathered data
        """
        logger.info(f"Starting agentic analysis: {query}")
        
        # Enhance query with optional flags
        enhanced_query = query
        
        if include_explanation:
            enhanced_query += " Provide a detailed molecular explanation."
        
        if include_literature:
            enhanced_query += " Search for relevant research papers."
        
        if generate_report:
            enhanced_query += " Generate a comprehensive clinical report."
        
        try:
            # Run the agent
            response = self.agent_executor.invoke({"input": enhanced_query})
            
            logger.info("Agent analysis complete")
            
            return {
                "status": "success",
                "query": query,
                "response": response['output'],
                "intermediate_steps": response.get('intermediate_steps', [])
            }
        
        except Exception as e:
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "query": query,
                "error": str(e)
            }
    
    def quick_predict(self, drug_name: str, gene_name: str) -> Dict[str, Any]:
        """
        Quick prediction without explanations or literature search.
        """
        query = f"Predict the interaction between drug '{drug_name}' and gene '{gene_name}'. Just give me the probability."
        return self.analyze_interaction(
            query,
            include_literature=False,
            include_explanation=False,
            generate_report=False
        )
    
    def full_analysis(self, drug_name: str, gene_name: str) -> Dict[str, Any]:
        """
        Complete analysis with all features enabled.
        """
        query = f"Perform a complete analysis of the interaction between drug '{drug_name}' and gene '{gene_name}'."
        return self.analyze_interaction(
            query,
            include_literature=True,
            include_explanation=True,
            generate_report=True
        )

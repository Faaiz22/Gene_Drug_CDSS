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

# This import was from the first (incorrect) class definition
# from .agentic_tools import AGENTIC_TOOLS 

logger = logging.getLogger(__name__)

import streamlit as st
from typing import List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

# These imports are from the second (correct) class definition
from .agentic_tools import AGENTIC_TOOLS
from ..core_processing import CoreProcessor
from ..utils.exceptions import DataFetchException, ModelException, ValidationException


# --------------------------------------------------------------------
# DELETING THE FIRST, CONFLICTING CLASS DEFINITION
# The incorrect class (lines 20-149) has been removed.
# We are only keeping the second, more detailed class.
# --------------------------------------------------------------------


class DTIAgentOrchestrator:
    """
    High-level orchestrator for the agentic DTI prediction system.
    (This is the correct class definition that was previously at line 151)
    """
    
    def __init__(self, config: Dict, api_key: Optional[str] = None):
        self.config = config
        self.agent_config = config['agent']
        
        # Set up API key
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif not os.getenv("GOOGLE_API_KEY"):
            # This check is good. It will catch if main.py failed to set it.
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
            tools=AGENTENTIC_TOOLS,
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

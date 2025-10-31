"""
Agent Orchestrator for Autonomous DTI Analysis.
Manages the LangChain agent and coordinates tool execution.
"""

import os
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from .agentic_tools import AGENTIC_TOOLS

logger = logging.getLogger(__name__)


class DTIAgentOrchestrator:
    """
    High-level orchestrator for the agentic DTI prediction system.
    Coordinates LangChain agent with molecular data fetching, model inference,
    explanation generation, and literature search.
    """
    
    def __init__(self, config: Dict, api_key: Optional[str] = None):
        """
        Initialize the orchestrator with configuration and API credentials.
        
        Args:
            config: Application configuration dictionary
            api_key: Google API key for Gemini/PaLM models
            
        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.config = config
        self.agent_config = config.get('agent', {})
        
        # Validate and set API key
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY must be provided as argument or set in environment. "
                "Please enter your API key on the main CDSS page."
            )
        
        # Initialize LLM
        try:
            self.llm = self._initialize_llm()
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise ValueError(f"LLM initialization failed: {e}. Check your API key and internet connection.")
        
        # Create agent
        try:
            self.agent_executor = self._create_agent()
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise ValueError(f"Agent creation failed: {e}")
        
        logger.info("DTI Agent Orchestrator initialized successfully")
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """
        Initialize the language model based on configuration.
        
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If LLM provider is unsupported
        """
        provider = self.agent_config.get('llm_provider', 'google')
        
        if provider == 'google':
            model_name = self.agent_config.get('llm_model', 'gemini-1.5-pro-latest')
            temperature = self.agent_config.get('temperature', 0.0)
            
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                convert_system_message_to_human=True  # Required for Gemini
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Only 'google' is currently supported."
            )
    
    def _create_agent(self) -> AgentExecutor:
        """
        Create the ReAct agent with configured tools and prompts.
        
        Returns:
            Configured AgentExecutor instance
        """
        # Define the agent prompt template
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

Use the following format:

Thought: Consider what needs to be done
Action: the tool to use (one of [{tool_names}])
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have all the information I need to provide a final answer
Final Answer: the final response to the user

Begin your analysis!

User Query: {input}

{agent_scratchpad}
"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=AGENTIC_TOOLS,
            prompt=prompt
        )
        
        # Create executor with error handling
        max_iterations = self.agent_config.get('max_iterations', 15)
        
        executor = AgentExecutor(
            agent=agent,
            # ==================================================================
            # CORRECTED TYPO: Was AGENTENTIC_TOOLS
            # ==================================================================
            tools=AGENTIC_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=max_iterations,
            early_stopping_method="generate",
            return_intermediate_steps=True
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
            Dictionary containing:
            - status: 'success' or 'error'
            - query: Original query
            - response: Agent's final answer
            - intermediate_steps: List of tool calls and results
            - error: Error message if status is 'error'
        """
        logger.info(f"Starting agentic analysis for query: {query}")
        
        # Enhance query with optional flags
        enhanced_query = query
        
        if include_explanation:
            enhanced_query += " Provide a detailed molecular explanation of the interaction mechanism."
        
        if include_literature:
            enhanced_query += " Search PubMed for relevant research papers supporting this interaction."
        
        if generate_report:
            enhanced_query += " Generate a comprehensive clinical report with actionable recommendations."
        
        try:
            # Run the agent with error handling
            response = self.agent_executor.invoke(
                {"input": enhanced_query},
                return_only_outputs=False
            )
            
            logger.info("Agent analysis completed successfully")
            
            return {
                "status": "success",
                "query": query,
                "response": response.get('output', 'No response generated'),
                "intermediate_steps": response.get('intermediate_steps', [])
            }
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Agent execution failed: {error_msg}", exc_info=True)
            
            return {
                "status": "error",
                "query": query,
                "error": f"Analysis failed: {error_msg}",
                "response": None,
                "intermediate_steps": []
            }
    
    def quick_predict(self, drug_name: str, gene_name: str) -> Dict[str, Any]:
        """
        Quick prediction without explanations or literature search.
        
        Args:
            drug_name: Name or identifier of the drug
            gene_name: Name or identifier of the gene/protein
            
        Returns:
            Analysis results dictionary
        """
        query = (
            f"Predict the interaction probability between drug '{drug_name}' "
            f"and gene '{gene_name}'. Provide only the numerical probability "
            f"and a brief interpretation."
        )
        
        return self.analyze_interaction(
            query,
            include_literature=False,
            include_explanation=False,
            generate_report=False
        )
    
    def full_analysis(self, drug_name: str, gene_name: str) -> Dict[str, Any]:
        """
        Complete analysis with all features enabled.
        
        Args:
            drug_name: Name or identifier of the drug
            gene_name: Name or identifier of the gene/protein
            
        Returns:
            Comprehensive analysis results dictionary
        """
        query = (
            f"Perform a complete analysis of the interaction between "
            f"drug '{drug_name}' and gene '{gene_name}'. "
            f"Include molecular mechanisms, supporting literature, "
            f"and clinical implications."
        )
        
        return self.analyze_interaction(
            query,
            include_literature=True,
            include_explanation=True,
            generate_report=True
        )

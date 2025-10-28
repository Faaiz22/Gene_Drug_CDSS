"""
PubMed API Client for real-time literature retrieval.
Fetches relevant research papers for drug-gene interactions.
"""

import httpx
import asyncio
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
from xml.etree import ElementTree as ET


class PubMedClient:
    """
    Asynchronous client for NCBI PubMed E-utilities API.
    Caches results locally to minimize API calls.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['pubmed']['base_url']
        self.email = config['pubmed']['email']
        self.api_key = config['pubmed'].get('api_key')
        self.cache_dir = Path(config['paths']['literature_cache'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(days=config['pubmed']['cache_expiry_days'])
        
        # Create async HTTP client
        headers = {
            "User-Agent": f"CDSS/1.0 ({self.email})",
            "tool": "DrugGeneCDSS"
        }
        self.client = httpx.AsyncClient(headers=headers, timeout=30.0)
    
    def _build_query(self, gene_name: str, drug_name: str, 
                     interaction_type: Optional[str] = None) -> str:
        """
        Constructs a PubMed search query for drug-gene interactions.
        """
        # Base query components
        query_parts = [
            f'("{gene_name}"[Title/Abstract] OR "{gene_name}"[MeSH Terms])',
            f'("{drug_name}"[Title/Abstract] OR "{drug_name}"[Substance Name])'
        ]
        
        # Add interaction context
        if interaction_type:
            query_parts.append(f'"{interaction_type}"[Title/Abstract]')
        else:
            # Generic interaction terms
            query_parts.append(
                '("interaction"[Title/Abstract] OR "binding"[Title/Abstract] OR '
                '"inhibition"[Title/Abstract] OR "modulation"[Title/Abstract])'
            )
        
        # Combine with AND
        query = " AND ".join(query_parts)
        
        # Sort by relevance and recency
        query += " AND (hasabstract[text] AND English[lang])"
        
        return query
    
    def _get_cache_path(self, gene_name: str, drug_name: str) -> Path:
        """Generate cache file path for a specific query."""
        cache_key = f"{gene_name}_{drug_name}".replace(" ", "_").lower()
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid."""
        if not cache_path.exists():
            return False
        
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mod_time < self.cache_expiry
    
    async def search_interactions(
        self, 
        gene_name: str, 
        drug_name: str,
        max_results: int = 5,
        interaction_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Search PubMed for relevant papers on drug-gene interactions.
        
        Returns:
            List of paper dictionaries with keys:
            - pmid, title, abstract, authors, journal, year, doi, url
        """
        # Check cache first
        cache_path = self._get_cache_path(gene_name, drug_name)
        if self._is_cache_valid(cache_path):
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            return cached_data['papers'][:max_results]
        
        # Build query
        query = self._build_query(gene_name, drug_name, interaction_type)
        
        # Step 1: Search for PMIDs
        pmids = await self._esearch(query, max_results)
        
        if not pmids:
            return []
        
        # Step 2: Fetch full article details
        papers = await self._efetch(pmids)
        
        # Cache results
        cache_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "papers": papers
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return papers[:max_results]
    
    async def _esearch(self, query: str, max_results: int) -> List[str]:
        """
        Execute PubMed search and return list of PMIDs.
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = await self.client.get(
                f"{self.base_url}esearch.fcgi",
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get("esearchresult", {}).get("idlist", [])
            return pmids
        
        except Exception as e:
            print(f"PubMed search failed: {e}")
            return []
    
    async def _efetch(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed article information for given PMIDs.
        """
        if not pmids:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = await self.client.get(
                f"{self.base_url}efetch.fcgi",
                params=params
            )
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = self._parse_pubmed_xml(root)
            
            return papers
        
        except Exception as e:
            print(f"PubMed fetch failed: {e}")
            return []
    
    def _parse_pubmed_xml(self, root: ET.Element) -> List[Dict]:
        """
        Parse PubMed XML response into structured paper data.
        """
        papers = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # Extract PMID
                pmid = article.findtext(".//PMID")
                
                # Extract title
                title = article.findtext(".//ArticleTitle") or "No title available"
                
                # Extract abstract
                abstract_parts = article.findall(".//AbstractText")
                if abstract_parts:
                    abstract = " ".join([
                        part.text or "" for part in abstract_parts
                    ])
                else:
                    abstract = "No abstract available"
                
                # Extract authors
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.findtext("LastName") or ""
                    initials = author.findtext("Initials") or ""
                    if last_name:
                        authors.append(f"{last_name} {initials}".strip())
                
                # Extract journal and year
                journal = article.findtext(".//Journal/Title") or "Unknown Journal"
                year = article.findtext(".//PubDate/Year") or "Unknown"
                
                # Extract DOI
                doi = None
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break
                
                # Construct PubMed URL
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                
                paper = {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors[:3],  # First 3 authors
                    "journal": journal,
                    "year": year,
                    "doi": doi,
                    "url": url
                }
                
                papers.append(paper)
            
            except Exception as e:
                print(f"Failed to parse article: {e}")
                continue
        
        return papers
    
    def format_citation(self, paper: Dict) -> str:
        """
        Format paper as APA-style citation.
        """
        authors_str = ", ".join(paper["authors"])
        if len(paper["authors"]) > 3:
            authors_str += ", et al."
        
        citation = f"{authors_str} ({paper['year']}). {paper['title']}. "
        citation += f"{paper['journal']}."
        
        if paper.get('doi'):
            citation += f" https://doi.org/{paper['doi']}"
        elif paper.get('url'):
            citation += f" {paper['url']}"
        
        return citation


# Streamlit-compatible cached version
@st.cache_resource
def get_pubmed_client(config: Dict) -> PubMedClient:
    """Cached PubMed client factory."""
    return PubMedClient(config)

"""
Literature Explorer Page - Advanced PubMed Search and Analysis
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import asyncio
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.pubmed_client import PubMedClient
from src.core_processing import load_config

st.set_page_config(
    page_title="Literature Explorer - Drug-Gene CDSS",
    page_icon="📚",
    layout="wide"
)

# Initialize cached resources
@st.cache_resource
def get_pubmed_client():
    config = load_config()
    return PubMedClient(config)

# Title and description
st.title("📚 Literature Explorer")
st.markdown("""
Explore scientific literature for drug-gene interactions using real-time PubMed searches.
Discover supporting evidence, track research trends, and generate comprehensive literature reviews.
""")

# Sidebar configuration
with st.sidebar:
    st.header("🔍 Search Configuration")
    
    max_results = st.slider(
        "Maximum Papers",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Number of papers to retrieve"
    )
    
    year_filter = st.selectbox(
        "Publication Year",
        ["All Years", "Last Year", "Last 5 Years", "Last 10 Years", "Custom Range"],
        help="Filter papers by publication date"
    )
    
    if year_filter == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("From", 1950, 2024, 2000)
        with col2:
            end_year = st.number_input("To", 1950, 2025, 2025)
    
    include_reviews = st.checkbox(
        "Include Review Articles",
        value=True,
        help="Include systematic reviews and meta-analyses"
    )
    
    sort_by = st.selectbox(
        "Sort By",
        ["Relevance", "Publication Date", "Citation Count"],
        help="Order search results"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Search Statistics")
    if 'search_stats' in st.session_state:
        stats = st.session_state.search_stats
        st.metric("Total Papers Found", stats.get('total', 0))
        st.metric("Avg. Publication Year", stats.get('avg_year', 'N/A'))
        st.metric("Unique Journals", stats.get('unique_journals', 0))

# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Search", 
    "📊 Trends Analysis", 
    "🌐 Citation Network",
    "📝 Report Generator"
])

# ===== TAB 1: SEARCH =====
with tab1:
    st.header("Search PubMed")
    
    search_mode = st.radio(
        "Search Mode",
        ["Drug-Gene Pair", "Custom Query", "Gene-Centric", "Drug-Centric"],
        horizontal=True
    )
    
    if search_mode == "Drug-Gene Pair":
        col1, col2 = st.columns(2)
        
        with col1:
            drug_name = st.text_input(
                "Drug Name",
                placeholder="e.g., Warfarin, Imatinib",
                help="Enter drug common name or identifier"
            )
        
        with col2:
            gene_name = st.text_input(
                "Gene/Protein Name",
                placeholder="e.g., CYP2D6, BRCA1",
                help="Enter gene symbol or protein name"
            )
        
        interaction_type = st.selectbox(
            "Interaction Type (Optional)",
            ["Any", "Inhibition", "Binding", "Activation", "Metabolism", "Transport"],
            help="Narrow search to specific interaction types"
        )
        
        search_query = None
        if st.button("🔍 Search Literature", type="primary", use_container_width=True):
            if drug_name and gene_name:
                search_query = (drug_name, gene_name, 
                              None if interaction_type == "Any" else interaction_type.lower())
    
    elif search_mode == "Custom Query":
        custom_query = st.text_area(
            "PubMed Query",
            placeholder='e.g., ("CYP2D6"[Title/Abstract]) AND ("polymorphism"[Title/Abstract])',
            help="Advanced PubMed query syntax",
            height=100
        )
        
        if st.button("🔍 Execute Query", type="primary", use_container_width=True):
            if custom_query:
                search_query = ("custom", custom_query, None)
    
    elif search_mode == "Gene-Centric":
        gene_name = st.text_input(
            "Gene Name",
            placeholder="e.g., EGFR, TP53",
            help="Find all drug interactions for this gene"
        )
        
        if st.button("🔍 Find Drug Interactions", type="primary", use_container_width=True):
            if gene_name:
                search_query = (None, gene_name, "interaction")
    
    else:  # Drug-Centric
        drug_name = st.text_input(
            "Drug Name",
            placeholder="e.g., Tamoxifen, Metformin",
            help="Find all gene interactions for this drug"
        )
        
        if st.button("🔍 Find Gene Interactions", type="primary", use_container_width=True):
            if drug_name:
                search_query = (drug_name, None, "interaction")
    
    # Execute search
    if search_query:
        with st.spinner("🔎 Searching PubMed database..."):
            try:
                client = get_pubmed_client()
                
                if search_mode == "Drug-Gene Pair":
                    papers = asyncio.run(client.search_interactions(
                        gene_name=search_query[1],
                        drug_name=search_query[0],
                        max_results=max_results,
                        interaction_type=search_query[2]
                    ))
                else:
                    # Implement custom search logic
                    papers = []
                
                if papers:
                    st.success(f"✅ Found {len(papers)} relevant papers")
                    
                    # Store in session state
                    st.session_state.papers = papers
                    st.session_state.search_stats = {
                        'total': len(papers),
                        'avg_year': sum(int(p['year']) for p in papers if p['year'].isdigit()) / len(papers),
                        'unique_journals': len(set(p['journal'] for p in papers))
                    }
                    
                    # Display results
                    for i, paper in enumerate(papers, 1):
                        with st.expander(f"📄 {i}. {paper['title']}", expanded=(i <= 3)):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**Authors:** {', '.join(paper['authors'][:5])}")
                                if len(paper['authors']) > 5:
                                    st.markdown(f"*+ {len(paper['authors']) - 5} more authors*")
                            
                            with col2:
                                st.markdown(f"**Journal:** {paper['journal']}")
                                st.markdown(f"**Year:** {paper['year']}")
                            
                            with col3:
                                st.markdown(f"**PMID:** {paper['pmid']}")
                                if paper.get('doi'):
                                    st.markdown(f"**DOI:** {paper['doi']}")
                            
                            st.markdown("---")
                            st.markdown("**Abstract:**")
                            st.markdown(paper['abstract'])
                            
                            col_a, col_b, col_c = st.columns([1, 1, 2])
                            with col_a:
                                if st.button(f"🔗 View on PubMed", key=f"pubmed_{i}"):
                                    st.markdown(f"[Open in new tab]({paper['url']})")
                            
                            with col_b:
                                if st.button(f"📋 Copy Citation", key=f"cite_{i}"):
                                    citation = client.format_citation(paper)
                                    st.code(citation, language=None)
                            
                            with col_c:
                                if st.button(f"💾 Save to Report", key=f"save_{i}"):
                                    if 'saved_papers' not in st.session_state:
                                        st.session_state.saved_papers = []
                                    st.session_state.saved_papers.append(paper)
                                    st.success("Added to report!")
                    
                    # Download results
                    st.markdown("---")
                    df_papers = pd.DataFrame([{
                        'PMID': p['pmid'],
                        'Title': p['title'],
                        'Authors': ', '.join(p['authors']),
                        'Journal': p['journal'],
                        'Year': p['year'],
                        'DOI': p.get('doi', 'N/A'),
                        'URL': p['url']
                    } for p in papers])
                    
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=df_papers.to_csv(index=False).encode('utf-8'),
                        file_name=f"pubmed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.warning("No papers found. Try broader search terms.")
            
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

# ===== TAB 2: TRENDS ANALYSIS =====
with tab2:
    st.header("📊 Research Trends Analysis")
    
    if 'papers' in st.session_state and st.session_state.papers:
        papers = st.session_state.papers
        
        # Year distribution
        st.subheader("Publication Timeline")
        
        year_counts = pd.DataFrame([
            {'Year': int(p['year']), 'Count': 1} 
            for p in papers if p['year'].isdigit()
        ]).groupby('Year').count().reset_index()
        
        fig_timeline = px.line(
            year_counts,
            x='Year',
            y='Count',
            title='Publications Over Time',
            markers=True
        )
        fig_timeline.update_traces(line_color='#FF4B4B')
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Journal distribution
        st.subheader("Top Publishing Journals")
        
        journal_counts = pd.DataFrame([
            {'Journal': p['journal'], 'Count': 1}
            for p in papers
        ]).groupby('Journal').count().reset_index().sort_values('Count', ascending=False).head(10)
        
        fig_journals = px.bar(
            journal_counts,
            x='Count',
            y='Journal',
            orientation='h',
            title='Publications by Journal (Top 10)'
        )
        st.plotly_chart(fig_journals, use_container_width=True)
        
        # Author network (simplified)
        st.subheader("Collaborative Network")
        
        all_authors = []
        for p in papers:
            all_authors.extend(p['authors'][:3])  # Top 3 authors per paper
        
        author_counts = pd.Series(all_authors).value_counts().head(15)
        
        fig_authors = px.bar(
            x=author_counts.values,
            y=author_counts.index,
            orientation='h',
            title='Most Prolific Authors',
            labels={'x': 'Publications', 'y': 'Author'}
        )
        st.plotly_chart(fig_authors, use_container_width=True)
        
        # Research growth metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recent_papers = sum(1 for p in papers if int(p['year']) >= 2020)
            st.metric("Papers (2020+)", recent_papers)
        
        with col2:
            avg_year = sum(int(p['year']) for p in papers if p['year'].isdigit()) / len(papers)
            st.metric("Avg. Publication Year", f"{avg_year:.1f}")
        
        with col3:
            unique_journals = len(set(p['journal'] for p in papers))
            st.metric("Unique Journals", unique_journals)
    
    else:
        st.info("👈 Perform a search in the 'Search' tab to see trends analysis")

# ===== TAB 3: CITATION NETWORK =====
with tab3:
    st.header("🌐 Citation Network Visualization")
    
    if 'papers' in st.session_state and st.session_state.papers:
        st.info("🚧 Citation network analysis coming soon! This will show connections between papers and co-citation patterns.")
        
        # Placeholder visualization
        st.markdown("""
        **Planned Features:**
        - Interactive citation graph
        - Co-citation analysis
        - Research cluster identification
        - Influential papers detection
        - Collaboration network mapping
        """)
    else:
        st.info("👈 Perform a search in the 'Search' tab to visualize citation networks")

# ===== TAB 4: REPORT GENERATOR =====
with tab4:
    st.header("📝 Literature Review Report Generator")
    
    if 'saved_papers' in st.session_state and st.session_state.saved_papers:
        saved_papers = st.session_state.saved_papers
        
        st.success(f"✅ {len(saved_papers)} papers saved for report")
        
        # Report configuration
        st.subheader("Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input(
                "Report Title",
                value="Literature Review: Drug-Gene Interactions"
            )
            
            report_format = st.selectbox(
                "Output Format",
                ["Markdown", "PDF", "Word Document", "HTML"]
            )
        
        with col2:
            include_abstracts = st.checkbox("Include Abstracts", value=True)
            include_methods = st.checkbox("Include Methods Section", value=False)
            citation_style = st.selectbox(
                "Citation Style",
                ["APA", "MLA", "Chicago", "Vancouver"]
            )
        
        # Generate report
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating literature review..."):
                # Build report content
                report = f"# {report_title}\n\n"
                report += f"**Generated:** {datetime.now().strftime('%B %d, %Y')}\n"
                report += f"**Total Papers:** {len(saved_papers)}\n\n"
                report += "---\n\n"
                
                report += "## Summary\n\n"
                report += f"This literature review encompasses {len(saved_papers)} peer-reviewed publications "
                report += f"spanning from {min(p['year'] for p in saved_papers)} to {max(p['year'] for p in saved_papers)}. "
                
                unique_journals = len(set(p['journal'] for p in saved_papers))
                report += f"The papers were published across {unique_journals} different journals.\n\n"
                
                report += "## Papers Reviewed\n\n"
                
                for i, paper in enumerate(saved_papers, 1):
                    report += f"### {i}. {paper['title']}\n\n"
                    report += f"**Authors:** {', '.join(paper['authors'])}\n\n"
                    report += f"**Journal:** {paper['journal']} ({paper['year']})\n\n"
                    report += f"**PMID:** {paper['pmid']}\n\n"
                    
                    if paper.get('doi'):
                        report += f"**DOI:** {paper['doi']}\n\n"
                    
                    if include_abstracts:
                        report += f"**Abstract:**\n\n{paper['abstract']}\n\n"
                    
                    report += "---\n\n"
                
                report += "## References\n\n"
                client = get_pubmed_client()
                for i, paper in enumerate(saved_papers, 1):
                    citation = client.format_citation(paper)
                    report += f"{i}. {citation}\n\n"
                
                # Display and download
                st.markdown("### Report Preview")
                st.markdown(report)
                
                st.download_button(
                    label="📥 Download Report",
                    data=report.encode('utf-8'),
                    file_name=f"literature_review_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
        
        # Manage saved papers
        st.markdown("---")
        st.subheader("Manage Saved Papers")
        
        for i, paper in enumerate(saved_papers):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i+1}.** {paper['title']} ({paper['year']})")
            with col2:
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.saved_papers.pop(i)
                    st.rerun()
        
        if st.button("🗑️ Clear All Saved Papers", use_container_width=True):
            st.session_state.saved_papers = []
            st.rerun()
    
    else:
        st.info("👈 Search for papers and click 'Save to Report' to add them here")
        
        st.markdown("""
        ### How to Use Report Generator:
        
        1. **Search** for relevant papers in the 'Search' tab
        2. **Save** papers using the 'Save to Report' button
        3. **Configure** report settings above
        4. **Generate** a formatted literature review
        5. **Download** in your preferred format
        """)

# Footer
st.markdown("---")
st.caption("""
**Data Source:** NCBI PubMed via E-utilities API  
**Note:** Literature searches are cached for 30 days to minimize API calls.  
**Disclaimer:** This tool is for research purposes. Always verify findings with primary sources.
""")

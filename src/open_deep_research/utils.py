import os
import asyncio
import json
import datetime
import re
import requests
import random 
import concurrent
import hashlib
import aiohttp
import httpx
import time
from typing import List, Optional, Dict, Any, Union, Literal, Annotated, cast
from urllib.parse import unquote
from collections import defaultdict
import itertools
import platform

from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient as AsyncAzureAISearchClient
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup
from markdownify import markdownify
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI

from open_deep_research.configuration import Configuration
from open_deep_research.state import Section
from open_deep_research.prompts import SUMMARIZATION_PROMPT

import structlog
logger = structlog.get_logger(__name__)

# Import smart content fetching functionality from playwright_utils
try:
    from open_deep_research.playwright_utils import smart_content_fetch, fetch_with_playwright
    PLAYWRIGHT_UTILS_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_UTILS_AVAILABLE = False
    logger.warning("playwright_utils module not available, some advanced content fetching features will be limited")


def get_config_value(value, default_value=None):
    """
    Helper function to handle string, dict, and enum cases of configuration values
    
    Args:
        value: The configuration value to process
        default_value: Default value to return if value is None or invalid
    
    Returns:
        Processed configuration value or default_value
    """
    if value is None:
        return default_value
    
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        try:
            return value.value
        except AttributeError:
            return default_value if default_value is not None else value

def get_config_value_from_runnable(config: RunnableConfig, key: str, default_value=None):
    """
    Helper function to get configuration value from RunnableConfig
    
    Args:
        config: RunnableConfig object or dictionary
        key: Configuration key name
        default_value: Default value
    
    Returns:
        Configuration value or default value
    """
    if not config:
        return default_value
    
    # If config is a dictionary type (RunnableConfig)
    if isinstance(config, dict) and 'configurable' in config:
        return config['configurable'].get(key, default_value)
    
    # If config has configurable attribute
    if hasattr(config, 'configurable') and config.configurable:
        return config.configurable.get(key, default_value)
    
    # Fallback to direct access (backward compatibility)
    return getattr(config, key, default_value) if hasattr(config, key) else default_value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": ["max_results", "topic"],
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
        "googlesearch": ["max_results", "include_raw_content"],
        "geminigooglesearch": ["max_results", "include_raw_content"],
        "azureaisearch": ["max_results", "topic", "include_raw_content"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(
    search_response,
    max_tokens_per_source=20000,
    include_raw_content=True,
    deduplication_strategy: Literal["keep_first", "keep_last"] = "keep_first"
):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
        deduplication_strategy: Whether to keep the first or last search result for each unique URL
    Returns:
        str: Formatted string with deduplicated sources
    """
    logger.info(f"deduplicate_and_format_sources - search_response: {search_response}")
     # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])

    # Deduplicate by URL
    if deduplication_strategy == "keep_first":
        unique_sources = {}
        for source in sources_list:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source
    elif deduplication_strategy == "keep_last":
        unique_sources = {source['url']: source for source in sources_list}
    else:
        raise ValueError(f"Invalid deduplication strategy: {deduplication_strategy}")

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                logger.warning(f"Source content missing raw_content - source_url: {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

@traceable
async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True):
    """
    Performs concurrent web searches with the Tavily API

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return
        topic (Literal["general", "news", "finance"]): Topic to filter results by
        include_raw_content (bool): Whether to include raw content in the results

    Returns:
            List[dict]: List of search responses from Tavily API:
                {
                    'query': str,
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """
    logger.info(f"Starting Tavily search - queries_count: {len(search_queries)}, max_results: {max_results}, topic: {topic}, include_raw_content: {include_raw_content}, queries: {search_queries[:3]}")
    
    try:
        tavily_async_client = AsyncTavilyClient()
        search_tasks = []
        for query in search_queries:
                search_tasks.append(
                    tavily_async_client.search(
                        query,
                        max_results=max_results,
                        include_raw_content=include_raw_content,
                        topic=topic
                    )
                )

        logger.debug(f"Preparing to execute concurrent search - tasks_count: {len(search_tasks)}")
        
        # Execute all searches concurrently
        search_docs = await asyncio.gather(*search_tasks)
        
        total_results = sum(len(doc.get('results', [])) for doc in search_docs)
        logger.info(f"Tavily search completed - queries_processed: {len(search_queries)}, total_results: {total_results}, avg_results_per_query: {total_results / len(search_queries) if search_queries else 0}")
        
        return search_docs
        
    except Exception as e:
        logger.error(f"Tavily search failed - error: {str(e)}, error_type: {type(e).__name__}, queries_count: {len(search_queries)}")
        raise

@traceable
async def azureaisearch_search_async(search_queries: list[str], max_results: int = 5, topic: str = "general", include_raw_content: bool = True) -> list[dict]:
    """
    Performs concurrent web searches using the Azure AI Search API.

    Args:
        search_queries (List[str]): list of search queries to process
        max_results (int): maximum number of results to return for each query
        topic (str): semantic topic filter for the search.
        include_raw_content (bool)

    Returns:
        List[dict]: list of search responses from Azure AI Search API, one per query.
    """
    # configure and create the Azure Search client
    # ensure all environment variables are set
    if not all(var in os.environ for var in ["AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_INDEX_NAME", "AZURE_AI_SEARCH_API_KEY"]):
        raise ValueError("Missing required environment variables for Azure Search API which are: AZURE_AI_SEARCH_ENDPOINT, AZURE_AI_SEARCH_INDEX_NAME, AZURE_AI_SEARCH_API_KEY")
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))

    reranker_key = '@search.reranker_score'

    async with AsyncAzureAISearchClient(endpoint, index_name, credential) as client:
        async def do_search(query: str) -> dict:
            # search query 
            paged = await client.search(
                search_text=query,
                vector_queries=[{
                    "fields": "vector",
                    "kind": "text",
                    "text": query,
                    "exhaustive": True
                }],
                semantic_configuration_name="fraunhofer-rag-semantic-config",
                query_type="semantic",
                select=["url", "title", "chunk", "creationTime", "lastModifiedTime"],
                top=max_results,
            )
            # async iterator to get all results
            items = [doc async for doc in paged]
            # Umwandlung in einfaches Dict-Format
            results = [
                {
                    "title": doc.get("title"),
                    "url": doc.get("url"),
                    "content": doc.get("chunk"),
                    "score": doc.get(reranker_key),
                    "raw_content": doc.get("chunk") if include_raw_content else None
                }
                for doc in items
            ]
            return {"query": query, "results": results}

        # parallelize the search queries
        tasks = [do_search(q) for q in search_queries]
        return await asyncio.gather(*tasks)


@traceable
def perplexity_search(search_queries):
    """Search the web using the Perplexity API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
  
    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    logger.info(f"Starting Perplexity search - queries_count: {len(search_queries)}, queries: {search_queries[:3]}")
    
    # Check API key
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        logger.error("PERPLEXITY_API_KEY environment variable not found")
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    search_docs = []
    for i, query in enumerate(search_queries):
        logger.debug(f"Processing Perplexity query - query_index: {i+1}, query: {query}")
        
        try:
            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "Search the web and provide factual information with sources."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            }
            
            logger.debug(f"Sending Perplexity API request - query: {query}")
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Parse the response
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            citations = data.get("citations", ["https://perplexity.ai"])
            
            logger.debug(f"Perplexity API response parsed successfully - query: {query}, content_length: {len(content)}, citations_count: {len(citations)}")
            
            # Create results list for this query
            results = []
            
            # First citation gets the full content
            results.append({
                "title": f"Perplexity Search, Source 1",
                "url": citations[0],
                "content": content,
                "raw_content": content,
                "score": 1.0  # Adding score to match Tavily format
            })
            
            # Add additional citations without duplicating content
            for i, citation in enumerate(citations[1:], start=2):
                results.append({
                    "title": f"Perplexity Search, Source {i}",
                    "url": citation,
                    "content": "See primary source for full content",
                    "raw_content": None,
                    "score": 0.5  # Lower score for secondary sources
                })
            
            # Format response to match Tavily structure
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": results
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API request failed - query: {query}, error: {str(e)}, status_code: {getattr(e.response, 'status_code', None)}")
            # Add empty result to maintain structure consistency
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })
        except Exception as e:
            logger.error(f"Perplexity search processing exception - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })
    
    logger.info(f"Perplexity search completed - queries_processed: {len(search_queries)}, successful_queries: {len([doc for doc in search_docs if not doc.get('error')])}, total_results: {sum(len(doc.get('results', [])) for doc in search_docs)}")
    logger.info(f"Perplexity search results - {search_docs}")
    
    return search_docs

@traceable
async def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5, 
                     include_domains: Optional[List[str]] = None, 
                     exclude_domains: Optional[List[str]] = None,
                     subpages: Optional[int] = None):
    """Search the web using the Exa API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        max_characters (int, optional): Maximum number of characters to retrieve for each result's raw content.
                                       If None, the text parameter will be set to True instead of an object.
        num_results (int): Number of search results per query. Defaults to 5.
        include_domains (List[str], optional): List of domains to include in search results. 
            When specified, only results from these domains will be returned.
        exclude_domains (List[str], optional): List of domains to exclude from search results.
            Cannot be used together with include_domains.
        subpages (int, optional): Number of subpages to retrieve per result. If None, subpages are not retrieved.
        
    Returns:
        List[dict]: List of search responses from Exa API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    # Check that include_domains and exclude_domains are not both specified
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")
    
    # Initialize Exa client (API key should be configured in your .env file)
    exa = Exa(api_key = f"{os.getenv('EXA_API_KEY')}")
    
    # Define the function to process a single query
    async def process_query(query):
        # Use run_in_executor to make the synchronous exa call in a non-blocking way
        loop = asyncio.get_event_loop()
        
        # Define the function for the executor with all parameters
        def exa_search_fn():
            # Build parameters dictionary
            kwargs = {
                # Set text to True if max_characters is None, otherwise use an object with max_characters
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,  # This is an amazing feature by EXA. It provides an AI generated summary of the content based on the query
                "num_results": num_results
            }
            
            # Add optional parameters only if they are provided
            if subpages is not None:
                kwargs["subpages"] = subpages
                
            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
                
            return exa.search_and_contents(query, **kwargs)
        
        response = await loop.run_in_executor(None, exa_search_fn)
        
        # Format the response to match the expected output structure
        formatted_results = []
        seen_urls = set()  # Track URLs to avoid duplicates
        
        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default
        
        # Access the results from the SearchResponse object
        results_list = get_value(response, 'results', [])
        
        # First process all main results
        for result in results_list:
            # Get the score with a default of 0.0 if it's None or not present
            score = get_value(result, 'score', 0.0)
            
            # Combine summary and text for content if both are available
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')
            
            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content
            
            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')
            
            # Skip if we've seen this URL before (removes duplicate entries)
            if url in seen_urls:
                continue
                
            seen_urls.add(url)
            
            # Main result entry
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content
            }
            
            # Add the main result to the formatted results
            formatted_results.append(result_entry)
        
        # Now process subpages only if the subpages parameter was provided
        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = get_value(subpage, 'score', 0.0)
                    
                    # Combine summary and text for subpage content
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')
                    
                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary
                    
                    subpage_url = get_value(subpage, 'url', '')
                    
                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue
                        
                    seen_urls.add(subpage_url)
                    
                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text
                    })
        
        # Collect images if available (only from main results to avoid duplication)
        images = []
        for result in results_list:
            image = get_value(result, 'image')
            if image and image not in images:  # Avoid duplicate images
                images.append(image)
                
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results
        }
    
    # Process all queries sequentially with delay to respect rate limit
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (0.25s = 4 requests per second, well within the 5/s limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(0.25)
            
            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            logger.error(f"Exa query processing failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
            # Add a placeholder result for failed queries to maintain index alignment
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })
            
            # Add additional delay if we hit a rate limit error
            if "429" in str(e):
                logger.warning("Exa API rate limit, adding additional delay")
                await asyncio.sleep(1.0)  # Add a longer delay if we hit a rate limit
    
    return search_docs

@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """
    Performs concurrent searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL (Entry ID) of the paper
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str|None  # Full paper content if available
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # Create retriever for each query
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )
            
            # Run the synchronous retriever in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata
                
                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get('entry_id', '')
                
                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if 'Summary' in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if 'primary_category' in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines 
                content = "\n".join(content_parts)
                
                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # Using entry_id as the URL
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None
                }
                results.append(result)
                
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions gracefully
            logger.error(f"arXiv single query processing failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (3 seconds per ArXiv's rate limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(3.0)
            
            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            logger.error(f"arXiv query main loop processing failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # Add additional delay if we hit a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.warning("arXiv API rate limit, adding additional delay")
                await asyncio.sleep(5.0)  # Add a longer delay if we hit a rate limit
    
    return search_docs

@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # print(f"Processing PubMed query: '{query}'")
            
            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )
            
            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()
            
            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            logger.debug(f"PubMed query returned results - query: {query}, results_count: {len(docs)}")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []
                
                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")
                
                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")
                
                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")
                
                # Generate PubMed URL from the article UID
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # Join all content parts with newlines
                content = "\n".join(content_parts)
                
                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            logger.error(f"PubMed single query processing failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
            import traceback
            logger.debug(f"PubMed error details - traceback: {traceback.format_exc()}")
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process all queries with a reasonable delay between them
    search_docs = []
    
    # Start with a small delay that increases if we encounter rate limiting
    delay = 1.0  # Start with a more conservative delay
    
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests
            if i > 0:  # Don't delay the first request
                # print(f"Waiting {delay} seconds before next query...")
                await asyncio.sleep(delay)
            
            result = await process_single_query(query)
            search_docs.append(result)
            
            # If query was successful with results, we can slightly reduce delay (but not below minimum)
            if result.get('results') and len(result['results']) > 0:
                delay = max(0.5, delay * 0.9)  # Don't go below 0.5 seconds
            
        except Exception as e:
            # Handle exceptions gracefully
            logger.error(f"PubMed main loop processing failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
            
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # If we hit an exception, increase delay for next query
            delay = min(5.0, delay * 1.5)  # Don't exceed 5 seconds
    
    return search_docs

@traceable
async def linkup_search(search_queries, depth: Optional[str] = "standard"):
    """
    Performs concurrent web searches using the Linkup API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        depth (str, optional): "standard" (default)  or "deep". More details here https://docs.linkup.so/pages/documentation/get-started/concepts

    Returns:
        List[dict]: List of search responses from Linkup API, one per query. Each response has format:
            {
                'results': [            # List of search results
                    {
                        'title': str,   # Title of the search result
                        'url': str,     # URL of the result
                        'content': str, # Summary/snippet of content
                    },
                    ...
                ]
            }
    """
    client = LinkupClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
                client.async_search(
                    query,
                    depth,
                    output_type="searchResults",
                )
            )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {
                "results": [
                    {"title": result.name, "url": result.url, "content": result.content}
                    for result in response.results
                ],
            }
        )

    return search_results

@traceable
async def google_custom_search_async(search_queries: Union[str, List[str]], max_results: int = 5, include_raw_content: bool = True):
    """
    Performs concurrent web searches using Google.
    Uses Google Custom Search API if environment variables are set, otherwise falls back to web scraping.

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to fetch full page content

    Returns:
        List[dict]: List of search responses from Google, one per query
    """
    
    # Handle case where search_queries is a single string
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    
    logger.info(f"Starting Google search - queries_count: {len(search_queries)}, max_results: {max_results}, include_raw_content: {include_raw_content}, queries: {search_queries[:3]}")  # Only log first 3 queries

    # Check for API credentials from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)
    # use_api = False
    
    logger.info(f"Google search mode determined - use_api: {use_api}, has_api_key: {bool(api_key)}, has_cx: {bool(cx)}")
    
    # Define user agent generator
    def get_useragent():
        """Generates a random user agent string."""
        # 使用更真实的 User-Agent，模拟常见浏览器
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
        ]
        return random.choice(user_agents)
    
    # Create executor for running synchronous operations
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    # Use a semaphore to limit concurrent requests - 降低并发数
    semaphore = asyncio.Semaphore(5 if use_api else 1)
    
    async def search_single_query(query):
        async with semaphore:
            try:
                results = []
                
                # API-based search
                if use_api:
                    # The API returns up to 10 results per request
                    for start_index in range(1, max_results + 1, 10):
                        # Calculate how many results to request in this batch
                        num = min(10, max_results - (start_index - 1))
                        
                        # Make request to Google Custom Search API
                        params = {
                            'q': query,
                            'key': api_key,
                            'cx': cx,
                            'start': start_index,
                            'num': num
                        }
                        logger.debug(f"Requesting Google API - query: {query}, num_results: {num}, start_index: {start_index}")

                        connector = aiohttp.TCPConnector(ssl=False)
                        timeout = aiohttp.ClientTimeout(total=300)
                        headers = {
                            "User-Agent": get_useragent(),
                            "Accept": "*/*"
                        }
                        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
                            async with session.get(url='https://www.googleapis.com/customsearch/v1', params=params) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    logger.error(f"Google API request failed - status_code: {response.status}, error_text: {error_text[:500]}")  # Limit error text length
                                    break
                                    
                                data = await response.json()
                                
                                # Process search results
                                for item in data.get('items', []):
                                    result = {
                                        "title": item.get('title', ''),
                                        "url": item.get('link', ''),
                                        "content": item.get('snippet', ''),
                                        "score": None,
                                        "raw_content": item.get('snippet', '')
                                    }
                                    results.append(result)
                        
                        # Respect API quota with a small delay
                        await asyncio.sleep(0.2)
                        
                        # If we didn't get a full page of results, no need to request more
                        if not data.get('items') or len(data.get('items', [])) < num:
                            break
                
                # Web scraping based search
                else:
                    # Add delay between requests - 增加延迟时间
                    await asyncio.sleep(2.0 + random.random() * 3.0)  # 2-5秒随机延迟
                    logger.debug(f"Starting Google web scraping - query: {query}")

                    # Define scraping function
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []
                            
                            # 添加重试机制
                            max_retries = 3
                            retry_count = 0
                            
                            while fetched_results < max_results and retry_count < max_retries:
                                try:
                                    # 创建 session 来保持 cookies
                                    session = requests.Session()
                                    
                                    # 设置更真实的请求头
                                    session.headers.update({
                                        "User-Agent": get_useragent(),
                                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                                        "Accept-Language": "en-US,en;q=0.9",
                                        "Accept-Encoding": "gzip, deflate, br",
                                        "DNT": "1",
                                        "Connection": "keep-alive",
                                        "Upgrade-Insecure-Requests": "1",
                                        "Sec-Fetch-Dest": "document",
                                        "Sec-Fetch-Mode": "navigate",
                                        "Sec-Fetch-Site": "none",
                                        "Sec-Fetch-User": "?1",
                                        "Cache-Control": "max-age=0",
                                        "sec-ch-ua": '"Chromium";v="131", "Not A(Brand";v="99", "Google Chrome";v="131"',
                                        "sec-ch-ua-mobile": "?0",
                                        "sec-ch-ua-platform": '"Windows"'
                                    })
                                    
                                    # 设置 cookies
                                    session.cookies.update({
                                        'CONSENT': 'YES+cb',
                                        'SOCS': 'CAISHAgBEhJnd3NfMjAyMzA4MTAtMF9SQzIaAmVuIAEaBgiAyqymBg',
                                    })
                                    
                                    # 首先访问 Google 主页来获取 cookies
                                    try:
                                        homepage_resp = session.get(
                                            "https://www.google.com",
                                            timeout=10,
                                            proxies={
                                                'http': os.environ.get('HTTP_PROXY'),
                                                'https': os.environ.get('HTTPS_PROXY')
                                            } if os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY') else None
                                        )
                                        time.sleep(0.5)  # 短暂延迟
                                    except:
                                        pass  # 忽略主页访问错误
                                    
                                    # Send request to Google
                                    resp = session.get(
                                        url="https://www.google.com/search",
                                        params={
                                            "q": query,
                                            "num": max_results + 2,
                                            "hl": lang,
                                            "start": start,
                                            "safe": safe,
                                            "lr": lang,  # 添加语言限制
                                            "gl": "us",  # 添加地理位置
                                        },
                                        timeout=30,
                                        # 添加代理支持（如果配置了）
                                        proxies={
                                            'http': os.environ.get('HTTP_PROXY'),
                                            'https': os.environ.get('HTTPS_PROXY')
                                        } if os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY') else None,
                                        allow_redirects=True
                                    )
                                    
                                    # 检查是否被重定向到验证页面
                                    if resp.status_code == 429 or "sorry/index" in resp.url:
                                        logger.warning(f"Google rate limit detected, retry {retry_count + 1}/{max_retries}")
                                        retry_count += 1
                                        # 指数退避策略
                                        time.sleep(5 * (2 ** retry_count) + random.random() * 5)
                                        continue
                                        
                                    resp.raise_for_status()
                                    
                                    # Parse results
                                    soup = BeautifulSoup(resp.text, "html.parser")
                                    
                                    # 更全面的选择器列表
                                    result_selectors = [
                                        "div.g",  # 经典选择器
                                        "div.Gx5Zad",  # 移动版
                                        "div.MjjYud",  # 新版桌面
                                        "div.kvH3mc",  # 另一种格式
                                        "div.ezO2md",  # 旧选择器
                                        "div[data-hveid]",  # 基于属性
                                        "div.jfp3ef",  # 其他可能
                                        "div.N54PNb",  # 其他可能
                                        "div.hlcw0c",  # 其他可能
                                        "div.kCrYT",  # 移动版结果
                                        "div.BNeawe",  # 简化版结果
                                    ]
                                    
                                    result_block = []
                                    for selector in result_selectors:
                                        result_block = soup.select(selector)
                                        if result_block:
                                            logger.debug(f"Found results using selector: {selector}, count: {len(result_block)}")
                                            break
                                    
                                    # 如果还是没找到，尝试查找所有包含链接的 div
                                    if not result_block:
                                        all_divs = soup.find_all("div")
                                        result_block = []
                                        for div in all_divs:
                                            # 查找包含 /url?q= 链接的 div
                                            if div.find("a", href=lambda x: x and x.startswith("/url?q=")):
                                                result_block.append(div)
                                    
                                    new_results = 0
                                    
                                    for result in result_block:
                                        # 查找链接 - 更灵活的方式
                                        link_tag = result.find("a", href=lambda x: x and (x.startswith("/url?q=") or x.startswith("http")))
                                        
                                        if not link_tag:
                                            continue
                                        
                                        # 尝试多种标题选择器
                                        title_tag = None
                                        title_selectors = [
                                            ("h3", {}),
                                            ("h3", {"class": "LC20lb"}),
                                            ("h3", {"class": "DKV0Md"}),
                                            ("div", {"class": "BNeawe"}),
                                            ("span", {"class": "CVA68e"}),
                                            ("div", {"role": "heading"}),
                                            ("span", {"dir": "auto"}),
                                        ]
                                        
                                        for tag_name, attrs in title_selectors:
                                            title_tag = result.find(tag_name, attrs)
                                            if title_tag and title_tag.text.strip():
                                                break
                                        
                                        # 如果还是没有找到标题，尝试从链接本身获取
                                        if not title_tag:
                                            title_tag = link_tag
                                        
                                        # 尝试多种描述选择器    
                                        description_tag = None
                                        desc_selectors = [
                                            ("div", {"class": "VwiC3b"}),
                                            ("span", {"class": "FrIlee"}),
                                            ("span", {"class": "st"}),
                                            ("div", {"class": "BNeawe", "style": True}),
                                            ("div", {"data-content-feature": "1"}),
                                            ("div", {"style": "-webkit-line-clamp:2"}),
                                            ("span", {"class": "aCOpRe"}),
                                        ]
                                        
                                        for tag_name, attrs in desc_selectors:
                                            description_tag = result.find(tag_name, attrs)
                                            if description_tag and description_tag.text.strip():
                                                break
                                        
                                        # 如果没有找到描述，尝试获取结果块的所有文本
                                        if not description_tag:
                                            # 获取结果中的所有文本，排除标题
                                            all_text = result.get_text(separator=" ", strip=True)
                                            if title_tag:
                                                title_text = title_tag.get_text(strip=True)
                                                desc_text = all_text.replace(title_text, "").strip()
                                                if desc_text:
                                                    # 创建一个虚拟的描述标签
                                                    from bs4 import NavigableString
                                                    description_tag = NavigableString(desc_text[:200])
                                        
                                        if link_tag and title_tag:
                                            # 提取链接
                                            href = link_tag.get("href", "")
                                            if href.startswith("/url?q="):
                                                link = unquote(href.split("?q=")[1].split("&")[0])
                                            elif href.startswith("http"):
                                                link = href
                                            else:
                                                continue
                                            
                                            # 过滤 Google 内部链接
                                            if link.startswith("http") and "google.com" not in link:
                                                if link in fetched_links:
                                                    continue
                                                
                                                fetched_links.add(link)
                                                title = title_tag.text.strip() if hasattr(title_tag, 'text') else str(title_tag).strip()
                                                description = description_tag.text.strip() if description_tag and hasattr(description_tag, 'text') else str(description_tag).strip() if description_tag else ""
                                                
                                                # Store result in the same format as the API results
                                                search_results.append({
                                                    "title": title,
                                                    "url": link,
                                                    "content": description,
                                                    "score": None,
                                                    "raw_content": description
                                                })
                                                
                                                fetched_results += 1
                                                new_results += 1
                                                
                                                if fetched_results >= max_results:
                                                    break
                                    
                                    if new_results == 0:
                                        break
                                        
                                    start += 10
                                    time.sleep(2.0 + random.random() * 2.0)  # 增加分页延迟到 2-4 秒
                                    
                                    # 重置重试计数
                                    retry_count = 0
                                    
                                except requests.exceptions.RequestException as e:
                                    logger.error(f"Google search request failed, retry {retry_count}/{max_retries}: {str(e)}")
                                    if retry_count < max_retries - 1:
                                        retry_count += 1
                                        logger.warning(f"Google search request failed, retry {retry_count}/{max_retries}: {str(e)}")
                                        time.sleep(5 * (2 ** retry_count))
                                        continue
                                    else:
                                        break
                            
                            return search_results
                                
                        except Exception as e:
                            logger.error(f"Google web scraping failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
                            return []
                    
                    # Execute search in thread pool
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(
                        executor, 
                        lambda: google_search(query, max_results)
                    )
                    
                    # Process the results
                    results = search_results
                
                logger.info(f"Google search completed - results: {results}")
                
                # 如果爬虫模式没有获取到结果，尝试使用 DuckDuckGo 作为备用
                if len(results) == 0:
                    logger.warning(f"Google web scraping returned no results, falling back to DuckDuckGo - query: {query}")
                    try:
                        # 使用 DuckDuckGo 搜索
                        with DDGS() as ddgs:
                            ddg_results = list(ddgs.text(query, max_results=max_results))
                            
                            # 转换 DuckDuckGo 结果格式
                            for i, ddg_result in enumerate(ddg_results):
                                results.append({
                                    "title": ddg_result.get('title', ''),
                                    "url": ddg_result.get('href', ''),
                                    "content": ddg_result.get('body', ''),
                                    "score": 1.0 - (i * 0.1),
                                    "raw_content": ddg_result.get('body', ''),
                                    "source": "duckduckgo_fallback"
                                })
                            
                            logger.info(f"DuckDuckGo fallback search completed - results_count: {len(results)}, query: {query}")
                            
                            # 添加随机暂停，避免被检测为机器人
                            pause_duration = random.uniform(1, 5)
                            logger.debug(f"Random pause for {pause_duration:.2f} seconds to avoid rate limiting")
                            await asyncio.sleep(pause_duration)
                    
                    except Exception as e:
                        logger.error(f"DuckDuckGo fallback search failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
                
                # Use smart content fetching to optimize data retrieval process
                if include_raw_content and results and PLAYWRIGHT_UTILS_AVAILABLE:
                    try:
                        logger.info(f"Starting smart content fetching - results_count: {len(results)}, query: {query}")
                        
                        # Extract all URLs that need content fetching
                        urls_to_fetch = [result['url'] for result in results if result.get('url')]
                        
                        if urls_to_fetch:
                            logger.debug(f"[Playwright]Preparing smart content fetching - urls_count: {len(urls_to_fetch)}, urls: {urls_to_fetch[:5]}")
                            
                            # Use smart content fetching
                            # Specify domains that need special handling (social media, dynamic websites, etc.)
                            special_domains = ['x.com', 'twitter.com', 'linkedin.com', 'facebook.com', 'instagram.com']
                            
                            # Fetch content
                            fetch_results = await smart_content_fetch(
                                urls=urls_to_fetch,
                                use_playwright_for=special_domains,
                                fallback_to_playwright=True  # Fallback to Playwright on failure
                            )
                            logger.info(f"[Playwright]Smart content fetching completed - fetch_results: {fetch_results}")
                            
                            # Map fetched content back to original results
                            url_to_content = {
                                fetch_result['original_url']: fetch_result 
                                for fetch_result in fetch_results
                            }
                            logger.info(f"[Playwright]url_to_content: {url_to_content}")
                            
                            # Update search results
                            success_count = 0
                            failed_count = 0
                            for result in results:
                                url = result['url']
                                if url in url_to_content:
                                    fetch_result = url_to_content[url]
                                    # logger.info(f"[loop] result: {result} \n url: {url}, \n url_to_content: {url_to_content}, \n fetch_result: {fetch_result} \n")
                                    if fetch_result['success']:
                                        # Successfully fetched content
                                        result['raw_content'] = fetch_result['content']
                                        
                                        # If title is empty, use fetched title
                                        if not result.get('title') and fetch_result.get('title'):
                                            result['title'] = fetch_result['title']
                                        
                                        # Add fetch method information
                                        result['fetch_method'] = fetch_result.get('method', 'unknown')
                                        result['fetch_status'] = fetch_result.get('status')
                                        success_count += 1
                                        
                                    else:
                                        # Fetch failed, use error information
                                        error_msg = fetch_result.get('error', 'Unknown error')
                                        result['raw_content'] = f"[Smart fetch failed: {error_msg}]"
                                        result['fetch_method'] = 'failed'
                                        result['fetch_error'] = error_msg
                                        failed_count += 1
                                        logger.debug(f"Content fetch failed - url: {url}, error: {error_msg}")
                                else:
                                    # URL not in mapping, keep original content
                                    result['raw_content'] = result.get('content', '')
                                    failed_count += 1
                            
                            logger.info(f"Smart content fetching completed - success_count: {success_count}, failed_count: {failed_count}, total_processed: {len(results)}")
                    
                    except Exception as e:
                        logger.error(f"Smart content fetching exception - error: {str(e)}, error_type: {type(e).__name__}, query: {query}")
                        # Fallback: keep original content as raw_content
                        for result in results:
                            result['raw_content'] = result.get('content', '')
                
                elif include_raw_content and results and not PLAYWRIGHT_UTILS_AVAILABLE:
                    logger.warning("playwright_utils not available, using basic content fetching")
                    # Fallback to simple content copying
                    for result in results:
                        result['raw_content'] = result.get('content', '')
                
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                }
            except Exception as e:
                logger.error(f"Google search query failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": [],
                    "error": str(e)
                }
    
    try:
        # Create tasks for all search queries
        search_tasks = [search_single_query(query) for query in search_queries]
        
        # Execute all searches concurrently
        search_results = await asyncio.gather(*search_tasks)
        
        return search_results
    finally:
        # Only shut down executor if it was created
        if executor:
            executor.shutdown(wait=False)

@traceable
async def gemini_google_search_async(search_queries: Union[str, List[str]], max_results: int = 5, include_raw_content: bool = True, model_name: Optional[str] = None):
    """
    Enhanced search using Gemini (combining Google Search and AI analysis)
    
    Args:
        search_queries (Union[str, List[str]]): List of search queries
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to include raw content
        model_name (Optional[str]): Gemini model name to use. If None, uses default model
        
    Returns:
        List[dict]: List of formatted search results
    """
    # Ensure search_queries is in list format
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    
    # First use regular Google search to get results
    search_results = await google_custom_search_async(search_queries, max_results, include_raw_content)
    logger.info(f"Google search results: {search_results}")
    
    # Check if Gemini API Key is available for enhanced analysis
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set, using regular Google search results")
        return search_results
    
    # Use Gemini for enhanced analysis of search results
    try:
        # Use provided model name or default
        if model_name:
            # Extract model name from full format if necessary (e.g., "google_genai:gemini-2.5-flash" -> "gemini-2.5-flash")
            if ":" in model_name:
                model_name = model_name.split(":", 1)[1]
        else:
            model_name = "gemini-2.5-flash-lite-preview-06-17"
            
        gemini_model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=2
        )
        
        enhanced_results = []
        
        for result in search_results:
            query = result['query']
            logger.info(f"Gemini Google Search [ChatGoogleGenerativeAI] - query: {query}")
            
            # Skip if query is empty or missing
            if not query or not query.strip():
                logger.warning(f"Skipping empty or missing query in search result")
                enhanced_results.append(result)
                continue
            
            results = result['results']
            logger.info(f"Gemini Google Search [ChatGoogleGenerativeAI] - results: {results}")
            # Check if results exist before processing
            if results and len(results) > 0:
                # Build analysis prompt
                sources_summary = "\n".join([
                    f"Title: {item['title']}\nURL: {item['url']}\nContent: {item['content'][:300]}...\n"
                    for item in results[:3]  # Only analyze first 3 results
                ])
            else:
                sources_summary = "\n"
            
            analysis_prompt = f"""
            Based on the following search results, provide a comprehensive analysis and summary for the query "{query}":

            Search Results:
            {sources_summary}

            Please provide:
            1. A concise summary of the query topic (2-3 sentences)
            2. Key findings and important information
            3. Possible follow-up questions or related topics

            Current date: {datetime.datetime.now().strftime('%Y-%m-%d')}
            """
            
            try:
                # Call Gemini for analysis
                response = await gemini_model.ainvoke(analysis_prompt)
                logger.info(f"Gemini analysis response: {response}")
                # logger.info(f"Gemini analysis response (first 100 chars): {str(response)[:100]}")
                ai_analysis = response.content if hasattr(response, 'content') else str(response)
                
                # Enhance original results
                enhanced_result = {
                    **result,
                    "answer": ai_analysis,
                    "enhanced_by_gemini": True
                }
                enhanced_results.append(enhanced_result)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Gemini analysis failed - query: {query}, error: {str(e)}, error_type: {type(e).__name__}")
                # If Gemini analysis fails, return original results
                enhanced_results.append(result)
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Gemini Google Search initialization error - error: {str(e)}, error_type: {type(e).__name__}")
        # If Gemini is unavailable, return regular search results
        return search_results

async def scrape_pages(titles: List[str], urls: List[str]) -> str:
    """
    Scrape page content using smart content fetching technology and format as readable documents.
    
    This function:
    1. Takes a list of page titles and URLs
    2. Uses smart content fetching (Playwright for complex sites, HTTP for simple ones)
    3. Converts HTML content to clean text
    4. Formats all content with clear source attribution
    
    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from
        
    Returns:
        str: A formatted string containing the full content of each page,
             with clear section dividers and source attribution
    """
    
    logger.info(f"Starting page content scraping - pages_count: {len(urls)}, urls: {urls[:5]}")
    
    # 🚀 Use smart content fetching instead of traditional httpx method
    if PLAYWRIGHT_UTILS_AVAILABLE:
        try:
            logger.info(f"Using smart content fetching for pages - pages_count: {len(urls)}, playwright_available: True")
            
            # Specify domains that need special handling
            special_domains = ['x.com', 'twitter.com', 'linkedin.com', 'facebook.com', 'instagram.com', 'youtube.com']
            
            # Use smart content fetching
            fetch_results = await smart_content_fetch(
                urls=urls,
                use_playwright_for=special_domains,
                fallback_to_playwright=True
            )
            
            # Create formatted output
            formatted_output = "Search results: \n\n"
            
            success_count = 0
            failure_count = 0
            
            for i, (title, url, fetch_result) in enumerate(zip(titles, urls, fetch_results)):
                formatted_output += f"\n\n--- SOURCE {i+1}: {title} ---\n"
                formatted_output += f"URL: {url}\n"
                formatted_output += f"Fetch method: {fetch_result.get('method', 'unknown')}\n"
                
                if fetch_result['success']:
                    # Successfully fetched content
                    content = fetch_result['content']
                    original_length = len(content)
                    
                    # If content is too long, truncate appropriately
                    if len(content) > 50000:  # 50KB limit
                        content = content[:50000] + "\n\n[Content truncated...]"
                        logger.debug(f"Content truncated - url: {url}, original_length: {original_length}, truncated_length: {len(content)}")
                    
                    formatted_output += f"Status: ✅ Successfully fetched\n\n"
                    formatted_output += f"FULL CONTENT:\n{content}"
                    success_count += 1
                else:
                    # Fetch failed
                    error_msg = fetch_result.get('error', 'Unknown error')
                    formatted_output += f"Status: ❌ Fetch failed ({error_msg})\n\n"
                    formatted_output += f"CONTENT: [Unable to fetch content: {error_msg}]"
                    failure_count += 1
                    logger.debug(f"Page content fetch failed - url: {url}, error: {error_msg}")
                
                formatted_output += "\n\n" + "-" * 80 + "\n"
            
            logger.info(f"Smart content fetching completed - total_pages: {len(urls)}, success_count: {success_count}, failure_count: {failure_count}")
            
            return formatted_output
            
        except Exception as e:
            logger.error(f"Smart content fetching failed, falling back to traditional method - error: {str(e)}, error_type: {type(e).__name__}")
            # Fall back to traditional method
    
    # 🔄 Fall back to traditional httpx method
    logger.info(f"Using traditional HTTP client for pages - pages_count: {len(urls)}, playwright_available: {PLAYWRIGHT_UTILS_AVAILABLE}")
    
    # Create an async HTTP client
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []
        
        # Fetch each URL and convert to markdown
        for url in urls:
            try:
                # Fetch the content
                response = await client.get(url)
                response.raise_for_status()
                
                # Convert HTML to markdown if successful
                if response.status_code == 200:
                    # Handle different content types
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        # Convert HTML to markdown
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        # For non-HTML content, just mention the content type
                        pages.append(f"Content type: {content_type} (not converted to markdown)")
                else:
                    pages.append(f"Error: Received status code {response.status_code}")
        
            except Exception as e:
                # Handle any exceptions during fetch
                pages.append(f"Error fetching URL: {str(e)}")
        
        # Create formatted output
        formatted_output = f"Search results: \n\n"
        
        for i, (title, url, page) in enumerate(zip(titles, urls, pages)):
            formatted_output += f"\n\n--- SOURCE {i+1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"FULL CONTENT:\n {page}"
            formatted_output += "\n\n" + "-" * 80 + "\n"
    
    return formatted_output

@tool
async def duckduckgo_search(search_queries: List[str]):
    """Perform searches using DuckDuckGo with retry logic to handle rate limits
    
    Args:
        search_queries (List[str]): List of search queries to process
        
    Returns:
        str: A formatted string of search results
    """
    
    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()
        
        def perform_search():
            max_retries = 3
            retry_count = 0
            backoff_factor = 2.0
            last_exception = None
            
            while retry_count <= max_retries:
                try:
                    results = []
                    with DDGS() as ddgs:
                        # Change query slightly and add delay between retries
                        if retry_count > 0:
                            # Random delay with exponential backoff
                            delay = backoff_factor ** retry_count + random.random()
                            print(f"Retry {retry_count}/{max_retries} for query '{query}' after {delay:.2f}s delay")
                            time.sleep(delay)
                            
                            # Add a random element to the query to bypass caching/rate limits
                            modifiers = ['about', 'info', 'guide', 'overview', 'details', 'explained']
                            modified_query = f"{query} {random.choice(modifiers)}"
                        else:
                            modified_query = query
                        
                        # Execute search
                        ddg_results = list(ddgs.text(modified_query, max_results=5))
                        
                        # Format results
                        for i, result in enumerate(ddg_results):
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('href', ''),
                                'content': result.get('body', ''),
                                'score': 1.0 - (i * 0.1),  # Simple scoring mechanism
                                'raw_content': result.get('body', '')
                            })
                        
                        # Return successful results
                        return {
                            'query': query,
                            'follow_up_questions': None,
                            'answer': None,
                            'images': [],
                            'results': results
                        }
                except Exception as e:
                    # Store the exception and retry
                    last_exception = e
                    retry_count += 1
                    print(f"DuckDuckGo search error: {str(e)}. Retrying {retry_count}/{max_retries}")
                    
                    # If not a rate limit error, don't retry
                    if "Ratelimit" not in str(e) and retry_count >= 1:
                        print(f"Non-rate limit error, stopping retries: {str(e)}")
                        break
            
            # If we reach here, all retries failed
            print(f"All retries failed for query '{query}': {str(last_exception)}")
            # Return empty results but with query info preserved
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(last_exception)
            }
            
        return await loop.run_in_executor(None, perform_search)

    # Process queries with delay between them to reduce rate limiting
    search_docs = []
    urls = []
    titles = []
    for i, query in enumerate(search_queries):
        # Add delay between queries (except first one)
        if i > 0:
            delay = 2.0 + random.random() * 2.0  # Random delay 2-4 seconds
            await asyncio.sleep(delay)
        
        # Process the query
        result = await process_single_query(query)
        search_docs.append(result)
        
        # Safely extract URLs and titles from results, handling empty result cases
        if result['results'] and len(result['results']) > 0:
            for res in result['results']:
                if 'url' in res and 'title' in res:
                    urls.append(res['url'])
                    titles.append(res['title'])
    
    # If we got any valid URLs, scrape the pages
    if urls:
        return await scrape_pages(titles, urls)
    else:
        return "No valid search results found. Please try different search queries or use a different search API."

TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args:
        queries (List[str]): List of search queries
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    # Use tavily_search_async with include_raw_content=True to get content directly
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # Format the search results directly using the raw_content already provided
    formatted_output = f"Search results: \n\n"
    
    # Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    async def noop():
        return None

    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 30_000
    # TODO: share this behavior across all search implementations / tools
    if configurable.process_search_results == "summarize":
        if configurable.summarization_model_provider == "anthropic":
            extra_kwargs = {"betas": ["extended-cache-ttl-2025-04-11"]}
        else:
            extra_kwargs = {}

        summarization_model = init_chat_model(
            model=configurable.summarization_model,
            model_provider=configurable.summarization_model_provider,
            max_retries=configurable.max_structured_output_retries,
            **extra_kwargs
        )
        summarization_tasks = [
            noop() if not result.get("raw_content") else summarize_webpage(summarization_model, result['raw_content'][:max_char_to_include])
            for result in unique_results.values()
        ]
        summaries = await asyncio.gather(*summarization_tasks)
        unique_results = {
            url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
            for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
        }
    elif configurable.process_search_results == "split_and_rerank":
        embeddings = init_embeddings("openai:text-embedding-3-small")
        results_by_query = itertools.groupby(unique_results.values(), key=lambda x: x['query'])
        all_retrieved_docs = []
        for query, query_results in results_by_query:
            retrieved_docs = split_and_rerank_search_results(embeddings, query, query_results)
            all_retrieved_docs.extend(retrieved_docs)

        stitched_docs = stitch_documents_by_url(all_retrieved_docs)
        unique_results = {
            doc.metadata['url']: {'title': doc.metadata['title'], 'content': doc.page_content}
            for doc in stitched_docs
        }

    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:max_char_to_include]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


@tool
async def azureaisearch_search(queries: List[str], max_results: int = 5, topic: str = "general") -> str:
    """
    Fetches results from Azure AI Search API.
    
    Args:
        queries (List[str]): List of search queries
        
    Returns:
        str: A formatted string of search results
    """
    # Use azureaisearch_search_async with include_raw_content=True to get content directly
    search_results = await azureaisearch_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # Format the search results directly using the raw_content already provided
    formatted_output = f"Search results: \n\n"
    
    # Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:30000]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


@tool
async def gemini_google_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None
) -> str:
    """
    Search using Gemini native Google Search tool
    
    Args:
        queries (List[str]): List of search queries
        max_results (int): Maximum number of results
        config: Runtime configuration
        
    Returns:
        str: Formatted search results string
    """
    # Extract model name from config if available
    model_name = None
    if config:
        from open_deep_research.configuration import Configuration
        configurable = Configuration.from_runnable_config(config)
        # Note: Using researcher_model for graph workflow compatibility
        model_name = getattr(configurable, 'researcher_model', None)
        
    # Use gemini_google_search_async to perform search
    search_results = await gemini_google_search_async(
        queries,
        max_results=max_results,
        include_raw_content=True,
        model_name=model_name
    )
    
    # Format output
    formatted_output = "Google Search Results (via Gemini):\n\n"
    
    # Deduplicate and format results
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    
    # Process search results
    async def noop():
        return None
    
    # If result processing is configured, apply appropriate processing method
    configurable = Configuration.from_runnable_config(config) if config else None
    max_char_to_include = 30_000
    
    if configurable and configurable.process_search_results == "summarize":
        # Use summary mode to process results
        if configurable.summarization_model_provider == "anthropic":
            extra_kwargs = {"betas": ["extended-cache-ttl-2025-04-11"]}
        else:
            extra_kwargs = {}

        summarization_model = init_chat_model(
            model=configurable.summarization_model,
            model_provider=configurable.summarization_model_provider,
            max_retries=configurable.max_structured_output_retries,
            **extra_kwargs
        )
        summarization_tasks = [
            noop() if not result.get("raw_content") else summarize_webpage(summarization_model, result['raw_content'][:max_char_to_include])
            for result in unique_results.values()
        ]
        summaries = await asyncio.gather(*summarization_tasks)
        unique_results = {
            url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
            for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
        }
    
    # Format final output
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:max_char_to_include]}"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "No valid search results found using Gemini Google Search. Please try different search queries."


async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict, enable_playwright_optimization: bool = True, config: Optional[RunnableConfig] = None) -> str:
    """Select and execute the appropriate search API with optional Playwright optimization.
    
    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API
        enable_playwright_optimization: Whether to enable Playwright optimization for raw search results
        config: Optional runtime configuration
        
    Returns:
        Formatted string containing search results
        
    Raises:
        ValueError: If an unsupported search API is specified
    """
    logger.info(f"Starting search execution - search_api: {search_api}, query_count: {len(query_list)}, queries: {query_list[:3]}, params: {params_to_pass}, enable_playwright_optimization: {enable_playwright_optimization}")
    
    # Disable Playwright optimization by default on Windows due to subprocess compatibility issues
    current_platform = platform.system()
    if current_platform == "Windows" and enable_playwright_optimization:
        logger.warning(f"Windows environment detected, automatically disabling Playwright optimization due to asyncio subprocess compatibility issues - platform: {current_platform}")
        enable_playwright_optimization = False
    
    try:
        if search_api == "tavily":
            logger.debug("Using Tavily search")
            # Tavily search tool used with both workflow and agent 
            # and returns a formatted source string
            search_results = await tavily_search.ainvoke({'queries': query_list, **params_to_pass})
            logger.info(f"Tavily search completed - result_length: {len(search_results)}")
            logger.info(f"Tavily search results: {search_results}")
            # return result
        elif search_api == "duckduckgo":
            logger.debug("Using DuckDuckGo search")
            # DuckDuckGo search tool used with both workflow and agent 
            search_results = await duckduckgo_search.ainvoke({'search_queries': query_list})
            logger.info(f"DuckDuckGo search completed - result_length: {len(search_results)}")
            logger.info(f"DuckDuckGo search results: {search_results}")
            # return result
        elif search_api == "perplexity":
            logger.debug("Using Perplexity search")
            search_results = perplexity_search(query_list, **params_to_pass)
            logger.info(f"Perplexity search completed - result_length: {len(search_results)}")
            logger.info(f"Perplexity search results: {search_results}")
        elif search_api == "exa":
            logger.debug("Using Exa search")
            search_results = await exa_search(query_list, **params_to_pass)
            logger.info(f"Exa search completed - result_length: {len(search_results)}")
            logger.info(f"Exa search results: {search_results}")
        elif search_api == "arxiv":
            logger.debug("Using arXiv search")
            search_results = await arxiv_search_async(query_list, **params_to_pass)
            logger.info(f"arXiv search completed - result_length: {len(search_results)}")
            logger.info(f"arXiv search results: {search_results}")
        elif search_api == "pubmed":
            logger.debug("Using PubMed search")
            search_results = await pubmed_search_async(query_list, **params_to_pass)
            logger.info(f"PubMed search completed - result_length: {len(search_results)}")
            logger.info(f"PubMed search results: {search_results}")
        elif search_api == "linkup":
            logger.debug("Using Linkup search")
            search_results = await linkup_search(query_list, **params_to_pass)
            logger.info(f"Linkup search completed - result_length: {len(search_results)}")
            logger.info(f"Linkup search results: {search_results}")
        elif search_api == "googlesearch":
            logger.debug("Using Google search")
            search_results = await google_custom_search_async(query_list, **params_to_pass)
            logger.info(f"Google search completed - result_length: {len(search_results)}")
            logger.info(f"Google search results: {search_results}")
        elif search_api == "azureaisearch":
            logger.debug("Using Azure AI search")
            search_results = await azureaisearch_search(query_list, **params_to_pass)
            logger.info(f"Azure AI search completed - result_length: {len(search_results)}")
            logger.info(f"Azure AI search results: {search_results}")
        elif search_api == "geminigooglesearch":
            logger.debug("Using Gemini Google search")
            # Extract model name from config if available
            model_name = None
            if config:
                from open_deep_research.configuration import MultiAgentConfiguration
                configurable = MultiAgentConfiguration.from_runnable_config(config)
                model_name = get_config_value(configurable.researcher_model)
            search_results = await gemini_google_search_async(query_list, model_name=model_name, **params_to_pass)
            logger.info(f"Gemini Google search completed - result_length: {len(search_results)}")
            logger.info(f"Gemini Google search results: {search_results}")
        else:
            logger.error(f"Unsupported search API - search_api: {search_api}")
            raise ValueError(f"Unsupported search API: {search_api}")
    
        # logger.info(f"Search completed - search_api: {search_api}, result_length: {len(search_results) if search_results else 0}")
        # logger.info(f"Search results: {search_results}")
    except Exception as e:
        logger.error(f"Search API call failed - search_api: {search_api}, error: {str(e)}, error_type: {type(e).__name__}")
        raise

    # 🚀 Optional Playwright optimization step
    if enable_playwright_optimization and PLAYWRIGHT_UTILS_AVAILABLE:
        try:
            logger.info("Starting Playwright optimization")
            
            # Collect all search results for optimization
            all_results = []
            for result in search_results:
                if isinstance(result, dict) and 'results' in result:
                    all_results.extend(result['results'])
            
            if all_results:
                logger.info(f"Preparing Playwright optimization - results_count: {len(all_results)}")
                
                # Apply Playwright optimization
                optimized_results = await optimize_search_results_with_playwright(
                    all_results,
                    max_concurrent=3,
                    content_char_limit=50000
                )
                
                # Show optimization summary
                summary = get_enhanced_search_summary(optimized_results)
                logger.info(f"Playwright optimization summary - summary: {summary}")
                
                # Map optimized results back to original structure
                optimized_map = {res['url']: res for res in optimized_results}
                mapped_count = 0
                for result in search_results:
                    if isinstance(result, dict) and 'results' in result:
                        for i, res in enumerate(result['results']):
                            if res['url'] in optimized_map:
                                result['results'][i] = optimized_map[res['url']]
                                mapped_count += 1
                
                logger.info(f"Playwright optimization completed - total_results: {len(all_results)}, optimized_results: {len(optimized_results)}, mapped_results: {mapped_count}")
            else:
                logger.debug("No search results to optimize")
        
        except Exception as e:
            logger.error(f"Playwright optimization failed - error: {str(e)}, error_type: {type(e).__name__}")
    elif not enable_playwright_optimization:
        logger.debug("Playwright optimization disabled")
    elif not PLAYWRIGHT_UTILS_AVAILABLE:
        logger.warning("Playwright tool not available, skipping optimization")

    return deduplicate_and_format_sources(search_results, max_tokens_per_source=50000, deduplication_strategy="keep_first")


class Summary(BaseModel):
    summary: str
    key_excerpts: list[str]


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content."""
    try:
        user_input_content = "Please summarize the article"
        if isinstance(model, ChatAnthropic):
            user_input_content = [{
                "type": "text",
                "text": user_input_content,
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }]

        summary = await model.with_structured_output(Summary).with_retry(stop_after_attempt=2).ainvoke([
            {"role": "system", "content": SUMMARIZATION_PROMPT.format(webpage_content=webpage_content)},
            {"role": "user", "content": user_input_content},
        ])
    except:
        # fall back on the raw content
        return webpage_content

    def format_summary(summary: Summary):
        excerpts_str = "\n".join(f'- {e}' for e in summary.key_excerpts)
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{excerpts_str}\n</key_excerpts>"""

    return format_summary(summary)


def split_and_rerank_search_results(embeddings: Embeddings, query: str, search_results: list[dict], max_chunks: int = 5):
    # split webpage content into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, add_start_index=True
    )
    documents = [
        Document(
            page_content=result.get('raw_content') or result['content'],
            metadata={"url": result['url'], "title": result['title']}
        )
        for result in search_results
    ]
    all_splits = text_splitter.split_documents(documents)

    # index chunks
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)

    # retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(query, k=max_chunks)
    return retrieved_docs


def stitch_documents_by_url(documents: list[Document]) -> list[Document]:
    url_to_docs: defaultdict[str, list[Document]] = defaultdict(list)
    url_to_snippet_hashes: defaultdict[str, set[str]] = defaultdict(set)
    for doc in documents:
        snippet_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
        url = doc.metadata['url']
        # deduplicate snippets by the content
        if snippet_hash in url_to_snippet_hashes[url]:
            continue

        url_to_docs[url].append(doc)
        url_to_snippet_hashes[url].add(snippet_hash)

    # stitch retrieved chunks into a single doc per URL
    stitched_docs = []
    for docs in url_to_docs.values():
        stitched_doc = Document(
            page_content="\n\n".join([f"...{doc.page_content}..." for doc in docs]),
            metadata=cast(Document, docs[0]).metadata
        )
        stitched_docs.append(stitched_doc)

    return stitched_docs


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    # return datetime.datetime.now().strftime("%a %b %-d, %Y")
    # # Use %d instead of %-d for Windows compatibility
    return datetime.datetime.now().strftime("%a %b %d, %Y")


async def load_mcp_server_config(path: str) -> dict:
    """Load MCP server configuration from a file."""

    def _load():
        with open(path, "r") as f:
            config = json.load(f)
        return config

    config = await asyncio.to_thread(_load)
    return config

async def intelligent_search_web_unified(
    queries: List[str],
    config: RunnableConfig,
    search_api: str = "tavily",
    search_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Unified intelligent search interface
    
    Decides whether to use intelligent research mode or traditional search based on configuration
    
    Args:
        queries: List of search queries
        config: Runtime configuration
        search_api: Search API type
        search_params: Search parameters
        
    Returns:
        Search results string
    """
    logger.info(f"Starting unified intelligent search - queries_count: {len(queries)}, search_api: {search_api}, queries: {queries[:3]}")  # Only log first 3 queries
    
    try:
        # Get research mode configuration
        research_mode = "simple"  # Default value
        
        # Try to get research_mode and iteration count from configuration
        configurable = config.get("configurable", {})
        if isinstance(configurable, dict):
            research_mode = configurable.get("research_mode", "simple")
            max_iterations = configurable.get("max_research_iterations", 3)
        else:
            max_iterations = 3
        
        logger.debug(f"Search configuration parsed - research_mode: {research_mode}, max_iterations: {max_iterations}, search_params: {search_params}")
        
        # If intelligent mode is enabled, use intelligent research manager
        if research_mode and research_mode != "simple":
            try:
                logger.info(f"Enabling intelligent research mode - research_mode: {research_mode}")
                
                from open_deep_research.intelligent_research import (
                    IntelligentResearchManager,
                    ResearchMode
                )
                
                # Create intelligent research manager
                manager = IntelligentResearchManager()
                
                # Parse research mode
                try:
                    mode = ResearchMode(research_mode)
                    logger.debug(f"Research mode parsed successfully - mode: {mode.value}")
                except ValueError:
                    mode = ResearchMode.REFLECTIVE  # Default reflective mode
                    logger.warning(f"Research mode parsing failed, using default mode - original_mode: {research_mode}, default_mode: {mode.value}")
                
                logger.info(f"Starting intelligent research execution - query: {queries[0] if queries else ''}, mode: {mode.value}, max_iterations: {max_iterations}")
                
                # Execute intelligent research
                result = await manager.conduct_research(
                    query=queries[0] if queries else "",
                    config=config,
                    mode=mode,
                    max_iterations=max_iterations
                )
                
                # Format intelligent research results
                if "error" not in result:
                    logger.info(f"Intelligent research executed successfully - iterations: {result.get('iterations', 0)}")
                    
                    # 🔥 First choice: directly return final report/answer from intelligent research
                    if "final_report" in result:
                        logger.info(f"Returning intelligent research final report - report_length: {len(result['final_report'])}")
                        return result["final_report"]
                    
                    if "answer" in result:
                        logger.info(f"Returning intelligent research answer - answer_length: {len(result['answer'])}")
                        return result["answer"]
                    
                    # 🔍 Second choice: if search results exist, format output
                    if "search_results" in result and result["search_results"]:
                        logger.info(f"Formatting intelligent research search results - results_count: {len(result['search_results'])}")
                        formatted_results = []
                        for i, res in enumerate(result["search_results"][:10], 1):
                            if isinstance(res, dict):
                                title = res.get("title", "No title")
                                content = res.get("content", res.get("snippet", ""))
                                url = res.get("url", "")
                                formatted_results.append(f"{i}. **{title}**\n{content[:500]}...\nURL: {url}\n")
                        
                        if formatted_results:
                            logger.debug(f"Intelligent research results formatting completed - formatted_count: {len(formatted_results)}")
                            return "\n".join(formatted_results)
                    
                    # 🔄 Last fallback: use enhanced queries for traditional search only when traditional search API is available
                    if search_api != "none":
                        enhanced_queries = result.get("queries", queries)
                        logger.info(f"Using enhanced queries for traditional search - enhanced_queries_count: {len(enhanced_queries)}, enhanced_queries: {enhanced_queries[:3]}")
                        return await select_and_execute_search(search_api, enhanced_queries, search_params or {}, config=config)
                    else:
                        # If it's none API, return basic results from intelligent research
                        queries_count = len(result.get("queries", []))
                        logger.info(f"Intelligent research completed (no additional search needed) - generated_queries_count: {queries_count}")
                        return f"Intelligent research completed, generated {queries_count} enhanced queries:\n" + \
                               "\n".join(f"- {q}" for q in result.get("queries", []))
                else:
                    error_msg = result.get('error', 'Unknown error')
                    logger.warning(f"Intelligent research failed, falling back to traditional search - error: {error_msg}")
                    
            except ImportError as e:
                logger.warning(f"Intelligent research module import failed, falling back to traditional search - import_error: {str(e)}")
            except Exception as e:
                logger.error(f"Intelligent research execution failed, falling back to traditional search - error: {str(e)}, error_type: {type(e).__name__}")
        
        # Use traditional search as fallback
        logger.info(f"Using traditional search mode - search_api: {search_api}")
        
        # Check if Playwright optimization is enabled
        enable_optimization = search_params and search_params.get('enable_playwright_optimization', True)
        logger.info(f"Playwright optimization setting - enable_optimization: {enable_optimization}")
        
        result = await select_and_execute_search(
            search_api, 
            queries, 
            search_params or {}, 
            enable_playwright_optimization=enable_optimization,
            config=config
        )
        
        logger.info(f"Unified intelligent search completed - search_api: {search_api}, result_length: {len(result) if result else 0}")
        
        return result
        
    except Exception as e:
        logger.error(f"Search execution failed - error: {str(e)}, error_type: {type(e).__name__}, search_api: {search_api}, queries_count: {len(queries)}")
        # Final fallback: return empty result
        return f"Search failed: {str(e)}"

async def optimize_search_results_with_playwright(
    search_results: List[Dict[str, Any]], 
    max_concurrent: int = 3,
    content_char_limit: int = 40000
) -> List[Dict[str, Any]]:
    """
    Optimize search results using Playwright intelligent content fetching
    
    Args:
        search_results: List of search results, each containing 'url', 'title', 'content' and other fields
        max_concurrent: Maximum number of concurrent fetches
        content_char_limit: Character limit for content
        
    Returns:
        List of optimized search results
    """
    if not PLAYWRIGHT_UTILS_AVAILABLE:
        logger.warning("playwright_utils not available, returning original search results")
        return search_results
    
    try:
        logger.info(f"Starting search results optimization - results_count: {len(search_results)}")
        
        # Extract URLs that need content fetching
        urls_to_fetch = []
        url_to_result_map = {}
        
        for result in search_results:
            url = result.get('url')
            if url:
                urls_to_fetch.append(url)
                url_to_result_map[url] = result
        
        if not urls_to_fetch:
            logger.debug(f"No URLs found for content fetching: {urls_to_fetch}")
            return search_results
        
        # Specify domains that need special handling
        special_domains = [
            'x.com', 'twitter.com', 
            'linkedin.com', 'facebook.com', 'instagram.com',
            'youtube.com', 'reddit.com', 'tiktok.com'
        ]
        
        # Use smart content fetching
        fetch_results = await smart_content_fetch(
            urls=urls_to_fetch,
            use_playwright_for=special_domains,
            fallback_to_playwright=True
        )
        
        # Count fetch results
        success_count = 0
        failure_count = 0
        
        # Update search results
        for fetch_result in fetch_results:
            original_url = fetch_result.get('original_url')
            if original_url in url_to_result_map:
                result = url_to_result_map[original_url]
                
                if fetch_result['success']:
                    # Successfully fetched content
                    content = fetch_result['content']
                    
                    # Content length limit
                    if len(content) > content_char_limit * 2:
                        content = content[:content_char_limit * 2] + "\n\n[Content truncated...]"
                    
                    # Update results
                    result['raw_content'] = content
                    result['enhanced_content'] = content  # Enhanced content
                    
                    # If original title is empty or too short, use fetched title
                    if (not result.get('title') or len(result.get('title', '')) < 10) and fetch_result.get('title'):
                        result['title'] = fetch_result['title']
                    
                    # Add fetch metadata
                    result['fetch_method'] = fetch_result.get('method', 'unknown')
                    result['fetch_status'] = fetch_result.get('status')
                    result['content_enhanced'] = True
                    
                    success_count += 1
                    
                else:
                    # Fetch failed
                    error_msg = fetch_result.get('error', 'Unknown error')
                    result['raw_content'] = result.get('content', '')  # Keep original content
                    result['fetch_method'] = 'failed'
                    result['fetch_error'] = error_msg
                    result['content_enhanced'] = False
                    
                    failure_count += 1
        
        logger.info(f"Search results optimization completed - success_count: {success_count}, failure_count: {failure_count}")
        return search_results
        
    except Exception as e:
        logger.error(f"Search results optimization failed - error: {str(e)}, error_type: {type(e).__name__}")
        # Return original results, but mark as not enhanced
        for result in search_results:
            result['content_enhanced'] = False
            result['raw_content'] = result.get('content', '')
        return search_results

def get_enhanced_search_summary(search_results: List[Dict[str, Any]]) -> str:
    """
    Get summary information of enhanced search results
    
    Args:
        search_results: List of search results
        
    Returns:
        Summary information string
    """
    if not search_results:
        return "No search results"
    
    total_results = len(search_results)
    enhanced_count = sum(1 for r in search_results if r.get('content_enhanced', False))
    
    # Count fetch methods
    methods = {}
    for result in search_results:
        method = result.get('fetch_method', 'unknown')
        methods[method] = methods.get(method, 0) + 1
    
    method_summary = ", ".join([f"{method}: {count}" for method, count in methods.items()])
    
    return f"Search Results Summary: Total {total_results} results, {enhanced_count} content enhanced successfully\n" \
           f"Fetch Methods Distribution: {method_summary}"
           
def extract_json_from_markdown(self, content: str) -> str:
    """Extract JSON string from Markdown formatted response"""
    # Match content between ```json and ```
    json_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no ```json format found, try to match any ``` code block
    code_block_pattern = r'```.*?\n(.*?)\n```'
    match = re.search(code_block_pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If none found, return original content
    return content.strip()
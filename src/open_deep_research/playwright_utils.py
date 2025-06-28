"""
Playwright-based web content fetching utilities for open_deep_research
"""

import asyncio
import random
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlparse
import structlog
logger = structlog.get_logger(__name__)

try:
    import aiohttp
    from bs4 import BeautifulSoup
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp or BeautifulSoup not available. Some features may be limited.")

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. Install with: pip install playwright")

from langsmith import traceable


class PlaywrightContentFetcher:
    """Advanced content fetcher using Playwright to handle complex websites"""
    
    def __init__(self, 
                 browser_type: str = "chromium",
                 headless: bool = True,
                 user_data_dir: Optional[str] = None):
        """
        Initialize Playwright content fetcher
        
        Args:
            browser_type: Browser type ("chromium", "firefox", "webkit")
            headless: Whether to run in headless mode
            user_data_dir: User data directory for session persistence
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required. Install with: pip install playwright")
        
        self.browser_type = browser_type
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def start(self):
        """Start browser and context"""
        self.playwright = await async_playwright().start()
        
        # Select browser
        if self.browser_type == "chromium":
            browser_launcher = self.playwright.chromium
        elif self.browser_type == "firefox":
            browser_launcher = self.playwright.firefox
        elif self.browser_type == "webkit":
            browser_launcher = self.playwright.webkit
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
        
        # Launch browser
        launch_options = {
            "headless": self.headless,
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor"
            ]
        }
        
        if self.user_data_dir:
            launch_options["user_data_dir"] = self.user_data_dir
        
        self.browser = await browser_launcher.launch(**launch_options)
        
        # Create context
        context_options = {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": self._get_user_agent(),
            "locale": "en-US",
            "timezone_id": "America/New_York"
        }
        
        self.context = await self.browser.new_context(**context_options)
        
        # Set additional anti-detection measures
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        logger.info(f"Playwright {self.browser_type} browser started")
        
    async def close(self):
        """Close browser and context"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        
        logger.info("Playwright browser closed")
    
    def _get_user_agent(self) -> str:
        """Get random User-Agent"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
        ]
        return random.choice(user_agents)
    
    async def fetch_content(self, 
                          url: str, 
                          wait_for_selector: Optional[str] = None,
                          wait_for_load_state: str = "domcontentloaded",
                          timeout: int = 30000,
                          extract_text_only: bool = True) -> Dict[str, Any]:
        """
        Fetch content from a single URL
        
        Args:
            url: Target URL
            wait_for_selector: Wait for specific selector to appear
            wait_for_load_state: Wait for loading state
            timeout: Timeout in milliseconds
            extract_text_only: Whether to extract text content only
            
        Returns:
            Dictionary containing content information
        """
        page = await self.context.new_page()
        
        try:
            # Set additional anti-detection measures
            await page.set_extra_http_headers({
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            })
            
            # Random delay
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Navigate to page
            response = await page.goto(url, 
                                     wait_until=wait_for_load_state, 
                                     timeout=timeout)
            
            # Wait for specific element (if specified)
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=timeout)
            
            # Additional wait to ensure dynamic content loads
            await page.wait_for_timeout(random.randint(1000, 3000))
            
            # Extract content
            if extract_text_only:
                # Remove script and style tags
                await page.evaluate("""
                    [...document.querySelectorAll('script, style, noscript')].forEach(el => el.remove());
                """)
                content = await page.inner_text('body')
            else:
                content = await page.content()
            
            # Get page information
            title = await page.title()
            final_url = page.url
            
            return {
                "url": final_url,
                "original_url": url,
                "title": title,
                "content": content,
                "status": response.status if response else None,
                "success": True,
                "method": "playwright",
                "browser": self.browser_type
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch {url} with Playwright: {str(e)}")
            return {
                "url": url,
                "original_url": url,
                "title": "",
                "content": f"[Playwright fetch error: {str(e)}]",
                "status": None,
                "success": False,
                "error": str(e),
                "method": "playwright",
                "browser": self.browser_type
            }
        finally:
            await page.close()

@traceable
async def fetch_with_playwright(urls: List[str], 
                               browser_type: str = "chromium",
                               max_concurrent: int = 3,
                               **kwargs) -> List[Dict[str, Any]]:
    """
    使用 Playwright 批量获取多个 URL 的内容
    
    Args:
        urls: URL 列表
        browser_type: 浏览器类型
        max_concurrent: 最大并发数
        **kwargs: 传递给 fetch_content 的其他参数
        
    Returns:
        内容结果列表
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("Playwright is required. Install with: pip install playwright")
    
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_single(url: str, fetcher: PlaywrightContentFetcher):
        async with semaphore:
            return await fetcher.fetch_content(url, **kwargs)
    
    async with PlaywrightContentFetcher(browser_type=browser_type) as fetcher:
        tasks = [fetch_single(url, fetcher) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "url": urls[i],
                    "original_url": urls[i],
                    "title": "",
                    "content": f"[Exception: {str(result)}]",
                    "status": None,
                    "success": False,
                    "error": str(result),
                    "method": "playwright",
                    "browser": browser_type
                })
            else:
                processed_results.append(result)
        
        return processed_results

@traceable
async def smart_content_fetch(urls: List[str], 
                            use_playwright_for: Optional[List[str]] = None,
                            fallback_to_playwright: bool = True) -> List[Dict[str, Any]]:
    """
    Smart content fetching: Choose optimal fetching method based on URL characteristics
    
    Args:
        urls: List of URLs
        use_playwright_for: List of domains to force Playwright usage
        fallback_to_playwright: Whether to fallback to Playwright when regular methods fail
        
    Returns:
        List of content results
    """
    logger.info(f"smart_content_fetch>> Urls: {urls}, use_playwright_for: {use_playwright_for}, fallback_to_playwright: {fallback_to_playwright} \n")
    
    # Default websites that need Playwright
    playwright_domains = {
        "x.com", "twitter.com", 
        "facebook.com", "instagram.com",
        "linkedin.com", "tiktok.com",
        "youtube.com", "reddit.com"
    }
    
    if use_playwright_for:
        playwright_domains.update(use_playwright_for)
    
    # Categorize URLs
    playwright_urls = []
    regular_urls = []
    
    for url in urls:
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        if any(d in domain for d in playwright_domains):
            playwright_urls.append(url)
        else:
            regular_urls.append(url)
    
    results = []
    
    # Handle special websites with Playwright (enhanced error handling)
    if playwright_urls:
        logger.info(f"Using Playwright for {len(playwright_urls)} URLs")
        logger.info(f"Playwright Urls: {playwright_urls}")
        try:
            playwright_results = await fetch_with_playwright(playwright_urls)
            results.extend(playwright_results)
        except NotImplementedError as e:
            logger.error(f"Playwright subprocess not supported on this platform: {e}")
            # Downgrade Playwright URLs to regular method
            logger.warning(f"Falling back to regular HTTP for {len(playwright_urls)} URLs")
            regular_urls.extend(playwright_urls)
        except Exception as e:
            logger.error(f"Playwright failed unexpectedly: {e}")
            # Create error results for failed URLs
            for url in playwright_urls:
                results.append({
                    "url": url,
                    "original_url": url,
                    "title": "",
                    "content": f"[Playwright failed: {str(e)}]",
                    "status": None,
                    "success": False,
                    "error": str(e),
                    "method": "playwright_failed"
                })
    
    # Handle other websites with regular method
    if regular_urls:
        logger.info(f"Using regular HTTP client for {len(regular_urls)} URLs")
        logger.info(f"Regular Urls: {regular_urls}")
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp and BeautifulSoup are required for regular HTTP fetching")
            # If dependencies unavailable, try fallback to Playwright (if supported and enabled)
            if fallback_to_playwright and PLAYWRIGHT_AVAILABLE:
                logger.warning("Falling back to Playwright for regular URLs due to missing dependencies")
                try:
                    fallback_results = await fetch_with_playwright(regular_urls)
                    results.extend(fallback_results)
                except NotImplementedError:
                    logger.error("Playwright subprocess not supported for fallback")
                    for url in regular_urls:
                        results.append({
                            "url": url,
                            "original_url": url,
                            "title": "",
                            "content": "[Missing dependencies and Playwright not available]",
                            "status": None,
                            "success": False,
                            "error": "Missing dependencies and Playwright subprocess not supported",
                            "method": "failed"
                        })
                except Exception as e:
                    logger.error(f"Playwright fallback also failed: {e}")
                    for url in regular_urls:
                        results.append({
                            "url": url,
                            "original_url": url,
                            "title": "",
                            "content": f"[All methods failed: {str(e)}]",
                            "status": None,
                            "success": False,
                            "error": str(e),
                            "method": "failed"
                        })
            else:
                for url in regular_urls:
                    results.append({
                        "url": url,
                        "original_url": url,
                        "title": "",
                        "content": "[Missing dependencies: aiohttp and BeautifulSoup required]",
                        "status": None,
                        "success": False,
                        "error": "Missing dependencies",
                        "method": "failed"
                    })
            logger.info(f"Results Length: {len(results)} \n Regular Urls: {regular_urls} \n Results: {results}")
            return results
        
        try:
            
            async def fetch_single_url(url: str) -> Dict[str, Any]:
                """Fetch content from single URL using aiohttp"""
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                try:
                    timeout = aiohttp.ClientTimeout(total=15)
                    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content_type = response.headers.get('Content-Type', '').lower()
                                
                                if 'text/html' in content_type:
                                    html = await response.text()
                                    soup = BeautifulSoup(html, 'html.parser')
                                    
                                    # Remove script and style tags
                                    for script in soup(["script", "style", "noscript"]):
                                        script.decompose()
                                    
                                    title = soup.title.string if soup.title else ""
                                    content = soup.get_text()
                                    
                                    # Clean text
                                    content = '\n'.join(line.strip() for line in content.splitlines() if line.strip())
                                    
                                    return {
                                        "url": response.url.human_repr(),
                                        "original_url": url,
                                        "title": title.strip() if title else "",
                                        "content": content[:1000] + "..." if len(content) > 1000 else content,
                                        "status": response.status,
                                        "success": True,
                                        "method": "aiohttp"
                                    }
                                else:
                                    return {
                                        "url": url,
                                        "original_url": url,
                                        "title": f"Binary content ({content_type})",
                                        "content": f"[Binary content type: {content_type}]",
                                        "status": response.status,
                                        "success": True,
                                        "method": "aiohttp"
                                    }
                            else:
                                return {
                                    "url": url,
                                    "original_url": url,
                                    "title": "",
                                    "content": f"[HTTP Error: {response.status}]",
                                    "status": response.status,
                                    "success": False,
                                    "error": f"HTTP {response.status}",
                                    "method": "aiohttp"
                                }
                                
                except Exception as e:
                    return {
                        "url": url,
                        "original_url": url,
                        "title": "",
                        "content": f"[Fetch error: {str(e)}]",
                        "status": None,
                        "success": False,
                        "error": str(e),
                        "method": "aiohttp"
                    }
            
            # Fetch all regular URLs concurrently
            semaphore = asyncio.Semaphore(5)  # Limit concurrency
            
            async def fetch_with_semaphore(url: str):
                async with semaphore:
                    await asyncio.sleep(random.uniform(0.1, 0.5))  # Random delay
                    return await fetch_single_url(url)
            
            tasks = [fetch_with_semaphore(url) for url in regular_urls]
            regular_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exception results
            processed_regular_results = []
            for i, result in enumerate(regular_results):
                if isinstance(result, Exception):
                    processed_regular_results.append({
                        "url": regular_urls[i],
                        "original_url": regular_urls[i],
                        "title": "",
                        "content": f"[Exception: {str(result)}]",
                        "status": None,
                        "success": False,
                        "error": str(result),
                        "method": "aiohttp"
                    })
                else:
                    processed_regular_results.append(result)
            logger.info(f"Processed Regular Results Length: {len(processed_regular_results)}")
            logger.debug(f"Processed Regular Results: {processed_regular_results}")
            results.extend(processed_regular_results)
            
        except Exception as e:
            if fallback_to_playwright and PLAYWRIGHT_AVAILABLE:
                logger.warning(f"Regular fetch failed, attempting Playwright fallback: {e}")
                try:
                    fallback_results = await fetch_with_playwright(regular_urls)
                    results.extend(fallback_results)
                except NotImplementedError:
                    logger.error("Playwright subprocess not supported for fallback")
                    for url in regular_urls:
                        results.append({
                            "url": url,
                            "original_url": url,
                            "title": "",
                            "content": f"[Regular fetch failed and Playwright not supported: {str(e)}]",
                            "status": None,
                            "success": False,
                            "error": f"Regular fetch failed: {str(e)}, Playwright subprocess not supported",
                            "method": "failed"
                        })
                except Exception as fallback_error:
                    logger.error(f"Both regular fetch and Playwright fallback failed: {fallback_error}")
                    for url in regular_urls:
                        results.append({
                            "url": url,
                            "original_url": url,
                            "title": "",
                            "content": f"[All methods failed: Regular={str(e)}, Playwright={str(fallback_error)}]",
                            "status": None,
                            "success": False,
                            "error": f"Regular: {str(e)}, Playwright: {str(fallback_error)}",
                            "method": "failed"
                        })
            else:
                # Return error results
                for url in regular_urls:
                    results.append({
                        "url": url,
                        "original_url": url,
                        "title": "",
                        "content": f"[Fetch failed: {str(e)}]",
                        "status": None,
                        "success": False,
                        "error": str(e),
                        "method": "failed"
                    })
    logger.info(f"smart_content_fetch>> Results Length: {len(results)}")
    logger.debug(f"smart_content_fetch>> Results: {results}")
    return results

# Installation guidance function
async def ensure_playwright_installed():
    """Ensure Playwright is properly installed and configured"""
    if not PLAYWRIGHT_AVAILABLE:
        return False, "Playwright not installed. Run: pip install playwright"
    
    try:
        # Test basic functionality
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto("data:text/html,<h1>Test</h1>")
            content = await page.inner_text("h1")
            await browser.close()
            
            if content.strip() == "Test":
                return True, "Playwright is working correctly"
            else:
                return False, "Playwright test failed"
                
    except Exception as e:
        return False, f"Playwright error: {str(e)}. You may need to run: playwright install"

if __name__ == "__main__":
    # Simple test
    async def test():
        working, message = await ensure_playwright_installed()
        print(f"Playwright status: {message}")
        
        if working:
            test_urls = ["https://httpbin.org/html", "https://x.com/idoubicc/status/1934813097455415716"]
            results = await fetch_with_playwright(test_urls)
            print(f"Test results: {len(results)} items")
            for result in results:
                print(f"  {result['url']}: {result['success']}")
                
            print("--------------------------------")
            print("Smart content fetch")
            print("--------------------------------")
            results = await smart_content_fetch(test_urls)
            for result in results:
                print(f"\n📋 URL: {result['url']}")
                print(f"   Method: {result['method']}")
                print(f"   Success: {result['success']}")
                print(f"   Title: {result.get('title', 'N/A')}")
                if result['success'] and result.get('content'):
                    preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                    print(f"   Content: {preview}")
                elif 'error' in result:
                    print(f"   Error: {result['error']}")
    
    asyncio.run(test()) 
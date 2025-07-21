import jinja2
from server import mcp
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


search_results_template_string = """
<search_results>
{% for search_result in search_results %}
<result>
<link>{{ search_result.link }}</link>
<description>
{{ search_result.description }}
</description>
</result>
{% endfor %}
</search_results>
"""
search_results_template = jinja2.Template(search_results_template_string)


@mcp.tool()
async def search_brave(search_query: str, max_retries: int = 3) -> str:
    """
    Uses the Brave Search tool to perform a web search and returns a list of results

    Args:
        search_query (str): A search query to enter into the Brave web search tool.
        max_retries (int): The maximum number of times to re-try if there is a failure.
    
    Returns:
        A list of search results, with each search result consisting of a link and a description.
    """

    playwright = await async_playwright().start()
    browser = await playwright.firefox.launch()
    page = await browser.new_page()
    failed = True
    for i in range(max_retries):
        try:
            await page.goto("https://search.brave.com")
            await page.get_by_test_id("searchbox").click()
            await page.keyboard.type(search_query, delay=100)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state('networkidle')
            page_content =  await page.content()
            failed = False
        except Exception as e:
            await page.wait_for_timeout(5000)
            continue

    if failed:
        return "<search_results>\nNo search results.\n</search_results>"

    soup = BeautifulSoup(page_content, 'html.parser')
    raw_search_results = soup.find_all("div", {"data-pos": True})
    results = []
    for raw_result in raw_search_results:
        try:
            header = raw_result.find("a")
            link = header.get("href")
            description = raw_result.find("div", {"class": "snippet-content"}).text
            text_description = f"{header.text}\n\n{description}"
            results.append({
                "link": link,
                "description": text_description
            })
        except Exception as e:
            continue
    
    if len(results) == 0:
        return "<search_results>\nNo search results.\n</search_results>"

    return search_results_template.render(search_results=results)


@mcp.tool()
async def search_duckduckgo(search_query: str, max_retries: int = 3) -> str:
    """
    Uses the DuckDuckGo Search tool to perform a web search and returns a list of results

    Args:
        search_query (str): A search query to enter into the Brave web search tool.
        max_retries (int): The maximum number of times to re-try if there is a failure.
    
    Returns:
        A list of search results, with each search result consisting of a link and a description.
    """

    playwright = await async_playwright().start()
    browser = await playwright.firefox.launch()
    page = await browser.new_page()
    failed = True
    for i in range(max_retries):
        try:
            await page.goto("https://duckduckgo.com")
            await page.locator("id=searchbox_input").click()
            await page.keyboard.type(search_query, delay=100)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state('networkidle')
            page_content = await page.content()
            failed = False
        except Exception as e:
            await page.wait_for_timeout(5000)
            continue

    if failed:
        return "<search_results>\nNo search results.\n</search_results>"

    soup = BeautifulSoup(page_content, 'html.parser')
    raw_search_results = soup.find_all("li", {"data-layout": "organic"})
    results = []
    for raw_result in raw_search_results:
        try:
            header = raw_result.find("h2")
            link = header.find("a").get("href")
            description = raw_result.find("div", {"data-result": "snippet"}).text
            text_description = f"{header.text}\n\n{description}"
            results.append({
                "link": link,
                "description": text_description
            })
        except Exception as e:
            continue
    
    if len(results) == 0:
        return "<search_results>\nNo search results.\n</search_results>"

    return search_results_template.render(search_results=results)


@mcp.tool()
async def get_page_content(link: str) -> str:
    """
    Retrieves the text content of a webpage.

    Args:
        link (str): The URL of the webpage.
    
    Returns:
        The ext content of the web page at the URL provided.
    """

    playwright = await async_playwright().start()
    browser = await playwright.firefox.launch()
    page = await browser.new_page()

    await page.goto(link)
    await page.wait_for_timeout(3000)
    page_content = await page.content()
    soup = BeautifulSoup(page_content, 'html.parser')
    return soup.text

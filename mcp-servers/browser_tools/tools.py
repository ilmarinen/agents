import logging
import jinja2
from server import mcp
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Tag

_playwright = None
_browser = None
_page = None


logger = logging.getLogger("mcp.search")
logger.setLevel(logging.DEBUG)


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


async def _ensure_page():
    """Create/reuse a Playwright page on first use."""
    global _playwright, _browser, _page
    if _page is not None and not _page.is_closed():
        return _page
    if _playwright is None:
        _playwright = await async_playwright().start()
    if _browser is None:
        _browser = await _playwright.firefox.launch(headless=True)
    _page = await _browser.new_page()
    return _page


def dom_manifest(
    html: str,
    tags: list[str] | None = None,
    keep_attrs: list[str] = ("id",),
    *,
    keep_all_attrs: bool = False,
    exclude_attrs: tuple[str, ...] = (),
    prune: bool = True,
    skip_tags: tuple[str, ...] = ("script", "style", "template"),
    indent: str = "  ",
    include_positions: bool = False,
    as_json: bool = False,
    max_attr_value_len: int | None = None,  # optional safety/truncation
) -> str:
    """
    Produce a compact DOM manifest that encodes hierarchy.

    Nodes are kept if:
      (a) tag ∈ `tags`, OR
      (b) it has any attribute in `keep_attrs`, OR
      (c) any descendant is kept (to preserve ancestry), OR
      (d) prune=False (keep everything except `skip_tags`).

    `keep_all_attrs=True` makes the manifest include *all* attributes on kept nodes
    (except those in `exclude_attrs`). Otherwise it includes only `keep_attrs`.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = soup.body or soup

    def _interesting(el: Tag) -> bool:
        name_ok = (tags is None) or (el.name in tags)
        has_keep_attr = any(el.has_attr(k) for k in keep_attrs)
        return name_ok or has_keep_attr

    def _nth_of_type(el: Tag) -> int:
        i, sib = 0, el
        while True:
            sib = sib.previous_sibling
            if sib is None:
                break
            if isinstance(sib, Tag) and sib.name == el.name:
                i += 1
        return i + 1

    def _serialize_attr_value(v):
        # Normalize lists/tuples; keep bools; truncate long strings if requested
        if isinstance(v, (list, tuple)):
            v = " ".join(map(str, v))
        if isinstance(v, str) and max_attr_value_len is not None and len(v) > max_attr_value_len:
            v = v[: max_attr_value_len] + "…"
        return v

    def _attrs_for_node(el: Tag) -> dict:
        if keep_all_attrs:
            keys = [k for k in el.attrs.keys() if k not in exclude_attrs]
        else:
            keys = [k for k in keep_attrs if el.has_attr(k)]
        out = {}
        for k in keys:
            out[k] = _serialize_attr_value(el.attrs.get(k))
        return out

    def build(el: Tag):
        if not isinstance(el, Tag) or el.name in skip_tags:
            return None

        child_nodes = []
        for c in el.children:
            if isinstance(c, Tag):
                node = build(c)
                if node is not None:
                    child_nodes.append(node)

        keep_self = _interesting(el)
        keep = keep_self or bool(child_nodes) or not prune
        if not keep:
            return None

        attrs = _attrs_for_node(el)
        node = {
            "tag": el.name,
            **({"pos": _nth_of_type(el)} if include_positions else {}),
            **attrs,
        }
        if child_nodes:
            node["children"] = child_nodes
        return node

    tree = build(root)
    if tree is None:
        return "" if not as_json else json.dumps({})

    if as_json:
        return json.dumps(tree, separators=(",", ":"), ensure_ascii=False)

    # Render indented text
    lines = []
    def render(n: dict, depth: int = 0):
        tag = n["tag"]
        pos = f"[{n['pos']}]" if include_positions and "pos" in n else ""
        id_part = f"#{n['id']}" if "id" in n else ""
        # everything except structural keys and 'id' (printed separately)
        extras_parts = []
        for k, v in n.items():
            if k in {"tag", "children", "id", "pos"}:
                continue
            if isinstance(v, bool):
                if v:  # HTML-style boolean attr
                    extras_parts.append(k)
            else:
                extras_parts.append(f'{k}="{v}"')
        extras = (" " + " ".join(extras_parts)) if extras_parts else ""
        lines.append(f"{indent * depth}{tag}{pos}{id_part}{extras}")
        for child in n.get("children", []):
            render(child, depth + 1)

    render(tree)
    return "\n".join(lines)


@mcp.tool()
async def visit_page(page_url: str, max_retries: int = 3) -> str:
    """
    Visit the URL and parse the page HTML into a manifest of all the
    elements and their attributes. Return this DOM manifest.

    Args:
        page_url (str): The URL of the web page that you want to visit
        max_retries (int): The maximum number of times to re-try if there is a failure.
    
    Returns:
        A DOM manifest of all the page elements and their attributes.
    """
    logger.debug(f"Visit page: {page_url}")
    page = await _ensure_page()
    page_content = None
    for i in range(max_retries):
        try:
            await page.goto(page_url)
            await page.wait_for_load_state('networkidle')
            page_content = await page.content()
            return dom_manifest(page_content, keep_attrs=["id", "class", "data-layout", "data-result", "data-pos"])
        except Exception as e:
            logger.exception("Error loading page")
            await page.wait_for_timeout(5000)
            continue

    return f"<visit_page_error>Error loadibng page at {page_url}</visit_page_error>"



@mcp.tool()
async def search(search_box_id: str, search_query: str, max_retries: int = 3) -> str:
    """
    Clicks on the search box, enters and kicks off the search query,
    then parse the search results page HTML into a manifest of all the
    elements and their attributes. Return this DOM manifest.

    Args:
        search_box_id (str): The DOM element id of the search box element.
        search_query (str): A search query to enter into the Brave web search tool.
        max_retries (int): The maximum number of times to re-try if there is a failure.
    
    Returns:
        Returns a DOM manifest of the search results page.
    """
    logger.debug(f"Search: {search_box_id}, {search_query}")
    page = await _ensure_page()
    for i in range(max_retries):
        try:
            await page.locator(f"id={search_box_id}").nth(0).click()
            await page.keyboard.type(search_query, delay=100)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(10000)
            page_content = await page.content()
            return dom_manifest(page_content, keep_attrs=["id", "class", "data-layout", "data-result", "data-pos", "data-testid"])
        except Exception as e:
            logger.exception("Error searching")
            await page.wait_for_timeout(5000)
            continue

    return "<search_results>\nNo search results.\n</search_results>"


@mcp.tool()
async def search(search_box_id: str, search_query: str, max_retries: int = 3) -> str:
    """
    Clicks on the search box, enters and kicks off the search query,
    then parse the search results page HTML into a manifest of all the
    elements and their attributes. Return this DOM manifest.

    Args:
        search_box_id (str): The DOM element id of the search box element.
        search_query (str): A search query to enter into the Brave web search tool.
        max_retries (int): The maximum number of times to re-try if there is a failure.
    
    Returns:
        Returns a DOM manifest of the search results page.
    """
    logger.debug(f"Search: {search_box_id}, {search_query}")
    page = await _ensure_page()
    for i in range(max_retries):
        try:
            await page.locator(f"id={search_box_id}").nth(0).click()
            await page.keyboard.type(search_query, delay=100)
            await page.keyboard.press("Enter")
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(10000)
            page_content = await page.content()
            return dom_manifest(page_content, keep_attrs=["id", "class", "data-layout", "data-result", "data-pos", "data-testid"])
        except Exception as e:
            logger.exception("Error searching")
            await page.wait_for_timeout(5000)
            continue

    return "<search_results>\nNo search results.\n</search_results>"


@mcp.tool()
async def extract_search_results(
    search_results_tag: str,
    result_link_tag: str,
    result_description_tag: str,
    search_results_css_class: str | None = None,
    search_results_data_layout: str | bool | None = None,
    search_results_data_result: str | bool | None = None,
    search_results_data_pos: str | bool | None = None,
    result_link_css_class: str | None = None,
    result_link_data_testid: str | None = None,
    result_link_data_layout: str | bool | None = None,
    result_link_data_result: str | bool | None = None,
    result_link_data_pos: str | bool | None = None,
    result_description_css_class: str | None = None,
    result_description_data_testid: str | None = None,
    result_description_data_layout: str | bool | None = None,
    result_description_data_result: str | bool | None = None,
    result_description_data_pos: str | bool | None = None
    ) -> str:
    """
    Parses the HTML of the search results page.
    Uses the search_results_tag and search_results_* attributes to select
    the DOM elements which constitute the search results to select those
    DOM elements.
    We then use:
    1. The result_link_tag and the result_link_* attributes to
    select the DOM element within each search result that contains the
    search result link. It is important the the attributes are present
    on the tag you selected.
    2. And the result_description_tag and the result_description_*
    attributes to select the DOM element within each search result that
    contains the search result description. It is important that the attributes
    are present on the tag you selected.

    To extract a list of search results as links and descriptions.
    And return the list in an XML style string template.

    Args:
        search_results_tag (str): The HTML tag that constitutes each search result in the list of search results.
        search_results_css_class (str | None): The CSS class that is present on each search result element.
        search_results_data_layout (str | bool | None): The value of the data-layout attribute on the search result element if present, or just a boolean true if it is present.
        search_results_data_result (str | bool | None): The value of the data-result attribute on the search result element if present, or just a boolean true if it is present.
        search_results_data_pos (str | bool | None): The value of the data-pos attribute on the search result element if present, or just a boolean true if it is present.

        result_link_tag (str): The HTML tag within a single search result element that contains the link to the result.
        result_link_css_class (str | None): The CSS class that is present on each search result link element.
        result_link_data_layout (str | bool | None): The value of the data-layout attribute on the search result link element if present, or just a boolean true if it is present.
        result_link_data_result (str | bool | None): The value of the data-result attribute on the search result link element if present, or just a boolean true if it is present.
        result_link_data_pos (str | bool | None): The value of the data-pos attribute on the search result link element if present, or just a boolean true if it is present.

        result_description_tag (str): The HTML tag within a single search result element that contains the descriptiopn of the result.
        result_description_css_class (str | None): he CSS class that is present on each search result description element.
        result_description_data_layout (str | bool | None): The value of the data-layout attribute on the search result description element if present, or just a boolean true if it is present.
        result_description_data_result (str | bool | None): The value of the data-result attribute on the search result description element if present, or just a boolean true if it is present.
        result_description_data_pos (str | bool | None): The value of the data-pos attribute on the search result description element if present, or just a boolean true if it is present.
    
    Returns:
        A list of search results, with each search result consisting of a link and a description.
    """
    page = await _ensure_page()
    search_results_attributes = dict()
    result_link_attributes = dict()
    result_description_attributes = dict()
    
    if search_results_css_class is not None:
        search_results_attributes["class"] = search_results_css_class
    
    if search_results_data_layout is not None:
        search_results_attributes["data-layout"] = search_results_data_layout
    
    if search_results_data_result is not None:
        search_results_attributes["data-result"] = search_results_data_result

    if search_results_data_pos is not None:
        search_results_attributes["data-pos"] = search_results_data_pos
    
    if result_link_css_class is not None:
        result_link_attributes["class"] = result_link_css_class

    if result_link_data_testid is not None:
        result_link_attributes["data-testid"] = result_link_data_testid
    
    if result_link_data_layout is not None:
        result_link_attributes["data-layout"] = result_link_data_layout
    
    if result_link_data_result is not None:
        result_link_attributes["data-result"] = result_link_data_result

    if result_link_data_pos is not None:
        result_link_attributes["data-pos"] = result_link_data_pos
    
    if result_description_css_class is not None:
        result_description_attributes["class"] = result_description_css_class

    if result_description_data_testid is not None:
        result_description_attributes["data-testid"] = result_description_data_testid
    
    if result_description_data_layout is not None:
        result_description_attributes["data-layout"] = result_description_data_layout
    
    if result_description_data_result is not None:
        result_description_attributes["data-result"] = result_description_data_result

    if result_description_data_pos is not None:
        result_description_attributes["data-pos"] = result_description_data_pos
    
    logger.debug(f"Extract search results")
    page_content = await page.content()
    soup = BeautifulSoup(page_content, 'html.parser')
    try:
        raw_search_results = soup.find_all(search_results_tag, search_results_attributes)
    except Exception as e:
        logger.exception("Error extracting search results")
    results = []
    for raw_result in raw_search_results:
        try:
            header = raw_result.find(result_link_tag, result_link_attributes)
            link = header.get("href")
            description = raw_result.find(result_description_tag, result_description_attributes).text
            text_description = f"{header.text}\n\n{description}"
            results.append({
                "link": link,
                "description": text_description
            })
        except Exception as e:
            logger.exception("Error extracting searvch results while iterating")
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
    page = await _ensure_page()

    await page.goto(link)
    await page.wait_for_timeout(3000)
    page_content = await page.content()
    soup = BeautifulSoup(page_content, 'html.parser')
    return soup.text

# Use TavilySearchResults as a built-in LangChain tool

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

# ── Built-in Tavily tool ──────────────────────────────────────
# LangChain ships with a pre-built Tavily integration
tavily_tool = TavilySearchResults(
    max_results=4,
    api_key=os.getenv('TAVILY_API_KEY'),
    description=(
        'Search the web for current, real-time information. '
        'Use for: current events, prices, recent releases, live data. '
        'Do NOT use for mathematical calculations or static facts you already know.'
    ),
)


# ── Add domain-specific custom tool ──────────────────────────
@tool
def format_satellite_report(satellite_name: str, findings: str) -> str:
    """
    Format a structured satellite intelligence report.
    Use this LAST, after gathering all information.
    Provide the satellite name and all findings as a string.
    """
    return f'''
=== SATELLITE INTELLIGENCE REPORT ===
Satellite: {satellite_name}
Generated: {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

FINDINGS:
{findings}

END OF REPORT
===================================
'''

tools = [tavily_tool, format_satellite_report]
llm   = ChatOpenAI(model='gpt-4o-mini', temperature=0,
api_key=os.getenv('OPENAI_API_KEY'))
agent = create_agent(llm, tools, debug=True)

user_query = (
    "Search for the latest news about the James Webb Space Telescope observations,"
    " then compile a brief intelligence report about its most recent scientific findings."
)
result = agent.invoke({'messages': [HumanMessage(content=user_query)]})

print('\n=== FINAL ANSWER ===')
print(result['messages'][-1].content)

# Agent that remembers previous exchanges in the same conversation

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_satellite_info(norad_id: int) -> str:
    """Get basic info about a satellite by NORAD ID. ISS=25544, Hubble=20580."""
    db = {25544: 'ISS: 420km altitude, 92.7 min period',
          20580: 'Hubble: 547km altitude, 95.4 min period'}
    return db.get(norad_id, f'Satellite {norad_id} not found')

@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units: km<->miles, minutes<->hours, kms<->mph."""
    conversions = {
        ('km','miles'):1/1.60934, ('miles','km'):1.60934,
        ('minutes','hours'):1/60, ('hours','minutes'):60,
        ('kms','mph'):2236.94, ('mph','kms'):1/2236.94,
    }
    factor = conversions.get((from_unit.lower(), to_unit.lower()))
    if not factor:
        return f'Unknown conversion: {from_unit} to {to_unit}'
    return f'{value} {from_unit} = {round(value * factor, 4)} {to_unit}'

tools = [get_satellite_info, convert_units]
llm   = ChatOpenAI(model='gpt-4o-mini', temperature=0,
api_key=os.getenv('OPENAI_API_KEY'))


memory = ConversationBufferMemory(
    memory_key='chat_history',   # The variable name in the prompt template
    return_messages=True,
)

# Create the agent (use native structured tool-calling)
agent = create_agent(llm, tools, debug=True)


def build_messages(user_text: str):
    mem_vars = memory.load_memory_variables({})
    chat = mem_vars.get('chat_history')
    messages = []
    if chat:
        # `chat` may be a list of Message objects (when return_messages=True)
        if isinstance(chat, list):
            messages.extend(chat)
        else:
            messages.append(HumanMessage(content=str(chat)))
    messages.append(HumanMessage(content=user_text))
    return messages


def run_turn(user_text: str):
    messages = build_messages(user_text)
    result = agent.invoke({'messages': messages})
    # The final assistant message is usually the last message
    assistant_msg = result['messages'][-1].content
    try:
        memory.save_context({'input': user_text}, {'output': assistant_msg})
    except Exception:
        # Some memory implementations may differ; ignore if not available
        pass
    return assistant_msg


print('Turn 1:')
ans1 = run_turn('Tell me about the ISS satellite.')
print('Answer:', ans1)

print('\nTurn 2 (refers to previous context):')
ans2 = run_turn('What is its altitude in miles?')
print('Answer:', ans2)

print('\nTurn 3:')
ans3 = run_turn('How does that compare to the Hubble telescope?')
print('Answer:', ans3)

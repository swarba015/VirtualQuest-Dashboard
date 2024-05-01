from dotenv import load_dotenv
import os
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain.tools import DuckDuckGoSearchRun
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSerperAPIWrapper
import panel as pn  # GUI

# Load the environment variables from a .env file
try:
    load_dotenv(r'C:\Users\nuzha\OneDrive\Desktop\SI 568\ChatBot\requirement.txt')
except Exception as e:
    print(f"Failed to load environment variables: {e}")


# Retrieve the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    print("API key is not set. Check your .env file.")
else:
    print(api_key)#print(api_key) 

# Initialize tool wrappers
wikipedia = WikipediaAPIWrapper()
python_repl = PythonREPL()
search = DuckDuckGoSearchRun()
try:
    llm = OpenAI(temperature=0)
except Exception as e:
    print(f"Error initializing the language model: {e}")


# Panel GUI extension initialization
pn.extension()
panels = []




# Initialize tools with specific functionalities

def initialize_tools():
    """
    Initializes tools with associated functions and descriptions.

    Returns:
        list of Tool objects configured for different tasks.
    """
    tools = [
        Tool(
            name="python repl",
            func=python_repl.run,
            description="Useful for when you need to use python to answer a question. You should input python code"
        ),
        Tool(
            name='wikipedia',
            func=wikipedia.run,
            description="Useful for when you need to look up a topic, country, or person on Wikipedia"
        ),
        Tool(
            name='DuckDuckGo Search',
            func=search.run,
            description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input."
         )
     ]

    return tools

try:
    tools = initialize_tools()
except Exception as e:
    print(f"Failed to initialize tools: {e}")
memory = ConversationBufferMemory (memory_key="chat_history")

# Initialize the zero-shot agent with tools and large language model
try:
    zero_shot_agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=llm,
        verbose=False, 
        memory=memory,
        handle_parsing_errors=True
    )
except Exception as e:
    print(f"Failed to initialize the agent: {e}")


# GUI setup
text_input = pn.widgets.TextInput(value="", placeholder='Enter your input here...')
button_run = pn.widgets.Button(name="Run", button_type="primary")
output_area = pn.pane.Markdown("Output will appear here")

output_area.css_classes = ['custom-background']

def on_button_click(event):
    """
    Handle button click event to process input using the zero-shot agent.

    Args:
        event: The event object representing the button click.
    """
    try:
        output = zero_shot_agent.run(text_input.value)
        output_area.object = output
    except Exception as e:
        output_area.object = f"An error occurred while processing your request: {e}"


button_run.on_click(on_button_click)

css = """
.custom-background {
    background-color: #f0f0f0;
    padding: 10px;
    border-radius: 5px;
}
"""

pn.config.raw_css.append(css)

dashboard = pn.Column(
    "# Tool Runner Dashboard",
    pn.Row(text_input, button_run),
    output_area
)

# Main execution guard
if __name__ == "__main__":
    try:
        pn.serve(dashboard)
    except Exception as e:
        print(f"Failed to start the server: {e}")


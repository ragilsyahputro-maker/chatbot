import streamlit as st 
import os 
import pandas as pd
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# --- 1. Page Configuration and Title ---
from database_tools import text_to_sql, init_database, get_database_info
# Set the title and a caption for the web page
st.title("💬 Pelayanan Chatbot")
st.caption("chat menjadi lebih mudah dengan pelayanan chatbot")

# --- 2. Sidebar for Settings ---

# Create a sidebar section for app settings using 'with st.sidebar:'
with st.sidebar:
    # Add a subheader to organize the settings
    st.subheader("Settings")
    st.subheader("[Google](https://google.com)")
    
    # Create a text input field for the Google AI API Key.
    # 'type="password"' hides the key as the user types it.
    google_api_key = st.text_input("Google AI API Key", type="password")
    
    # Create a button to reset the conversation.
    # 'help' provides a tooltip that appears when hovering over the button.

    link_button = st.button("Google", help="google")

    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

    init_db_button = st.button("Initialize Database", help="Create and populate the database with sample data")



if init_db_button:
        with st.spinner("Initializing database..."):
            result = init_database()
            st.success(result)

# --- 3. API Key and Client Initialization ---

# Check if the user has provided an API key.
# If not, display an informational message and stop the app from running further.
if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()



@tool
def execute_sql(sql_query: str):
    """
    Execute a SQL query against the sales database.
    
    Args:
        sql_query: The SQL query to execute. Must be a valid SQL query string.
              For example: "SELECT * FROM customers", "SELECT p.name, SUM(si.quantity) as total_sold FROM sale_items si JOIN products p ON si.product_id = p.product_id GROUP BY p.product_id ORDER BY total_sold DESC", etc.
    """
    result = text_to_sql(sql_query)
    # Format the result to clearly show the executed SQL query
    formatted_result = f"```sql\n{sql_query}\n```\n\nQuery Results:\n{result}"
    return formatted_result

@tool
def get_schema_info():
    """
    Get information about the database schema and sample data to help with query construction.
    This tool returns the schema of all tables and sample data (first 3 rows) from each table.
    Use this tool before writing SQL queries to understand the database structure.
    """
    return get_database_info()

# This block of code handles the creation of the Gemini API client.
# It's designed to be efficient: it only creates a new client if one doesn't exist
# or if the user has changed the API key in the sidebar.

# We use `st.session_state` which is Streamlit's way of "remembering" variables
# between user interactions (like sending a message or clicking a button).

# Condition 1: "genai_client" not in st.session_state
# Checks if we have *never* created the client before.
#
# Condition 2: getattr(st.session_state, "_last_key", None) != google_api_key
# This is a safe way to check if the current API key is different from the last one we used.
# `getattr(object, 'attribute_name', default_value)` tries to get an attribute from an object.
# If the attribute doesn't exist, it returns the default value (in this case, `None`).
# So, it checks: "Is the key stored in memory different from the one in the input box?"
if ("genai_client" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        # If the conditions are met, create a new client.
        st.session_state.genai_client = genai.Client(api_key=google_api_key)
        # Store the new key in session state to compare against later.
        st.session_state._last_key = google_api_key
        # Since the key changed, we must clear the old chat and message history.
        # .pop() safely removes an item from session_state.
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
       
    except Exception as e:
        # If the key is invalid, show an error and stop.
        st.error(f"Invalid API Key: {e}")
        st.stop()

if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        # Initialize the LLM with the API key
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        
        # Create a simple ReAct agent with the LLM
        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[get_schema_info, execute_sql],
            prompt="""You are a helpful assistant that can answer questions about sales data using SQL.
            
            IMPORTANT: When a user asks a question about sales data, follow these steps:
            1. FIRST, use the get_schema_info tool to understand the database structure and see sample data
            2. THEN, write a SQL query based on the user's question and the database schema
            3. Execute the SQL query using the execute_sql tool
            4. Explain the results in a clear and concise way
            
            When writing SQL queries:
            - Use proper SQL syntax for SQLite
            - Use appropriate JOINs when querying across multiple tables
            - Use aliases for table names in complex queries (e.g., 'customers AS c')
            - Use aggregation functions (COUNT, SUM, AVG, etc.) when appropriate
            - Format the SQL query to be readable
            
            If you encounter any errors:
            - Explain what went wrong
            - Fix the SQL query and try again
            
            Remember: You must generate the SQL query yourself based on the user's question and the database schema.
            Do not ask the user to provide SQL queries.
            """
        )
        
        st.session_state._last_key = google_api_key
        # Since the key changed, we must clear the old message history.
        st.session_state.pop("messages", None)
    except Exception as e:
        # If the key is invalid, show an error and stop.
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()


# --- 4. Chat History Management ---

# Initialize the chat session if it doesn't already exist in memory.
if "chat" not in st.session_state:
    # Create a new chat instance using the 'gemini-2.5-flash' model.
    st.session_state.chat = st.session_state.genai_client.chats.create(model="gemini-2.5-flash")

# Initialize the message history (as a list) if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle the reset button click.
if reset_button:
    # If the reset button is clicked, clear the chat object and message history from memory.
    st.session_state.pop("agent", None)
    st.session_state.pop("chat", None)
    st.session_state.pop("messages", None)
    # st.rerun() tells Streamlit to refresh the page from the top.
    st.rerun()

# --- 5. Display Past Messages ---

# Loop through every message currently stored in the session state.
for msg in st.session_state.messages:
    # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
    with st.chat_message(msg["role"]):
        # Display the content of the message using Markdown for nice formatting.
        st.markdown(msg["content"])

# --- 6. Handle User Input and API Communication ---




# Create a chat input box at the bottom of the page.
# The user's typed message will be stored in the 'prompt' variable.
prompt = st.chat_input("Type your message here...")

# Check if the user has entered a message.
if prompt:
    # 1. Add the user's message to our message history list.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Display the user's message on the screen immediately for a responsive feel.
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Get the assistant's response.
    # Use a 'try...except' block to gracefully handle potential errors (e.g., network issues, API errors).
    try:
        # Convert the message history to the format expected by the agent
        messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Show a spinner while waiting for the response
        with st.spinner("Thinking..."):
            # Send the user's prompt to the agent
            response = st.session_state.agent.invoke({"messages": messages})
            
            # Extract the answer from the response
            if "messages" in response and len(response["messages"]) > 0:
                answer = response["messages"][-1].content
                
                # Extract SQL query from tool messages if present
                sql_query = None
                for i, msg in enumerate(response["messages"]):
                    # Check if this is a ToolMessage with execute_sql
                    if hasattr(msg, "tool_call_id") and hasattr(msg, "name") and msg.name == "execute_sql":
                        # Extract SQL query from the tool message content
                        if hasattr(msg, "content") and "```sql\n" in msg.content:
                            sql_parts = msg.content.split("```sql\n")
                            if len(sql_parts) > 1:
                                sql_query = sql_parts[1].split("\n```")[0].strip()
                                # Store the SQL query in session state for display
                                st.session_state.last_sql_query = sql_query
                    # Also check for tool calls in AIMessage
                    elif hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call.get("name") == "execute_sql" and "sql_query" in tool_call.get("args", {}):
                                sql_query = tool_call["args"]["sql_query"]
                                st.session_state.last_sql_query = sql_query
            else:
                answer = "I'm sorry, I couldn't generate a response."

    except Exception as e:
        # If any error occurs, create an error message to display to the user.
        answer = f"An error occurred: {e}"

    # 4. Display the assistant's response.
    with st.chat_message("assistant"):
        sql_query = None
        if hasattr(st.session_state, "last_sql_query"):
            sql_query = st.session_state.last_sql_query
            # Clear it after use
            del st.session_state.last_sql_query

        # Display the extracted SQL query in a code block if found
        if sql_query:
            st.code(sql_query, language="sql")
        st.markdown(answer)
    # 5. Add the assistant's response to the message history list.
    st.session_state.messages.append({"role": "assistant", "content": answer})
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_openai import ChatOpenAI


# Initialize the SQL database connection using provided credentials
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    # Construct the database URI
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    # Return a SQLDatabase object using the constructed URI
    return SQLDatabase.from_uri( db_uri )


# Define a function to create a chain for generating SQL queries based on user input and database schema
def get_sql_chain(db):
    # Template for generating SQL queries based on user questions and database schema
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;

    Your turn:

    Question: {question}
    SQL Query:
    """

    # Create a ChatPromptTemplate from the template
    prompt = ChatPromptTemplate.from_template( template )

    # Initialize the OpenAI model (GPT-4)
    llm = ChatOpenAI( model="gpt-4", temperature=0 )  # Using OpenAI's GPT-4 model

    # Function to retrieve the database schema
    def get_schema(_):
        return db.get_table_info()

    # Create and return the SQL generation chain
    return (
            RunnablePassthrough.assign( schema=get_schema )  # Assign the schema
            | prompt  # Pass the prompt to the LLM
            | llm  # Generate the SQL query
            | StrOutputParser()  # Parse the output as a string
    )


# Define a function to generate a response based on user query, database schema, and conversation history
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    # Get the SQL chain to generate the SQL query
    sql_chain = get_sql_chain( db )

    # Template for generating a natural language response based on the SQL query and database response
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""

    # Create a ChatPromptTemplate from the template
    prompt = ChatPromptTemplate.from_template( template )

    # Initialize the OpenAI model (GPT-4)
    llm = ChatOpenAI( model="gpt-4", temperature=0 )  # Using OpenAI's GPT-4 model

    # Create and return the response generation chain
    chain = (
            RunnablePassthrough.assign( query=sql_chain ).assign(
                schema=lambda _: db.get_table_info(),  # Assign the database schema
                response=lambda vars: db.run( vars["query"] ),  # Run the SQL query and get the response
            )
            | prompt  # Pass the prompt to the LLM
            | llm  # Generate the natural language response
            | StrOutputParser()  # Parse the output as a string
    )

    # Invoke the chain with the user's question and conversation history, and return the response
    return chain.invoke( {
        "question": user_query,
        "chat_history": chat_history,
    } )


# Ensure chat_history is initialized in Streamlit's session_state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage( content="Hello! I'm a SQL assistant. Ask me anything about your database." ),
    ]

# Load environment variables from a .env file
load_dotenv()

# Set up the Streamlit page configuration
st.set_page_config( page_title="Chat with MySQL", page_icon=":speech_balloon:" )

# Title for the Streamlit app
st.title( "Chat with MySQL" )

# Sidebar settings for database connection
with st.sidebar:
    st.subheader( "Settings" )
    st.write( "This is a simple chat application using MySQL. Connect to the database and start chatting." )

    # Input fields for database connection details
    st.text_input( "Host", value="localhost", key="Host" )
    st.text_input( "Port", value="3306", key="Port" )
    st.text_input( "User", value="root", key="User" )
    st.text_input( "Password", type="password", value="admin", key="Password" )
    st.text_input( "Database", value="Chinook", key="Database" )

    # Button to connect to the database
    if st.button( "Connect" ):
        with st.spinner( "Connecting to database..." ):
            # Initialize the database connection
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            # Store the database connection in session_state
            st.session_state.db = db
            st.success( "Connected to database!" )

# Display the conversation history in the chat interface
for message in st.session_state["chat_history"]:
    if isinstance( message, AIMessage ):
        with st.chat_message( "AI" ):
            st.markdown( message.content )
    elif isinstance( message, HumanMessage ):
        with st.chat_message( "Human" ):
            st.markdown( message.content )

# Input field for the user's query
user_query = st.chat_input( "Type a message..." )
if user_query is not None and user_query.strip() != "":
    # Append the user's query to the chat history
    st.session_state["chat_history"].append( HumanMessage( content=user_query ) )

    # Display the user's query in the chat interface
    with st.chat_message( "Human" ):
        st.markdown( user_query )

    # Generate and display the AI response based on the user's query
    with st.chat_message( "AI" ):
        response = get_response( user_query, st.session_state.db, st.session_state["chat_history"] )
        st.markdown( response )

    # Append the AI's response to the chat history
    st.session_state["chat_history"].append( AIMessage( content=response ) )

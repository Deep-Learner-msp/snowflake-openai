import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from analyze import AnalyzeGPT, SQL_Query, ChatGPT_Handler
import openai
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain import OpenAI, SerpAPIWrapper,GoogleSerperAPIWrapper
from langchain.chat_models import AzureChatOpenAI
import openai
import pandas as pd
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent,initialize_agent,AgentType,create_pandas_dataframe_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
import re
import datetime

# Only load the settings if they are running local and not in Azure
if os.getenv('WEBSITE_SITE_NAME') is None:
    env_path = Path('.') / 'secrets.env'
    load_dotenv(dotenv_path=env_path)

def load_setting(setting_name, session_name,default_value=''):  
    """  
    Function to load the setting information from session  
    """  
    if session_name not in st.session_state:  
        if os.environ.get(setting_name) is not None:
            st.session_state[session_name] = os.environ.get(setting_name)
        else:
            st.session_state[session_name] = default_value  




load_setting("AZURE_OPENAI_CHATGPT_DEPLOYMENT","chatgpt","ss-gpt")  
load_setting("AZURE_OPENAI_GPT4_DEPLOYMENT","gpt4","ss-gpt-32k")  
load_setting("AZURE_OPENAI_ENDPOINT","endpoint","https://openai-ss.openai.azure.com/")  
load_setting("OPENAI_API_KEY","apikey","be51f10009fa41258fcd750a2fba07f2")  
load_setting("SNOW_ACCOUNT", "snowaccount","liquxks-tx06668")
load_setting("SNOW_USER", "snowuser","SNOWFLAKEDEMO")
load_setting("SNOW_PASSWORD", "snowpassword","Snowflake123")
load_setting("SNOW_ROLE", "snowrole","ACCOUNTADMIN")
load_setting("SNOW_DATABASE", "snowdatabase","PORTFOLIO_DEMO")
load_setting("SNOW_SCHEMA", "snowschema","YAHOO")
load_setting("SNOW_WAREHOUSE", "snowwarehouse","COMPUTE_WH")

if "show_settings" not in st.session_state:
    st.session_state["show_settings"] = False


def saveOpenAI():
    st.session_state.chatgpt = st.session_state.txtChatGPT
    st.session_state.gpt4 = st.session_state.txtGPT4
    st.session_state.endpoint = st.session_state.txtEndpoint
    st.session_state.apikey = st.session_state.txtAPIKey
    st.session_state.snowaccount = st.session_state.txtSNOWAccount
    st.session_state.snowuser = st.session_state.txtSNOWUser
    st.session_state.snowpassword = st.session_state.txtSNOWPasswd
    st.session_state.snowrole = st.session_state.txtSNOWRole
    st.session_state.snowdatabase = st.session_state.txtSNOWDatabase
    st.session_state.snowschema = st.session_state.txtSNOWSchema
    st.session_state.snowwarehouse = st.session_state.txtSNOWWarehouse

    # We can close out the settings now
    st.session_state["show_settings"] = False


def toggleSettings():
    st.session_state["show_settings"] = not st.session_state["show_settings"]


openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_key = st.session_state.apikey
openai.api_base = st.session_state.endpoint
max_response_tokens = 2000
token_limit = 10000
temperature = 0.2

st.set_page_config(page_title="AlphaData Hub", page_icon="images/alpha_new_logo.png", layout="wide")


col1, col2 = st.columns([2, 5])

m = st.markdown("""
<style>
.logo {
    position: relative;
    top: 1rem;
    left: 1rem;
    z-index: 100;
}
.css-10trblm {
    color: #0A2F5D !important;
}
.css-9ycgxx {
    color: #0A2F5D !important;
}
div.stButton > button:first-child {
    background-color: #004990;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #0A2F5D;
    color:##ff99ff;
}
.logo-container {
    display: flex;
    justify-content: center;
    gap: 50px; /* Adjust this value for the desired gap */
}
.snowflake-logo {
    width: 250px; /* Adjust this value to change the size of the Snowflake logo */
    height: auto;
}
.tagline {
    font-size: 1.5em; /* Reduce the font size if necessary */
    color: #0A2F5D;
    text-align: center;
    white-space: nowrap; /* This will prevent the tagline from wrapping */
    # overflow: hidden; /* This will hide any overflow */
    text-overflow: ellipsis;
}
</style>""", unsafe_allow_html=True)

# Load the Snowflake SVG
with open('images/snowflake-ar21.svg', 'r') as f:
    snowflake_logo = f.read()
with col1:
    # Display the logos and tagline
    st.markdown(
        f"""
        <div class="logo-container">
            <div class="logo">
                <img class="cmp-image__image" src="https://www.crd.com/wp-content/uploads/2021/06/STT-Alpha_Logo.png" alt="State Street Global Advisors" width="200" height="55">
            </div>
            <div class="logo snowflake-logo">
                {snowflake_logo}
            </div>
        </div>
        <h2 class="tagline">State Street x Snowflake: Reshaping Investment Landscapes with Scalable Data Solutions</h2>
        """, unsafe_allow_html=True
    )

gpt_engine = st.radio("**GPT Model**", ("ChatGPT", "GPT4"))
if gpt_engine == "ChatGPT":
    gpt_engine = st.session_state.chatgpt
else:
    gpt_engine = st.session_state.gpt4

system_message = """
        You are a smart AI assistant to help answer business questions based on analyzing data. 
        You can plan solving the question with one more multiple thought step. At each thought step, you can write python code to analyze data to assist you. Observe what you get at each step to plan for the next step.
        You are given following utilities to help you retrieve data and communicate your result to end user.
        1. execute_sql(sql_query: str): A Python function can query data from the Snowflake given a query which you need to create. The query has to be syntactically correct for Snowflake and only use tables and columns under <<data_sources>>. The execute_sql function returns a Python pandas dataframe with UPPERCASE COLUMN NAMES and  contain the results of the query.
        2. Use plotly library for data visualization. 
        3. Use observe(label: str, data: any) utility function to observe data under the label for your evaluation. Use observe() function instead of print() as this is executed in streamlit environment. Due to system limitation, you will only see the first 10 rows of the dataset.
        4. To communicate with user, use show() function on data, text and plotly figure. show() is a utility function that can render different types of data to end user. Remember, you don't see data with show(), only user does. You see data with observe()
            - If you want to show  user a plotly visualization, then use ```show(fig)`` 
            - If you want to show user data which is a text or a pandas dataframe or a list, use ```show(data)```
            - Never use print(). User don't see anything with print()
        5. Lastly, don't forget to deal with data quality problem. You should apply data imputation technique to deal with missing data or NAN data.
        6. Always follow the flow of Thought: , Observation:, Action: and Answer: as in template below strictly. 

        """

few_shot_examples = """
<<Template>>
Question: User Question
Thought 1: Your thought here.
Action: 
```python
#Import neccessary libraries here
import numpy as np
#Query some data 
sql_query = "SOME SQL QUERY"
step1_df = execute_sql(sql_query)
step1_df.columns = step1_df.columns.str.upper()
# Replace 0 with NaN. Always have this step
step1_df['Some_Column'] = step1_df['Some_Column'].replace(0, np.nan)
#observe query result
observe("some_label", step1_df) #Always use observe() instead of print
## you can talk about the data based on user question and explain the data.
```
Observation: 
step1_df is displayed here
Thought 2: Your thought here
Action:  
```python
import plotly.express as px 
#from step1_df, perform some data analysis action to produce step2_df
#To see the data for yourself the only way is to use observe()
observe("some_label", step2_df) #Always use observe() 
#Decide to show it to user.
fig=px.line(step2_df)
#visualize fig object to user.  
show(fig)
#you can also directly display tabular or text data to end user.
show(step2_df)
```
Observation: 
step2_df is displayed here
Answer: Your final detailed descriptive answer and comment for the question
<</Template>>

"""

extract_patterns = [
    ("Thought:", r"(Thought \d+):\s*(.*?)(?:\n|$)"),
    ("Action:", r"```python\n(.*?)```"),
    ("Answer:", r"([Aa]nswer:) (.*)"),
    ]
extractor = ChatGPT_Handler(extract_patterns=extract_patterns)

model = AzureChatOpenAI(temperature=0,deployment_name=gpt_engine)




def create_db_connection():
    db_user="SNOWFLAKEDEMO"
    db_password="Snowflake123"
    account_identifier="liquxks-tx06668"
    db_name="PORTFOLIO_DEMO"
    db_schema="YAHOO"
    db_warehouse="COMPUTE_WH"
    db_role="ACCOUNTADMIN"

    db = SQLDatabase.from_uri(f"snowflake://{db_user}:{db_password}@{account_identifier}/{db_name}/{db_schema}?warehouse={db_warehouse}&role={db_role}")
    
    return db
db = create_db_connection()


def snowflake_agent(user_query):
    toolkit = SQLDatabaseToolkit(db=create_db_connection(), llm=model)

    agent_executor = create_sql_agent(llm=model, toolkit=toolkit, verbose=True)

    response = agent_executor.run(user_query)
    return response

def plot_data_agent(user_query):
    sql_query_tool = SQL_Query(db_user="SNOWFLAKEDEMO",
                    db_password="Snowflake123",
                    account_identifier="liquxks-tx06668",
                    db_name="PORTFOLIO_DEMO",
                    db_schema="YAHOO",
                    db_warehouse="COMPUTE_WH",
                    db_role="ACCOUNTADMIN")
    analyzer = AnalyzeGPT(
        content_extractor=extractor,
        sql_query_tool=sql_query_tool,
        system_message=system_message,
        few_shot_examples=few_shot_examples,
        st=st,
        gpt_deployment=gpt_engine,
        max_response_tokens=max_response_tokens,
        token_limit=token_limit,
        temperature=temperature,
        db_schema="YAHOO"
    )

    response = analyzer.run(user_query,False,False,st)
    return response

def ask_bot(user_query):
    messages = [
    SystemMessage(content="You are a skilled portfolio manager responsible for portfolio analysis, report generation, conducting research on the latest finance news, and creating visualizations. Your role involves analyzing and evaluating various investment portfolios, including mutual funds and stocks. You gather and analyze financial data, generate insightful reports, and provide recommendations to optimize portfolio performance. Staying up-to-date with market trends and economic news, you conduct thorough research to identify investment opportunities. Additionally, you develop user-friendly visualizations and interactive tools."),
    HumanMessage(content=user_query)]
    response = model(messages)
    return response


# search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name = "ask_bot",
        func=ask_bot,
        description="call this to run only when user asks about questions like 'who are you?' and 'how do you work?' and strat with greeting the user with his/her name"
    
    ),
    
    Tool(
        name = "snowflake_agent",
        func=snowflake_agent,
        description="call this to run only the sql queries on top of the DB connection that established"
    
    ),
    Tool(
        name = "plot_data_agent",
        func=plot_data_agent,
        description="call this for all visualisation queries, and don't use action, action input template for this, just use the tool as it is",return_direct=True
    )
]
    
def run(user_query):
    agent = initialize_agent(tools,model,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,handle_parsing_errors=True)
    response = agent.run(user_query)
    return response

if __name__ == "__main__":
    run(input("enter ur query"))

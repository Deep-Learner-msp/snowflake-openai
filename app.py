import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from analyze import AnalyzeGPT, SQL_Query, ChatGPT_Handler
import openai
from pathlib import Path
from dotenv import load_dotenv
import os
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
load_setting("AZURE_OPENAI_API_KEY","apikey","be51f10009fa41258fcd750a2fba07f2")  
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

# with st.sidebar:
#     options = ("SQL Assistant(Statistical)", "Data Analysis Assistant(Descriptive)")
#     index = st.radio(
#         "**Choose the app**", range(len(options)), format_func=lambda x: options[x]
#     )
#     if index == 0:
#         system_message = """
#         You are an agent designed to interact with a Snowflake with schema detail in Snowflake.
#         Given an input question, create a syntactically correct Snowflake query to run, then look at the results of the query and return the answer.
#         You can order the results by a relevant column to return the most interesting data in the database.
#         Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
#         You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
#         DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
#         Remember to format SQL query as in ```sql\n SQL QUERY HERE ``` in your response.

#         """
#         few_shot_examples = ""
#         extract_patterns = [("sql", r"```sql\n(.*?)```")]
#         extractor = ChatGPT_Handler(extract_patterns=extract_patterns)

#         faq_dict = {
#             "Annualized returns of ETFs for different amounts of years",
#             "Annualized returns of Mutual funds for different amouts of years",
#             "Split of ETF funds in investment type",
#             "Split of Total net assets in investment type",
#             "Split of Mutual funds in investment type",
#             "ETFs: Correlation of returns and volatility",
#             "MutualFunds Correlation of returns and volatility" ,
        # }

    # elif index == 1:
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
    

    # st.button("Settings", on_click=toggleSettings)
    if st.session_state["show_settings"]:
        with st.form("AzureOpenAI"):
            st.title("Azure OpenAI Settings")
            st.text_input(
                "ChatGPT deployment name:",
                key="txtChatGPT",
                value=st.session_state.chatgpt,
                help="Enter the name of ChatGPT deployment from Azure OpenAI",
            )
            st.text_input(
                "GPT-4 deployment name",
                key="txtGPT4",
                value=st.session_state.gpt4,
                help="Enter the GPT-4 deployment in Azure OpenAI. Defaults to above value if not specified",
            )
            st.text_input(
                "Azure OpenAI Endpoint:",
                key="txtEndpoint",
                value=st.session_state.endpoint,
                help="Enter the Azure Open AI Endpoint",
                placeholder="https://<endpointname>.openai.azure.com/",
            )
            st.text_input(
                "Azure OpenAI Key:",
                type="password",
                key="txtAPIKey",
                value=st.session_state.apikey,
                help="Enter Azure OpenAI Key",
            )

            st.title("Snowflake Settings")
            st.text_input(
                "Account Identifier:",
                key="txtSNOWAccount",
                value=st.session_state.snowaccount,
                help="Enter Snowflake Account Identifier. Do not enter with .snowflakecomputing.com",
                placeholder="<orgname>-<accountname>",
            )
            st.text_input(
                "User Name:", key="txtSNOWUser",value=st.session_state.snowuser, help="Enter Snowflake Username"
            )
            st.text_input(
                "Password:",
                type="password",
                key="txtSNOWPasswd",
                value=st.session_state.snowpassword,
                help="Enter Snowflake Password",
            )
            st.text_input("Role:", key="txtSNOWRole",value=st.session_state.snowrole, help="Enter Snowflake role")
            st.text_input(
                "Database:", key="txtSNOWDatabase",value=st.session_state.snowdatabase, help="Enter Snowflake Database"
            )
            st.text_input("Schema:", key="txtSNOWSchema",value=st.session_state.snowschema, help="Enter Snowflake Schema")
            st.text_input(
                "Warehouse:", key="txtSNOWWarehouse",value=st.session_state.snowwarehouse, help="Enter Snowflake Warehouse"
            )

            st.form_submit_button("Submit", on_click=saveOpenAI)

    chat_list = []
    if st.session_state.chatgpt != "":
        chat_list.append("ChatGPT")
    if st.session_state.gpt4 != "":
        chat_list.append("GPT-4")
    gpt_engine = st.radio("**GPT Model**", ("ChatGPT", "GPT4"))
    if gpt_engine == "ChatGPT":
        gpt_engine = st.session_state.chatgpt
    else:
        gpt_engine = st.session_state.gpt4
col1, col2 = st.columns([1, 1])  # Divide the page into two equal columns
with col1:
    question = st.text_area(" **Ask me a question**")

if st.button("Submit"):
    if (
        st.session_state.apikey == ""
        or st.session_state.endpoint == ""
        or st.session_state.chatgpt == ""
    ):
        st.error("You need to specify Azure Open AI Deployment Settings!")
    elif (
        st.session_state.snowaccount == ""
        or st.session_state.snowuser == ""
        or st.session_state.snowpassword == ""
        or st.session_state.snowrole == ""
    ):
        st.error("You need to specify Snowflake Settings!")
    else:
        sql_query_tool = SQL_Query(
            account_identifier=st.session_state.snowaccount,
            db_user=st.session_state.snowuser,
            db_password=st.session_state.snowpassword,
            db_role=st.session_state.snowrole,
            db_name=st.session_state.snowdatabase,
            db_schema=st.session_state.snowschema,
            db_warehouse=st.session_state.snowwarehouse
        )
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
            db_schema=st.session_state.snowschema
        )
        try:
            with st.spinner("Unlocking Data's Secrets: Analyzing... Hold on for 30-40 seconds of Data Wizardry!"):
                analyzer.run(question, False, False, st)  # Use st instead of col1
        except:
            st.error("Not implemented yet!")


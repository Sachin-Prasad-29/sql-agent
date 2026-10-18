from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import os
load_dotenv()

# db = SQLDatabase.from_uri(os.getenv("DB_URI"))
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK(status IN('pending', 'completed', 'in-progress')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")

model = ChatGroq(model='openai/gpt-oss-20b')

toolkit = SQLDatabaseToolkit(db=db,llm=model)

tools = toolkit.get_tools()

memory = InMemorySaver()


system_prompt = """
You are a task-management assistant. You interact only with the SQL database table named `tasks`.

Table schema:
- id: unique task identifier
- title: task title
- description: task details
- status: one of `pending`, `progress`, or `completed`
- created_at: task creation timestamp

Rules:
1. Use only the `tasks` table.
2. For task lists, return at most 10 rows, ordered by `created_at DESC`.
3. When listing tasks, present results as a clear Markdown table with:
   `id`, `title`, `description`, `status`, and `created_at`.
4. For a new task, create it with status `pending` unless the user explicitly provides a valid status.
5. Before updating or deleting a task, identify it by `id` whenever possible.
6. If a title matches multiple tasks, ask the user to choose the task ID; do not update or delete multiple tasks unless the user explicitly asks.
7. After creating, updating, or deleting a task, run a SELECT query to verify the result and report the outcome clearly.
8. Never invent task data or claim a database change succeeded unless the verification query confirms it.
9. If a request is unclear, ask a short clarifying question.
"""

@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent = get_agent()

st.subheader("Task Management Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me to manage your tasks")


if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message('ai'):
        with st.spinner("Thinking..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": "1"}}
            )
            result = response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role": "ai", "content": result})


 
import os
import time
import json
import logging
import datetime
from dotenv import load_dotenv

# Flask imports
from flask import Flask, render_template, request, Response, stream_with_context

# LangChain imports
from langchain.agents import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


# Load environment variables from a .env file
load_dotenv(override=True)

# Set up logging in the app.log file
log = logging.getLogger("assistant")
logging.basicConfig(filename="app.log", level=logging.INFO)

# Import and configure OpenAI
from langchain_openai import ChatOpenAI
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

chat_llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


# Flask app setup
app = Flask(__name__)

# Define MemorySaver instance for langgraph agent
memory = MemorySaver()


# Routes
# Index route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Serve the chat interface

# Stream route
@app.route("/stream", methods=["GET"])
def stream():

    graph = create_react_agent(
        model=chat_llm,
        tools=[],
        checkpointer=memory,
        debug=True
    )

    inputs = {"messages": [("user", request.args.get("query", ""))]}
    config = {"configurable": {"thread_id": "thread-1"}}

    HEARTBEAT_INTERVAL = 5

    @stream_with_context
    def generate():
        stream_iterator = graph.stream(inputs, config, stream_mode="values")
        last_sent_time = time.time()

        while True:
            # Check if we've been idle too long
            if time.time() - last_sent_time > HEARTBEAT_INTERVAL:
                # Send a heartbeat
                yield "data: [heartbeat]\n\n"
                last_sent_time = time.time()

            try:
                step = next(stream_iterator)
            except StopIteration:
                # No more data from the agent
                break
            except Exception as e:
                # On any exception, report it and stop
                yield f"data: Error: {str(e)}\n\n"
                return

            # We got a new message from the agent
            message = step["messages"][-1]
            if isinstance(message, tuple):
                pass
                # yield f"data: {message[1]}\n\n" # Uncomment to allow for user messages to be displayed
            else:
                if message.response_metadata.get("finish_reason") == "stop":
                    escaped_message = json.dumps(message.content)
                    yield f"data: {escaped_message}\n\n"
                    break
            last_sent_time = time.time()

        # Final marker
        yield "data: [DONE]\n\n"

    return Response(
        generate(),
        content_type="text/event-stream"
    )

# Function to add to the log in the app.log file
def log_run(run_status):
    if run_status in ["cancelled", "failed", "expired"]:
        log.error(str(datetime.datetime.now()) + " Run " + run_status + "\n")

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
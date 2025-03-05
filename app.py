import os
import time
import json
import logging
import datetime
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy

# Flask login imports
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Flask imports
from flask import Flask, render_template, request, redirect, url_for, flash, Response, stream_with_context

# Werkzeug imports
from werkzeug.security import generate_password_hash, check_password_hash

# Supabase imports
from supabase import create_client
from supabase.client import ClientOptions

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
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "mysecret")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("SUPABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

	
# Database setup
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin, db.Model):
   id = db.Column(db.Integer, primary_key=True)
   username = db.Column(db.String(150), unique=True, nullable=False)
   email = db.Column(db.String(150), unique=True, nullable=False)
   password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
   return db.session.get(User, int(user_id))

# Initialize Supabase and LangChain components
supabase_https_url = os.getenv("SUPABASE_HTTPS_URL")
supabase_key = os.getenv("SUPABASE_KEY")
 
supabase_client = create_client(supabase_https_url, supabase_key, options=ClientOptions(
   postgrest_client_timeout=120,
   storage_client_timeout=120,
   schema="public",
 ))

# Define MemorySaver instance for langgraph agent
memory = MemorySaver()


# Routes
# Index route
@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")  # Serve the chat interface

# Stream route
@app.route("/stream", methods=["GET"])
@login_required
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

# Sign up route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if user:
            flash("Username already registered.", "error")
            return redirect(url_for("signup"))
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password, method="pbkdf2:sha256")
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))
        login_user(user)
        flash("Logged in successfully!", "success")
        return redirect("/")
    return render_template("login.html")

# My Account route
@app.route("/my_account", methods=["GET", "POST"])
@login_required
def my_account():
    if request.method == "POST":
        current_password = request.form.get("current_password")
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")
        if not check_password_hash(current_user.password, current_password):
            flash("Current password is incorrect.", "error")
            return redirect(url_for("my_account"))
        if new_password != confirm_password:
            flash("New passwords do not match.", "error")
            return redirect(url_for("my_account"))
        current_user.password = generate_password_hash(new_password, method="pbkdf2:sha256")
        db.session.commit()
        flash("Password updated successfully!", "success")
        return redirect(url_for("index"))
    return render_template("my_account.html", user=current_user)

# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been successfully logged out.", "success")
    return redirect(url_for("login"))


# Function to add to the log in the app.log file
def log_run(run_status):
    if run_status in ["cancelled", "failed", "expired"]:
        log.error(str(datetime.datetime.now()) + " Run " + run_status + "\n")

# Run the Flask server
if __name__ == "__main__":
    with app.app_context():
        db.create_all() # Ensure the database is created
        print("Database created")
    app.run(debug=True)
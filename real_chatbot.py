# climate_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app) 

# Store LLM sessions by session ID
llm_sessions = {}
chat_histories = {}

load_dotenv()

# Initialize Gemini model once
google_api_key = os.getenv("GOOGLE_API_KEY")
 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.7,
    google_api_key=google_api_key  
)

@app.route('/start', methods=['POST', 'OPTIONS'])
def start_chat():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS preflight passed'}), 200

    data = request.get_json()

    # Collect and format data
    user_data = {
        "Name": data.get("name", ""),
        "Email": data.get("email", ""),
        "Commute": data.get("commute", ""),
        "Diet": data.get("diet", ""),
        "Electricity (kWh)": data.get("electricity", ""),
        "Single-Use Plastic": data.get("plastic", ""),
        "Flights Per Year": data.get("flights", ""),
        "Wants Weekly Tips": data.get("tips", "")
    }

    initial_prompt = f"""
You are Sparky, a friendly, encouraging, and expert AI Climate Coach.
A user has just filled out a form with their climate habits. Your task is to analyze this data and begin a helpful conversation with them.

Here is the user's data:
- Name: {user_data['Name']}
- Commute: {user_data['Commute']}
- Diet: {user_data['Diet']}
- Monthly Electricity Usage: {user_data['Electricity (kWh)']} kWh
- Single-Use Plastic: {user_data['Single-Use Plastic']}
- Flights Per Year: {user_data['Flights Per Year']}

First, silently estimate their carbon footprint based on this data. Then, craft a warm and welcoming introductory message that does the following:
1. Greet the user by their name.
2. Present your estimated carbon footprint in a gentle, non-judgmental way.
3. Praise them for one positive action you can see in their data (e.g., if flights are '0' or they bike).
4. Ask them an open-ended question to start the conversation, like "What would you like to explore first?" or "What's on your mind regarding your footprint?".

Generate ONLY this initial greeting message now.
"""

    message = HumanMessage(content=initial_prompt)
    response = llm.invoke([message])

    # Create a session ID and store the history
    session_id = str(uuid.uuid4())
    chat_histories[session_id] = [message]
    llm_sessions[session_id] = llm

    return jsonify({"response": response.content, "session_id": session_id})


@app.route('/chat', methods=['POST'])
def continue_chat():
    data = request.get_json()
    session_id = data.get("session_id")
    user_prompt = data.get("prompt", "")

    if not session_id or session_id not in chat_histories:
        return jsonify({"response": "Session expired or invalid. Please restart."}), 400

    # Continue the conversation
    user_message = HumanMessage(content=user_prompt)
    chat_histories[session_id].append(user_message)
    response = llm_sessions[session_id].invoke(chat_histories[session_id])
    chat_histories[session_id].append(response)

    return jsonify({"response": response.content})


if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

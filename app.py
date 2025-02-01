import os
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json

load_dotenv()

firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")

if firebase_credentials_json:
    firebase_credentials = json.loads(firebase_credentials_json)

    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

    db = firestore.client()
else:
    raise ValueError("FIREBASE_CREDENTIALS environment variable is missing!")

app = Flask(__name__)
CORS(app)

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Finance Recommendation API!"


@app.route("/suggest_financial_tasks/<userId>", methods=["GET"])
def suggest_financial_tasks(userId):
    try:
        user_ref = db.collection("Users").document(userId).collection("quizzes").document("latestQuiz")
        doc = user_ref.get()

        if not doc.exists:
            return jsonify({"error": "No quiz data found for this user"}), 404

        quiz_data = doc.to_dict()
        questions = quiz_data.get("questions", [])
        responses = quiz_data.get("responses", {})

        formatted_data = ""
        for idx, q in enumerate(questions):
            question_text = q.get("question", "")
            response_text = responses.get(str(idx), "No response")
            formatted_data += f"Q{idx+1}: {question_text}\nUser Response: {response_text}\n\n"

        prompt = ChatPromptTemplate.from_template("""
            Based on the user's quiz responses create 4 tasks that are similar to the Examples given below.
            The suggestions should be personalized based on their answers.
            You have to choose the difficulty of task on the basis of response.
            Examples: a) Your monthly pocket money is 1000 rupees and user have to complete the debt of 7000 dollar in 6 months,
            b) Your monthly earning is 25000 rupees and you have to pay the debt of the car loan of 500000 in 12 months,
            c) complete the debt of 1000000 in 60 months with a salary of 200000 rupees,
            d) complete the debt of 500000 in 30 months with a salary of 300000 using stock market only.
            Make you tasks are strictly like the Examples.   
            Here are the user's responses:
            {context}
        """)

        task_prompt = prompt.invoke({"context": formatted_data})
        response = llm.invoke(task_prompt)

        return jsonify({"suggested_tasks": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

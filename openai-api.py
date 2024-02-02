from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
cors = CORS(app)


# Set your OpenAI API key
openai.api_key = 'YOUR_KEY_HERE'

@app.route('/',methods=['GET'])
def hello():
    return jsonify({"ee":'hello'})

'''@app.route('/financial-advisor', methods=['POST'])
def financial_advisor():
    try:
        # Get the user's question from the JSON body of the POST request
        user_question = request.json.get('question')

        # Modify the system prompt accordingly
        system_prompt = "As a financial expert, answer the following question:\n\n"

        # Combine the system prompt and user question
        prompt = f"{system_prompt}{user_question}"

        # Call OpenAI API to generate a financial-based response
        response = openai.Completion.create(
            model="gpt-3.5-turbo-0125",  # Use the GPT-3.5-turbo model
            prompt=prompt,
            temperature=0.7,
            max_tokens=150,
            n=1,
            stop=None
        )

        # Extract the generated answer from the OpenAI response
        generated_answer = response['choices'][0]['text'].strip()

        # Prepare the response
        api_response = {'answer': generated_answer}

        return jsonify(api_response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500'''
@app.route('/financial-advisor', methods=['POST'])
def financial_advisor():
    try:
        user_question = request.json.get('question')

        # Call OpenAI Chat API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an experienced and highly skilled financial advisor with a deep understanding of various financial instruments, investment strategies, and economic trends. Your goal is to assist the user by offering well-informed and comprehensive responses to their financial inquiries.Feel free to leverage your expertise to provide actionable advice, and don't hesitate to seek further details from the user to tailor your responses to their specific financial situation."},
                {"role": "user", "content": user_question}
            ]
        )

        # Extract the generated answer
        generated_answer = response['choices'][0]['message']['content']

        # Prepare the response
        api_response = {'answer': generated_answer}

        return jsonify(api_response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

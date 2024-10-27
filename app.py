from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

app = Flask(__name__)

# Store chat history globally
chat_history_ids = None

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    global chat_history_ids  # Use the global variable to store chat history
    try:
        msg = request.form["msg"]
        input = msg
        response = get_chat_response(input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": "Error: " + str(e)})

def get_chat_response(text):
    global chat_history_ids  # Access the global chat history

    # Encode user input and append the EOS token
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Generate response using the model
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated response
    response_text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Update chat history
    chat_history_ids = chat_history_ids  # Keep the updated chat history

    return response_text  # Return only the response text

if __name__ == "__main__":
    app.run(debug=True)

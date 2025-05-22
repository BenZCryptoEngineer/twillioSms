from flask import Flask, request, jsonify
import os
from twilio.rest import Client
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
phone_number = os.getenv('TWILIO_PHONE_NUMBER')

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"status": "ok", "message": "Twilio SMS Receiver is running!"})

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/sms', methods=['POST'])
def receive_sms():
    """Receive SMS webhook from Twilio"""
    try:
        # Get incoming message details
        from_number = request.values.get('From', '')
        body = request.values.get('Body', '')
        
        # Log the received message
        print(f"Received message from {from_number}: {body}")
        
        # You can process the message or store it in a database here
        
        return jsonify({"status": "success", "message": "SMS received"})
    except Exception as e:
        print(f"Error receiving SMS: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/messages', methods=['GET'])
def get_messages():
    """Get recent messages sent to our number"""
    try:
        client = Client(account_sid, auth_token)
        
        # Get the most recent messages (default limit is 10)
        limit = request.args.get('limit', 10, type=int)
        messages = client.messages.list(to=phone_number, limit=limit)
        
        result = []
        for msg in messages:
            result.append({
                "from": msg.from_,
                "body": msg.body,
                "date_sent": str(msg.date_sent),
                "status": msg.status
            })
        
        return jsonify({"status": "success", "messages": result})
    except Exception as e:
        print(f"Error fetching messages: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 
# Twilio SMS Receiver

A simple Flask application that receives and fetches SMS messages using Twilio.

## Features

- Webhook endpoint to receive SMS messages from Twilio
- API endpoint to fetch recent messages
- Ready for deployment on Railway.app

## Local Development

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file based on `env.example` and add your Twilio credentials:
   ```
   cp env.example .env
   # Edit .env with your actual credentials
   ```
5. Run the application:
   ```
   python app.py
   ```

## Setting Up Twilio

1. Sign up for a Twilio account at [twilio.com](https://www.twilio.com/)
2. Get a Twilio phone number with SMS capabilities
3. Set up a webhook in your Twilio account to point to your `/sms` endpoint
   - When deployed, this will be: `https://your-railway-app-url.railway.app/sms`

## Deploying to Railway.app

1. Create an account on [Railway.app](https://railway.app/)
2. Connect your GitHub repository
3. Add the following environment variables in Railway:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `PHONE_NUMBER`
4. Deploy your app
5. After deployment, update your Twilio webhook URL to point to your Railway app

## API Endpoints

- `GET /`: Health check
- `GET /health`: Health check for Railway
- `POST /sms`: Webhook for receiving SMS messages from Twilio
- `GET /messages`: Get recent messages sent to your Twilio number
  - Optional query parameter: `limit` (default: 10)

## License

[MIT License](LICENSE)

from flask import Flask, request, jsonify
import os
import sys
import logging
from twilio.rest import Client
from dotenv import load_dotenv
from flask_cors import CORS

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 记录启动信息
logger.info("====== 应用启动 ======")
logger.info(f"Python 版本: {sys.version}")
logger.info(f"当前工作目录: {os.getcwd()}")

# Load environment variables
logger.info("正在加载环境变量...")
load_dotenv()

# Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
phone_number = os.getenv('PHONE_NUMBER')

logger.info(f"TWILIO_ACCOUNT_SID 已设置: {bool(account_sid)}")
logger.info(f"TWILIO_AUTH_TOKEN 已设置: {bool(auth_token)}")
logger.info(f"PHONE_NUMBER 已设置: {bool(phone_number)}")

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    logger.info("访问主页")
    return jsonify({"status": "ok", "message": "Twilio SMS Receiver is running!"})

@app.route('/health')
def health():
    logger.info("健康检查")
    return jsonify({"status": "ok"})

@app.route('/debug')
def debug():
    """返回调试信息，帮助排查问题"""
    logger.info("访问调试页面")
    
    # 收集调试信息
    debug_info = {
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "TWILIO_ACCOUNT_SID": bool(account_sid),
            "TWILIO_AUTH_TOKEN": bool(auth_token),
            "PHONE_NUMBER": bool(phone_number),
            "PORT": os.getenv('PORT')
        },
        "all_env_vars": dict(os.environ)
    }
    
    return jsonify(debug_info)

@app.route('/sms', methods=['POST'])
def receive_sms():
    """Receive SMS webhook from Twilio"""
    try:
        # Get incoming message details
        from_number = request.values.get('From', '')
        body = request.values.get('Body', '')
        
        # Log the received message
        logger.info(f"收到短信 - 来自: {from_number}, 内容: {body}")
        logger.debug(f"完整请求数据: {request.values.to_dict()}")
        
        # You can process the message or store it in a database here
        
        return jsonify({"status": "success", "message": "SMS received"})
    except Exception as e:
        logger.error(f"接收短信时出错: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/messages', methods=['GET'])
def get_messages():
    """Get recent messages sent to our number"""
    try:
        logger.info("尝试获取最近消息")
        
        if not account_sid or not auth_token:
            logger.error("缺少Twilio凭证")
            return jsonify({"status": "error", "message": "Missing Twilio credentials"}), 500
        
        client = Client(account_sid, auth_token)
        logger.info("Twilio客户端初始化成功")
        
        # Get the most recent messages (default limit is 10)
        limit = request.args.get('limit', 10, type=int)
        logger.info(f"获取最近 {limit} 条消息")
        
        messages = client.messages.list(to=phone_number, limit=limit)
        logger.info(f"成功获取 {len(messages)} 条消息")
        
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
        logger.error(f"获取消息时出错: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logger.info(f"应用将在端口 {port} 上启动")
    app.run(host='0.0.0.0', port=port, debug=False) 
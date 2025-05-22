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

# 记录所有请求的中间件
@app.before_request
def log_request():
    logger.info(f"收到请求: {request.method} {request.path}")
    logger.info(f"请求头: {dict(request.headers)}")
    if request.method == 'POST':
        logger.info(f"表单数据: {request.form.to_dict()}")
        logger.info(f"JSON数据: {request.get_json(silent=True)}")

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

@app.route('/echo', methods=['GET', 'POST'])
def echo():
    """简单的回显端点，用于测试Webhook"""
    logger.info("收到回显请求")
    if request.method == 'POST':
        return jsonify({
            "status": "success",
            "method": "POST",
            "form_data": request.form.to_dict(),
            "json_data": request.get_json(silent=True),
            "headers": dict(request.headers)
        })
    else:
        return jsonify({
            "status": "success",
            "method": "GET",
            "args": request.args.to_dict(),
            "headers": dict(request.headers)
        })

@app.route('/sms', methods=['POST', 'GET'])
def receive_sms():
    """Receive SMS webhook from Twilio"""
    logger.info("===== SMS Webhook被调用 =====")
    logger.info(f"请求方法: {request.method}")
    
    # 通用消息，适用于GET和POST
    if request.method == 'GET':
        logger.info("收到GET请求，这可能不是Twilio发送的")
        logger.info(f"GET参数: {request.args.to_dict()}")
        return jsonify({
            "status": "success", 
            "message": "SMS endpoint is working, but Twilio should send a POST request"
        })
    
    try:
        # 记录所有收到的表单数据
        logger.info("===== 详细的表单数据 =====")
        for key, value in request.form.items():
            logger.info(f"{key}: {value}")
        
        # Get incoming message details
        from_number = request.values.get('From', '')
        body = request.values.get('Body', '')
        message_sid = request.values.get('MessageSid', '')
        
        # Log the received message
        logger.info(f"收到短信 - 来自: {from_number}, 内容: {body}, SID: {message_sid}")
        logger.debug(f"完整请求数据: {request.values.to_dict()}")
        
        # You can process the message or store it in a database here
        
        # 如果是正确的Twilio请求，应该会在请求中包含這些参数
        if from_number and body:
            logger.info("这看起来是一个有效的Twilio SMS请求")
        else:
            logger.warning("请求缺少必要的Twilio SMS参数")
        
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
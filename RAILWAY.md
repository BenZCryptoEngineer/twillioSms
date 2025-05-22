# Railway.app 部署指南

## 什么是 Railway.app?

Railway.app 是一个现代化的应用部署平台，可以轻松部署和扩展应用程序。无需复杂的基础架构设置，只需将代码推送到GitHub，Railway就能自动构建和部署。

## 部署步骤

### 1. 创建 Railway 账户

访问 [Railway.app](https://railway.app/) 并使用GitHub账户登录。

### 2. 创建新项目

在Railway控制台中，点击"New Project"，然后选择"Deploy from GitHub"。

### 3. 连接GitHub仓库

选择包含此应用程序的GitHub仓库。

### 4. 配置环境变量

在项目设置中，添加以下环境变量：

- `TWILIO_ACCOUNT_SID` - 你的Twilio账户SID
- `TWILIO_AUTH_TOKEN` - 你的Twilio认证令牌
- `PHONE_NUMBER` - 你的Twilio电话号码（格式：+1XXXXXXXXXX）
- `PORT` - (可选) 应用端口，默认为5000

### 5. 部署应用

Railway会自动检测到项目根目录中的`railway.toml`文件和`Procfile`，并据此配置部署。

### 6. 获取应用URL

部署成功后，Railway会为应用生成一个URL（例如：https://your-app-name.up.railway.app）。

### 7. 配置Twilio Webhook

在Twilio控制台中，找到你的电话号码设置，将SMS webhook URL设置为：
```
https://your-app-name.up.railway.app/sms
```

## 监控和日志

Railway提供了内置的日志查看功能。在项目控制台中，点击"Logs"标签可以查看应用日志。

## 故障排除

如果部署失败，请检查：

1. 确保所有必需的环境变量都已正确设置
2. 检查Railway日志中的错误信息
3. 确认`railway.toml`和`Procfile`文件格式正确
4. 验证应用的健康检查端点(/health)是否正常工作

## 自动部署

Railway会自动监听GitHub仓库的变化，当你推送新代码时，会自动重新部署应用。 
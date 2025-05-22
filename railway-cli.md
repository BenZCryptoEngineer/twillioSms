# Railway CLI 工具使用指南

Railway CLI是一个命令行工具，可以帮助你在本地开发环境中管理和测试你的Railway项目。

## 安装Railway CLI

```bash
# 使用npm安装
npm i -g @railway/cli

# 或者使用Homebrew (macOS)
brew install railway
```

## 登录Railway账户

```bash
railway login
```

这将打开浏览器窗口，引导你完成登录过程。

## 连接到项目

```bash
# 在项目目录中运行
railway link
```

这会将你的本地目录链接到Railway项目。

## 本地开发

### 使用Railway环境变量在本地运行

```bash
railway run python app.py
```

这个命令会从Railway加载所有环境变量，然后运行你的应用。

### 查看项目环境变量

```bash
railway variables
```

### 添加/修改环境变量

```bash
railway variables set TWILIO_ACCOUNT_SID=your_account_sid
railway variables set TWILIO_AUTH_TOKEN=your_auth_token
railway variables set PHONE_NUMBER=+1XXXXXXXXXX
```

## 部署

### 手动部署

```bash
railway up
```

### 查看部署状态

```bash
railway status
```

### 查看日志

```bash
railway logs
```

## 其他有用的命令

### 打开项目仪表板

```bash
railway open
```

### 查看帮助

```bash
railway help
```

## 注意事项

- 确保在使用`railway run`之前已经运行了`railway link`
- 对于生产环境，最好使用GitHub自动部署而非CLI手动部署
- Railway CLI可以帮助快速调试环境变量和配置问题 
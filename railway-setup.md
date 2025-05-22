# Railway自动部署设置指南

## 检查并启用自动部署

1. 在Railway控制台中，打开你的项目
2. 点击"Settings"标签
3. 在左侧菜单中查找"Deployments"或"GitHub"选项
4. 确保"Auto Deploy"或"Automatic Deployments"已启用
5. 如果看到"GitHub Integration"选项，确保已正确连接GitHub仓库

## 验证GitHub权限

1. 在Railway控制台的"Settings"中，查找"GitHub"或"Integrations"部分
2. 点击"Disconnect"然后重新连接你的GitHub账户
3. 确保给予Railway足够的权限访问你的仓库

## 手动触发首次部署

有时首次部署需要手动触发，之后的更改才会自动部署：

1. 在项目控制台中点击"Deploy"按钮
2. 等待部署完成
3. 之后的代码推送应该会自动触发新的部署

## 检查分支配置

确保Railway监听的是正确的分支：

1. 在项目设置中查找"Source"或"GitHub"设置
2. 确认"Branch"设置为"main"（或你推送代码的分支）

## 检查部署日志

如果自动部署已触发但失败了：

1. 点击"Deployments"标签
2. 查看最近的部署记录
3. 点击查看详细日志，寻找错误信息

## 使用Railway CLI测试部署

有时使用CLI工具可以提供更详细的错误信息：

```bash
# 安装Railway CLI
npm i -g @railway/cli

# 登录
railway login

# 链接项目
railway link

# 手动部署
railway up
```

## 其他可能的解决方法

1. 尝试创建一个新的Railway项目，重新连接GitHub仓库
2. 在GitHub仓库设置中检查Webhooks，确保Railway的webhook已正确设置
3. 联系Railway支持寻求帮助：support@railway.app 
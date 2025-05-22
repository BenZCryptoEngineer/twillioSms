# Railway 部署故障排除指南

## 常见部署问题及解决方案

### 问题：Nixpacks 无法生成构建计划

如果你看到以下错误：
```
Nixpacks was unable to generate a build plan for this app.
```

**解决方案**:

1. **确保所有文件已提交到Git仓库**
   ```bash
   git add .
   git commit -m "添加所有应用文件"
   git push
   ```

2. **检查项目文件结构**
   确保以下关键文件在你的项目根目录中:
   - app.py (Flask应用)
   - requirements.txt (依赖列表)
   - Procfile (启动命令)
   - runtime.txt (指定Python版本)
   - Dockerfile (可选，但有助于明确构建过程)

3. **使用Docker部署模式**
   Railway支持直接从Dockerfile部署。如果自动检测失败，可以:
   - 创建一个基本的Dockerfile
   - 在Railway项目设置中选择"Docker"部署模式

## 部署前的检查清单

- [ ] 所有文件已提交到Git
- [ ] requirements.txt包含所有依赖
- [ ] Procfile包含正确的启动命令
- [ ] runtime.txt指定Python版本
- [ ] 在本地测试过应用程序
- [ ] 环境变量已在Railway控制台中设置

## 手动强制重新部署

如果你需要强制重新部署，可以使用Railway CLI:

```bash
railway up
```

或者在Railway控制台中点击"Deploy"按钮。

## 查看部署日志

详细的部署日志可以帮助诊断问题:

1. 在Railway控制台中点击你的服务
2. 选择"Deployments"标签
3. 点击最近的部署
4. 查看"Build Logs"和"Deploy Logs"

## 使用Nixpacks调试

你可以在本地使用Nixpacks测试构建:

```bash
# 安装nixpacks
npm install -g nixpacks

# 在项目目录中测试构建
nixpacks build .
```

## 其他注意事项

- Railway默认使用us-west-1区域，可以在项目设置中更改
- 确保你的应用有一个健康检查端点(/health)
- 如果使用自定义域名，需要在DNS中设置CNAME记录 
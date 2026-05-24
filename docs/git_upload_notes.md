# GitHub 上传说明

本项目目录包含大量原始数据、PDF 文献、Word/Excel 材料、模型二进制和生成长表。为避免 GitHub 仓库过大或泄露非代码材料，上传策略如下：

- 上传代码、配置、README、轻量文档和小型演示结果。
- 排除 `.codex-research-mcp/`、`.playwright-mcp/`、原始数据目录、PDF、Word、Excel、模型权重和大型 CSV。
- 若需要共享完整数据，应单独使用网盘、Release 附件、Git LFS 或数据仓库，并在报告中说明数据来源与脱敏规则。

推送命令模板：

```powershell
git remote add origin https://github.com/<owner>/<repo>.git
git branch -M main
git push -u origin main
```

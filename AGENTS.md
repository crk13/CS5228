# Agent 开发入口

动手前阅读任务相关的重要文档。公共框架接口只在 [接口协议](docs/experiment-interface.md) 中定义；本文保留命令和入口，不复制协议规则或参数说明。

## 重要文档

- [公共实验框架接口协议](docs/experiment-interface.md)：接入特征、模型、验证和实验时必读；全仓库唯一协议信源。
- [清洗数据说明与数据字典](data_cleaned/README.md)：读取数据或使用辅助数据前必读。
- [项目概述、团队分工与目录结构](docs/project.md)
- [Kaggle 任务说明原文](CS5228-2610%20Project%20_%20Kaggle.mhtml)
- [共享实验日志](experiments/log.csv)
- [运行依赖](requirements.txt)

## 常用开发与测试命令

均在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v  # 框架契约测试
python -m src.run_baseline              # 完整基线；追加日志并生成提交
```

修改框架或模型接口后运行契约测试；需要验证完整训练流程时运行基线。仅改文档时检查链接与示例语法，无需重新训练。

修改基础清洗逻辑后，重新生成并校验：

```bash
python -m src.data_cleaning.clean
python -m src.data_cleaning.checks
```

保持现有代码风格，注释精炼。接口变化时同步更新唯一接口协议、实现及相关测试；其他文档只维护入口链接。项目背景、分工或目录变化时更新项目说明。

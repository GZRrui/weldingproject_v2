# 薄板焊接代理与决策系统 v2

当前阶段：可复现前向基线、批次合并、校准区间与已有仿真工况推荐。连续优化、交互界面和工程验收尚未完成。

## 目录

- `welding.py`：导入、合并、训练、预测和已有工况检索入口。
- `data/raw/<批次>/`：不可覆盖的批次副本，含成功和失败记录。
- `data/processed/<数据指纹>/`：去重数据、完整来源记录、失败记录和划分摘要。
- `artifacts/runs/<时间>/`：候选模型、交叉验证结果、独立测试结果和版本信息。
- `examples/`：输入示例。
- `docs/`：物理口径、实施边界。
- `tests/`：关键行为验证。

## 运行

已有环境：`D:\miniconda3\envs\mywelding\python.exe`。下列命令中的 `python` 应使用该解释器，或先激活 `mywelding`。

```powershell
python welding.py import-batch --batch batch_001 --success clean_data_1219.csv --failed failed_cases.csv
python welding.py prepare
python welding.py train
python welding.py predict --run artifacts/runs/<本次目录> --input examples/predict.json --output artifacts/prediction.json
python welding.py recommend --run artifacts/runs/<本次目录> --input examples/recommend.json --output artifacts/recommendation.json
python -m unittest discover -s tests
```

初始批次导入一次即可。后续数据使用新的批次名称，核对仿真设置后执行导入及训练；不会覆盖旧模型，也不会自动将候选模型发布为工程模型。

数据采用固定哈希工况组划分约60%训练、20%校准、20%测试，两种焊接顺序同组。新增工况归属稳定。重复输入的输出若存在超出数值容差的差异，合并会停止，需先人工核查。相同重复记录只训练一次，完整来源仍保留。

训练分别处理两种顺序、七个目标，使用训练集内五折分组交叉验证选择均值、二阶响应面、Extra Trees或梯度提升。校准集只用于95%边际预测区间，测试集只评价最终选定模型。区间不是多目标联合保证，也不保证优化推荐点的覆盖。

推荐模块使用已知仿真实际输出筛选目标上限，再按归一化参数变化量与顺序切换成本排序。它可能检索到测试记录，因此不能拿这些推荐结果再评价代理模型泛化精度。输入可通过 `locked` 固定参数。连续参数优化、步长、调整幅度、成本权重和多目标折中将在后续阶段扩展。

## 下一步

增强模型比较（GPR、XGBoost、原有模型；原始/物理特征；分别/统一建模）：

```powershell
python enhanced.py --baseline artifacts/runs/20261103_145522_273665
```

在原训练分区内进行三折工况分组验证，按两种顺序的平均MAE选择每个目标的配置；校准和历史测试分区保持不变。GPR采用固定Matérn核的三个长度尺度比较，XGBoost比较两个深度，属于有限搜索而非穷尽调参。输出 `cv_selection.csv`、`comparison.csv`、`report.md` 和兼容预测/决策的候选模型包。原测试集已查看，仅是历史基准，未来数据应保留新的最终验证点。

项目V1约束决策已提供独立入口：

```powershell
python decision.py --run artifacts/runs/20261103_145522_273665 --input examples/optimize.json --output artifacts/decisions/<新目录>
```

按用户给定的可执行步长，在训练范围和最大调整幅度内穷举候选，两种顺序分别预测。步长示例是开发设定，尚不是经设备确认的工艺分辨率。使用 `预测值 + max(项目模型误差预算, 校准半径) ≤ 工艺上限` 过滤；不声称连续全局最优。导出全部评价、最多5条推荐、最多10条新工况ANSYS复核队列。无解时导出最接近限值的不可行方案，绝不将其标成可行。

该模块尚无熔深、强度等焊接质量约束，高电流区仅标注风险；输出是待复核的仿真开发候选。推荐结果的改善需要ANSYS验证，不用模型自身预测作为验证证据。旧文中的工程容差待定状态已由 `docs/project_tolerances_v1.md` 的项目自定V1开发限值替代；标准符合性与实验验收仍未建立。

区域诊断与参数水平留出验证：

```powershell
python validate.py --run artifacts/runs/<本次目录>
```

输出位于该模型目录的 `validation/`：中文报告、区域误差、最差工况、模型误差能力和水平留出结果。水平留出仅使用训练与校准分区，固定原有模型族；最终测试集不参与重新拟合。内部水平属于插值压力测试，两端水平属于边界外推压力测试。工程容差在 `configs/engineering_tolerances.json` 单独管理，缺少工程要求时保持空值。示例推荐中的限值不是正式验收标准。

1. 检查本次模型误差，结合工程容差设定验收阈值。
2. 增加参数水平留出、统一/分顺序模型比较和区域误差报告。
3. 加入经过验证的连续优化与ANSYS复核清单。
4. 在稳定接口上增加界面与模型发布流程。

仅加载可信来源的joblib模型文件。

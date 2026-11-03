# 工程范式参考

检索日期：2026-11-03。仅参考项目组织和验证思想，未复制外部代码或新增依赖。

- SMT：https://github.com/SMTorg/smt — 代理模型、采样及基准问题分离；本项目对应独立的数据准备、训练和验证入口。后续可比较Kriging等平滑代理。
- SMT tutorials：https://github.com/SMTorg/smt/tree/master/tutorial — 工程代理建模教程入口。
- MAPIE：https://github.com/scikit-learn-contrib/MAPIE — 预测区间与独立校准；本项目保留校准分区，并同时报告区域覆盖与区间误差。边际覆盖不等于每个区域或优化点覆盖。

当前不因参考项目而替换已有算法。先定位现有模型的区域与插值薄弱点，再决定是否新增模型族。

# MS-LSFTrack 方法说明

本文档对应公开版论文 v1，不包含后续 dev 预测位置或结构预测实验。

## 1. 任务

MS-LSFTrack 关注检测后多目标跟踪中的关联阶段。给定统一检测结果，目标是在在线场景中为每条已有轨迹选择正确的当前帧候选检测。

密集红外/小目标场景中，多个候选与轨迹尾节点距离可能非常接近，仅靠最近邻或简单运动距离容易造成身份切换。因此本文引入局部结构场来描述目标在局部群体中的相对空间关系。

## 2. 局部结构场

对某个目标检测中心 `x`，在半径 `r` 内收集同帧邻近检测：

```text
N_r(x) = { y_k | ||y_k - x|| <= r }
```

每个邻居以相对坐标和检测置信度表示：

```text
[(y_k - x) / r, score_k]
```

这些无序邻居点会被栅格化为固定大小的 Local Structure Field, LSF。网络使用共享 CNN 编码历史端 LSF 与候选端 LSF，并输出结构相容性分数。

## 3. 多尺度结构

论文公开版使用三个局部结构半径：

```text
10,15,20
```

不同尺度反映不同范围的局部上下文。小尺度更关注近邻干扰，大尺度提供更稳定的局部群体结构。

## 4. 密度引导多尺度融合

公开版使用：

```text
scale_mode = learned_density
density_prior_direction = normal
density_prior_lambda = 0.5
```

模型先学习每个候选的多尺度权重，再使用当前帧目标密度先验进行修正。密度只用于结构分支的多尺度融合，不参与最终运动-结构 ASR 融合增益。

## 5. ASR 运动-结构融合

公开版最终融合方式为：

```text
score_mode = ambiguity_rerank
asr_density_gain = 0.0
```

ASR 的核心思想是：

1. 以轨迹尾节点到候选检测的距离作为稳定运动锚点；
2. 当候选距离 margin 较小、运动关联存在歧义时，使用局部结构分数对候选进行重排；
3. easy motion case 中尽量保持运动锚点稳定，避免结构分支扰动明显正确的最近邻关联。

因此公开版方法可以概括为：

```text
tail-anchor ASR + tail-node local structure + density-guided multi-scale LSF
```

## 6. 与后续 dev 实验的区别

当前公开版不使用：

```text
predicted-position ASR
trajectory-level structure memory
predicted structure field
```

这些属于后续探索，不进入当前论文 v1 公开代码与结果。

## 7. 消融逻辑

公开结果包中的主线消融顺序是：

1. 单尺度 Structure Only：说明局部结构尺度会影响关联性能；
2. 多尺度 + 密度引导 Structure Only：说明密度引导的多尺度融合优于单纯学习或纯密度；
3. 运动-结构融合：确定 `ASR-default no fusion density`；
4. 模块消融：比较 no module / motion only / structure only / final motion+structure。

消融表位置：

```text
MS-LSFTrack-results/summary/pre_module_ablation_main_20260706/
```


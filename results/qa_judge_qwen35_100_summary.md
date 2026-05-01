# QA 大模型裁判评估汇总（100 分制）

- 评估样本数：2999
- Baseline 平均分：86.35/100
- Ours 平均分：69.60/100
- Baseline 总分：258956.0
- Ours 总分：208716.0
- 胜负统计：`{"baseline": 2348, "ours": 628, "tie": 23}`
- 并发数：24
- 输入模式：图片+问题+参考答案+两个候选回复
- 明细文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_100_details.jsonl`

## 分项平均分

| 维度 | 满分 | Baseline | Ours |
| --- | ---: | ---: | ---: |
| 核心结论/操作是否正确 | 40 | 35.57 | 29.22 |
| 关键细节完整性 | 25 | 20.97 | 15.59 |
| 图像证据一致性 | 20 | 17.80 | 13.83 |
| 铁路安全/专业性 | 10 | 8.93 | 7.58 |
| 表达清晰与无幻觉 | 5 | 3.33 | 3.39 |

## 评分机制

1. 核心结论/操作是否正确（40 分）：是否回答了问题的核心要求，和参考答案的主结论、主操作一致。
2. 关键细节完整性（25 分）：是否覆盖参考答案中的重要条件、步骤、注意事项，是否遗漏会影响判断的要点。
3. 图像证据一致性（20 分）：是否与图片可见内容一致；图片与参考答案冲突时，以图片中能确认的信息为准。
4. 铁路安全/专业性（10 分）：是否符合铁路场景下谨慎、限速、停车准备、信号/调度等安全原则。
5. 表达清晰与无幻觉（5 分）：表达是否清楚，是否有明显无关、编造、过度推断或只输出选项字母的问题。

## 分数解释

- 90-100：几乎完全正确，关键细节充分，与图片和参考答案一致。
- 75-89：主要正确，只有少量次要遗漏或表述不够完整。
- 60-74：部分正确，抓住主方向，但遗漏多个关键点。
- 40-59：有少量相关内容，但核心操作/判断明显不完整或不可靠。
- 20-39：大部分错误、空泛、与题目关系弱，或存在明显安全/事实问题。
- 1-19：几乎无效；只输出 A/B/C/D、空答案、答非所问、严重矛盾时通常落在这个区间。

## 追加实验：Baseline vs No-YOLO 直接 VLM

这次追加实验不使用 YOLO 检测结果，只把原始图片和问题交给同一个 Qwen3-VL 推理脚本变体 `ours_vllm_no_yolo.py`，再用 Qwen3.5-35B-A3B 按同一套 100 分制对 QA 题进行裁判评分。

- 评估样本数：3000
- Baseline 平均分：86.14/100
- No-YOLO VLM 平均分：78.79/100
- Baseline 总分：258425.0
- No-YOLO VLM 总分：236376.0
- 胜负统计：`{"baseline": 1503, "ours_no_yolo": 1255, "tie": 242}`
- 明细文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_baseline_vs_no_yolo_100_details.jsonl`
- 汇总文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_baseline_vs_no_yolo_100_summary.md`

### No-YOLO 分项平均分

| 维度 | 满分 | Baseline | No-YOLO VLM |
| --- | ---: | ---: | ---: |
| 核心结论/操作是否正确 | 40 | 35.37 | 32.15 |
| 关键细节完整性 | 25 | 20.79 | 18.44 |
| 图像证据一致性 | 20 | 18.17 | 17.47 |
| 铁路安全/专业性 | 10 | 8.78 | 8.00 |
| 表达清晰与无幻觉 | 5 | 3.18 | 2.86 |

## 追加实验：Baseline vs 微调后 No-YOLO VLM

这次追加实验使用 **LLaMA-Factory LoRA 微调后再合并得到的 Qwen3-VL**，不接 YOLO，只把原始图片和问题交给 `ours_vllm_no_yolo.py` 推理，然后继续用 Qwen3.5-35B-A3B 按同一套 100 分制做 QA 裁判评分。

- 评估样本数：2918
- Baseline 平均分：80.56/100
- 微调后 No-YOLO VLM 平均分：87.24/100
- Baseline 总分：235080.0
- 微调后 No-YOLO VLM 总分：254558.0
- 胜负统计：`{"baseline": 1037, "ours": 1876, "tie": 5}`
- 明细文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_baseline_vs_ft_no_yolo_100_details.jsonl`
- 汇总文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_baseline_vs_ft_no_yolo_100_summary.md`
- 微调后模型：`/mnt1/mnt2/data3/nlp/ws/model/Qwen3-VL-8B-RailVQA-Merged-fast`

### 微调后 No-YOLO 分项平均分

| 维度 | 满分 | Baseline | 微调后 No-YOLO VLM |
| --- | ---: | ---: | ---: |
| 核心结论/操作是否正确 | 40 | 33.24 | 36.09 |
| 关键细节完整性 | 25 | 19.52 | 21.06 |
| 图像证据一致性 | 20 | 16.65 | 17.64 |
| 铁路安全/专业性 | 10 | 8.05 | 8.93 |
| 表达清晰与无幻觉 | 5 | 3.17 | 3.57 |

## 追加实验：Baseline vs YOLO + 微调后 VLM

这次追加实验使用 **基座 YOLO26m** 先对图片画框，再把标注后的图像和问题一起送入 **LLaMA-Factory LoRA 微调后再合并得到的 Qwen3-VL**，对应推理脚本为 `ours_vllm.py`。

- 评估样本数：2891
- Baseline 平均分：82.33/100
- YOLO + 微调后 VLM 平均分：82.94/100
- Baseline 总分：238005.0
- YOLO + 微调后 VLM 总分：239783.0
- 胜负统计：`{"baseline": 1532, "ours": 1357, "tie": 2}`
- 明细文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_baseline_vs_ft_yolo_100_details.jsonl`
- 汇总文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_baseline_vs_ft_yolo_100_summary.md`
- 推理结果文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/ft_vlm_with_yolo_eval/proposed_vllm_results_260501_0746.json`
- 推理断点文件：`/mnt1/mnt2/data3/nlp/ws/proj/课题组/results/ft_vlm_with_yolo_eval/ours_vllm_resume.jsonl`
- 微调后模型：`/mnt1/mnt2/data3/nlp/ws/model/Qwen3-VL-8B-RailVQA-Merged-fast`
- YOLO 模型：`/mnt1/mnt2/data3/nlp/ws/model/YOLO26/yolo26m.pt`

### YOLO + 微调后 VLM 分项平均分

| 维度 | 满分 | Baseline | YOLO + 微调后 VLM |
| --- | ---: | ---: | ---: |
| 核心结论/操作是否正确 | 40 | 33.89 | 34.64 |
| 关键细节完整性 | 25 | 20.03 | 19.65 |
| 图像证据一致性 | 20 | 16.94 | 16.06 |
| 铁路安全/专业性 | 10 | 8.34 | 8.75 |
| 表达清晰与无幻觉 | 5 | 3.20 | 3.81 |

## 当前结论

| 对比方案 | 被比较方案平均分 | Baseline 平均分 | 差距 | 被比较方案胜场 | Baseline 胜场 | 平局 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO+VLM ours | 69.60 | 86.35 | -16.75 | 628 | 2348 | 23 |
| No-YOLO 直接 VLM | 78.79 | 86.14 | -7.35 | 1255 | 1503 | 242 |
| 微调后 No-YOLO VLM | 87.24 | 80.56 | +6.68 | 1876 | 1037 | 5 |
| YOLO + 微调后 VLM | 82.94 | 82.33 | +0.61 | 1357 | 1532 | 2 |

结论：在当前已完成的实验里，**最有效的提升仍然来自对 VLM 本身做 LoRA/SFT 微调**。未微调时，`ours_vllm_no_yolo.py` 虽然明显优于加入 YOLO 的 `ours_vllm.py`，但仍低于 `baseline_vllm.py`；微调后的 No-YOLO VLM 则已经**稳定反超 baseline**。当把**基座 YOLO26m** 再接到微调后的 VLM 前面时，平均分只比 baseline **小幅领先 +0.61**，而且胜场仍少于 baseline（1357 vs 1532），整体表现也**明显弱于微调后 No-YOLO VLM**。这说明在当前实现下，YOLO 画框提示并没有进一步放大微调收益，最稳的提升来源仍是 **VLM 自身微调**。

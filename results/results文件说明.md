# results 目录文件说明

本文档用于解释 `results` 目录下各个结果文件分别是什么、怎么来的、什么时候应该看哪个。

当前结果目录对应的远端路径是：

```text
/mnt1/mnt2/data3/nlp/ws/proj/课题组/results
```

## 一句话总结

最重要的文件有三个：

```text
baseline_vllm_run_summary.md          baseline 的最终速度和准确率汇总
baseline_vllm_results_260430_0907.json baseline 的最终完整结果
ours_vllm_resume.jsonl                ours 当前断点结果，跑完前主要看它
```

如果只是想快速知道 baseline 效果，看：

```text
baseline_vllm_run_summary.md
```

如果想看 baseline 每一道题的输入、模型回复和是否正确，看：

```text
baseline_vllm_trace.log
```

如果想判断 ours 是否优于 baseline，需要等 ours 跑完后，再对比：

```text
baseline_vllm_resume.jsonl
ours_vllm_resume.jsonl
```

现在 ours 已经跑完，已经生成了正式对比汇总：

```text
baseline_vs_ours_summary.md
```

注意不要再用 `baseline_summary.md` 判断最终效果。`baseline_summary.md` 是早期只包含 baseline 的旧汇总；正式对比应该看 `baseline_vs_ours_summary.md`。

## 当前快照

截至本次整理时：

```text
baseline_vllm_resume.jsonl: 6000 行，baseline 已完整跑完
ours_vllm_resume.jsonl:     1176 行，ours 已完成一部分，后续会继续增长
```

baseline 的最终运行汇总为：

```text
已完成：6000/6000
本次运行耗时：23m04s
平均耗时：0.27 秒/条
MC 准确率：2746/3000 = 91.53%
任务数量：qa 3000，mc 3000
```

注意：`ours_vllm_resume.jsonl` 是断点文件，只要 ours 还在继续跑，它的行数和大小就会继续变化。

## 文件逐个说明

### baseline_summary.json

这是早期用 `summarize_results.py` 对 baseline 的部分结果做出的 JSON 汇总。

它生成的时候 baseline 只跑到了约 `496/6000`，所以它不是最终结果。

现在 baseline 已经完整跑完，因此这个文件只作为早期中间统计记录保留，正式对比时可以忽略。

### baseline_summary.md

这是 `baseline_summary.json` 对应的 Markdown 版本，方便人直接阅读。

它里面记录的是早期 partial baseline：

```text
baseline: 496/6000
MC 准确率: 224/248 = 90.32%
```

因为 baseline 后来已经跑完 6000 条，所以这个文件现在不是最终结论。

它也不会包含 ours，因为它生成时 ours 还没有跑完。现在 baseline 和 ours 的正式对比文件是：

```text
baseline_vs_ours_summary.md
```

### baseline_vllm_resume.jsonl

这是 baseline vLLM 推理的断点续跑文件。

它是一行一个样本结果，格式是 JSONL：

```text
第 1 行：第 1 个样本的结果
第 2 行：第 2 个样本的结果
...
第 6000 行：第 6000 个样本的结果
```

这个文件是边跑边写入的。每完成一个 batch，脚本就会把这一批结果追加到这里。

它的作用有两个：

1. 中断后可以继续跑，不会重复已经完成的样本。
2. 可以作为最终对比分析的输入文件。

当前这个文件有 `6000` 行，说明 baseline 已经完整跑完。

不要随便删除它。删除后 baseline 的断点记录就没了，再跑会从头开始。

### baseline_vllm_results_260430_0907.json

这是 baseline 完整跑完后生成的最终 JSON 结果文件。

文件名里的时间含义是：

```text
260430_0907 = 2026-04-30 09:07
```

它和 `baseline_vllm_resume.jsonl` 保存的是同一批 baseline 推理结果，只是格式不同：

```text
baseline_vllm_resume.jsonl
一行一个 JSON，适合断点续跑和追加写入

baseline_vllm_results_260430_0907.json
完整 JSON 数组，适合最终保存、提交、整体分析
```

如果要交付 baseline 的完整原始结果，优先看这个文件。

### baseline_vllm_run_summary.json

这是 baseline 完整运行结束后自动生成的机器可读汇总。

里面记录：

```text
数据集路径
最终结果路径
断点文件路径
完成数量
任务数量
MC 正确数
MC 总数
MC 准确率
本次运行耗时
平均每条耗时
```

它适合脚本读取，不太适合人直接看。

### baseline_vllm_run_summary.md

这是 `baseline_vllm_run_summary.json` 对应的 Markdown 人类可读版本。

这是目前 baseline 最推荐查看的总览文件。

当前关键结果是：

```text
已完成：6000/6000
本次运行耗时：23m04s
平均耗时：0.27 秒/条
MC 准确率：2746/3000 = 91.53%
任务数量：qa 3000，mc 3000
```

如果只是想知道 baseline 跑得怎么样，看这个文件最快。

### baseline_vllm_trace.log

这是 baseline 的详细推理日志。

它会记录每个样本的完整信息：

```text
ID
任务类型
图片路径
输入问题
标准答案
模型完整回复
提取出的答案
是否正确
```

这个文件的作用是人工检查。

例如你想确认：

```text
baseline 到底给模型输入了什么？
模型完整回复是什么？
提取答案是否正确？
某一道题为什么错？
```

就应该看这个文件。

它比较大是正常的，因为保存了 6000 条完整输入输出。

### ours_vllm_resume.jsonl

这是 ours 方法的断点续跑文件。

ours 方法是在 baseline 的基础上，先用 YOLO 给图片加检测框，再把标注后的图片交给 Qwen-VL/vLLM 推理。

这个文件也是一行一个样本结果。每完成一个 batch，就会追加写入。

当前它还没到 6000 行，说明 ours 还没有完整跑完。随着 ours 继续运行，这个文件会持续变大。

等 ours 完整跑完后，脚本会额外生成最终结果文件，名字大致类似：

```text
proposed_vllm_results_260430_xxxx.json
```

如果中途停止，再次运行 ours 时会读取 `ours_vllm_resume.jsonl`，跳过已经完成的样本，从剩余样本继续跑。

### baseline_vs_ours_summary.md

这是 ours 跑完后新生成的正式对比汇总文件。

它由 `summarize_results.py` 同时读取下面两个文件生成：

```text
baseline_vllm_resume.jsonl
ours_vllm_resume.jsonl
```

它会把 baseline 和 ours 放在同一张表里，方便直接比较：

```text
完成数量
完成率
参与计分数量
正确数
准确率
文件更新时间
```

当前对比结果为：

```text
baseline: 6000/6000，MC 准确率 2746/3000 = 91.53%
ours:     6000/6000，MC 准确率 2746/3000 = 91.53%
```

也就是说，在当前默认 MC 自动评估指标下，ours 和 baseline 暂时打平。

注意：这个结论只代表 `mc` 选择题自动准确率。`qa` 开放问答默认只统计数量，不参与准确率。若要比较 QA，需要人工抽样或使用额外的语义评估方法。


## 这些文件是怎么来的

### baseline 相关文件

baseline 由下面的脚本生成：

```text
/mnt1/mnt2/data3/nlp/ws/proj/课题组/baseline.py
/mnt1/mnt2/data3/nlp/ws/proj/课题组/baseline_vllm.py
```

实际执行时，`baseline.py` 会调用 vLLM 版本的 `baseline_vllm.py`。

推理过程中会生成：

```text
baseline_vllm_resume.jsonl
baseline_vllm_trace.log
```

完整跑完后会生成：

```text
baseline_vllm_results_时间戳.json
baseline_vllm_run_summary.json
baseline_vllm_run_summary.md
```

### ours 相关文件

ours 由下面的脚本生成：

```text
/mnt1/mnt2/data3/nlp/ws/proj/课题组/ours.py
/mnt1/mnt2/data3/nlp/ws/proj/课题组/ours_vllm.py
```

实际执行时，`ours.py` 会调用 vLLM 版本的 `ours_vllm.py`。

推理过程中会生成：

```text
ours_vllm_resume.jsonl
```

完整跑完后会生成：

```text
proposed_vllm_results_时间戳.json
```

## 最终应该怎么对比 baseline 和 ours

等 ours 跑完以后，用下面两个文件做对比：

```text
baseline_vllm_resume.jsonl
ours_vllm_resume.jsonl
```

或者使用最终 JSON：

```text
baseline_vllm_results_260430_0907.json
proposed_vllm_results_时间戳.json
```

推荐用 `summarize_results.py` 生成统一汇总：

```bash
cd /mnt1/mnt2/data3/nlp/ws/proj/课题组

/mnt1/mnt2/data3/nlp/ws/uv_env/.venv/bin/python summarize_results.py \
  results/baseline_vllm_resume.jsonl \
  results/ours_vllm_resume.jsonl \
  --labels baseline ours \
  --dataset-file /mnt1/mnt2/data3/nlp/ws/course_data/dataset_divided_1/train.jsonl \
  --out-md results/baseline_vs_ours_summary.md
```

生成后重点看：

```text
baseline 的 MC 准确率
ours 的 MC 准确率
二者完成样本数是否都是 6000/6000
```

如果 ours 的准确率高于 baseline，并且完整跑完 6000 条，就可以作为更优方法采用。

## qa 和 mc 的评估指标

数据集里主要有两类任务：

```text
qa = Question Answering，开放问答题
mc = Multiple Choice，选择题
```

### mc 选择题如何评估

`mc` 是最容易自动评估的任务。

这类题一般有固定选项，例如：

```text
A. ...
B. ...
C. ...
D. ...
```

模型最终只需要回答一个选项字母，例如 `A`、`B`、`C` 或 `D`。

脚本会从模型输出里的 `Answer:` 部分提取第一个选项字母，然后和标准答案比较：

```text
模型提取答案 == 标准答案  -> 正确
模型提取答案 != 标准答案  -> 错误
```

因此 `mc` 的主要指标是准确率：

```text
MC Accuracy = MC 正确数量 / MC 总数量
```

例如 baseline 当前结果：

```text
MC 准确率：2746/3000 = 91.53%
```

这个指标比较可靠，因为选择题答案空间固定，自动判分不容易产生歧义。

### qa 开放问答如何评估

`qa` 是开放问答题，模型需要用自然语言直接回答。

例如：

```text
问题：图中天气状况如何？
标准答案：有雾，能见度较低。
模型回答 1：有雾，能见度较低。
模型回答 2：当前是雾天，可见距离较短。
模型回答 3：能见度不好，应该谨慎驾驶。
```

这三种回答在语义上可能都接近正确，但文字并不完全相同。

因此，`qa` 不适合像 `mc` 那样只用简单字符串相等来评估。否则会出现这种情况：

```text
标准答案：有雾，能见度较低
模型答案：雾天，视线较差
```

人看是对的，但简单字符串比较会判错。

所以当前默认汇总逻辑是：

```text
mc：参与准确率计算
qa：只统计数量，不默认参与准确率计算
```

这也是为什么汇总文件里会写：

```text
默认只把 mc 选择题纳入准确率；开放问答 qa 只统计数量。
```

### qa 的可选粗略自动评估

`summarize_results.py` 支持一个可选参数：

```text
--score-open
```

加上这个参数后，脚本会对 `qa` 做一个很粗略的自动评估：归一化完全匹配。

大致逻辑是：

```text
1. 把模型答案和标准答案转成小写
2. 去掉前后空白
3. 做简单规范化
4. 比较规范化后的文本是否完全一致
```

如果一致，就算正确；不一致，就算错误。

这个方法叫作：

```text
Normalized Exact Match，归一化完全匹配
```

它的优点是简单、可复现、速度快。

它的缺点也很明显：对开放问答不够公平。

例如下面这些语义相近的回答，可能仍然会被判为不一致：

```text
标准答案：有雾，能见度较低
模型答案：雾天，视线较差
```

所以 `--score-open` 只能作为粗略参考，不能完全代表 QA 的真实语义质量。

### 更合理的 qa 评估方式

如果后面要严谨比较 `qa`，更推荐以下方式：

```text
人工抽样评估
语义相似度评估
使用更强模型作为裁判评估
按关键词/实体/安全动作进行规则评分
```

对于当前实验，建议先以 `mc` 准确率作为主要自动指标，再人工检查一部分 `qa` 输出，确认 ours 是否在开放问答上也更合理。

### 当前新增的 qa 大模型裁判评估

现在已经新增了更细致的 QA 裁判评估脚本：

```text
evaluate_qa_with_qwen_judge.py
```

它使用部署在 A100 服务器上的 Qwen3.5-35B-A3B 多模态裁判模型，对 baseline 和 ours 的 QA 回复分别进行 1-100 分评分。

默认输入不是纯文本，而是：

```text
图片 + 问题 + 参考答案 + baseline 完整回复 + ours 完整回复
```

注意：QA 评估默认使用 `model_full_output`，也就是模型完整回复，而不是只使用 `extracted_answer`。这是因为 QA 题的 `extracted_answer` 有时会被抽成单个 `A/B/C/D` 字母，不适合用来评价开放问答质量。

正式输出文件为：

```text
qa_judge_qwen35_100_details.jsonl
qa_judge_qwen35_100_summary.md
qa_judge_qwen35_100_run.log
```

它们的作用分别是：

```text
qa_judge_qwen35_100_details.jsonl
逐题明细。每一行是一道 QA 题，包含图片路径、问题、参考答案、baseline 回复、ours 回复、两者总分、分项分数、优点、扣分点、胜负和裁判原始输出。

qa_judge_qwen35_100_summary.md
最终汇总。包含 baseline 和 ours 的平均分、总分、胜负统计、分项平均分和完整评分机制。

qa_judge_qwen35_100_run.log
运行日志。可以看当前进度、速度、预计剩余时间，以及是否有解析错误。
```

评分机制是 100 分制，分为五个维度：

```text
核心结论/操作是否正确：40 分
关键细节完整性：25 分
图像证据一致性：20 分
铁路安全/专业性：10 分
表达清晰与无幻觉：5 分
```

分数解释：

```text
90-100：几乎完全正确，关键细节充分，与图片和参考答案一致。
75-89：主要正确，只有少量次要遗漏或表述不够完整。
60-74：部分正确，抓住主方向，但遗漏多个关键点。
40-59：有少量相关内容，但核心操作/判断明显不完整或不可靠。
20-39：大部分错误、空泛、与题目关系弱，或存在明显安全/事实问题。
1-19：几乎无效；只输出 A/B/C/D、空答案、答非所问、严重矛盾时通常落在这个区间。
```

为了提速，脚本已经不是串行评估，而是并发请求 vLLM 服务：

```text
默认 workers=24
支持断点续跑
每 50 条打印一次速度和 ETA
```

如果中途停止，下次直接重新运行即可，它会读取已有的 `qa_judge_qwen35_100_details.jsonl`，跳过已经成功评估过的题目。

如果日志里看到类似下面的报错：

```text
JSONDecodeError: Expecting ',' delimiter
```

一般不是 vLLM 服务坏了，而是裁判模型某一次返回的 JSON 格式不够规范，或者 JSONL 断点文件里有一行不完整。当前脚本已经做了处理：

```text
1. 裁判返回非法 JSON 时，会自动要求裁判重新按合法 JSON 输出。
2. 读取断点文件时，如果遇到坏行，会跳过坏行并备份到 .bad_lines.jsonl。
3. 写入 JSONL 时使用文件锁，降低多个评估进程同时写入导致文件损坏的概率。
4. 只有包含 baseline_score 和 ours_score 的行才算完成；error 行不会进入最终汇总，后续会被补跑。
```

查看实时进度可以用：

```bash
tail -f /mnt1/mnt2/data3/nlp/ws/proj/课题组/results/qa_judge_qwen35_100_run.log
```

## 哪些文件可以忽略

当前可以暂时忽略：

```text
baseline_summary.json
baseline_summary.md
```

原因是它们是 baseline 早期只跑了 496 条时生成的旧汇总，不代表最终结果。

当前不能随便删除：

```text
baseline_vllm_resume.jsonl
ours_vllm_resume.jsonl
```

原因是它们是断点续跑文件。尤其是 `ours_vllm_resume.jsonl`，ours 还没跑完，删除后会丢失已完成进度。

## 推荐查看顺序

如果你只想快速看结论：

```text
1. baseline_vllm_run_summary.md
2. ours 跑完后生成的 baseline_vs_ours_summary.md
```

如果你想查具体某一道题：

```text
1. baseline_vllm_trace.log
2. baseline_vllm_resume.jsonl
3. ours_vllm_resume.jsonl
```

如果你要做最终提交或保存：

```text
1. baseline_vllm_results_260430_0907.json
2. proposed_vllm_results_时间戳.json
3. baseline_vs_ours_summary.md
```

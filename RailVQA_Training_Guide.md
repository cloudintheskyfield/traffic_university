# 基于 RailVQA 数据集的模型训练与微调指南 (2026版)

## 1. 当前任务所属的训练阶段

您手头的任务属于 **“多模态指令微调 (Visual Instruction Tuning / SFT - Supervised Fine-Tuning)”**。

结合您提供的 AI 训练技术栈全景图来看：
> - 底层：PyTorch / JAX / TensorFlow
> - 大模型预训练：Megatron-LM / DeepSpeed / FSDP / NeMo / TorchTitan
> - **👉 微调 (您所在的阶段)：Transformers + PEFT + Accelerate / LLaMA-Factory / Axolotl / Unsloth / torchtune**
> - 后训练 (RLHF/对齐)：TRL / verl / OpenRLHF / NeMo-Aligner
> - 传统 ML：scikit-learn / XGBoost / LightGBM / CatBoost

*   **预训练 (Pre-training)** 是从头教模型认识世界（从大量无标注数据中学习），成本极高。
*   **微调 (SFT)** 是教模型如何遵循人类的指令完成特定任务（如您的 `perception -> reasoning -> planning` 任务）。
*   **后训练 (Post-training / RLHF)** 通常指在 SFT 之后，为了让人类觉得答案更好或者更安全而做的强化学习对齐。由于您的数据集 `train.jsonl` 是成对的高质量思维链问答数据，直接做 SFT 即可。

## 2. 数据集字段与任务类型解析

您的 `train.jsonl` 是一个典型且高质量的 **多模态视觉问答 (VQA)** 数据集，且包含了 **思维链 (Chain-of-Thought, CoT)** 标注。

**字段对应关系：**
*   `image_id`: 对应的列车第一视角图像（如 `train_0001.jpg`）。
*   `cot_perception`: **感知 (Perception)** —— 描述环境、轨道、信号等客观视觉元素。
*   `cot_reasoning`: **推理 (Reasoning)** —— 结合物理规则和铁路安全的逻辑风险评估。
*   `cot_planning`: **规划 (Planning)** —— 给出的驾驶操作或行动建议。
*   `question` & `answer` (或选择题的 `options`): 针对当前画面的提问和最终答案。

**任务目标：** 
教给大模型一套固定的思考范式：看图 -> 感知细节 -> 逻辑推理 -> 规划动作 -> 给出结论。

## 3. 2026年最新开源多模态大模型 (LMM/VLM) 推荐

当前已是2026年，视觉多模态大模型（VLM）的发展极其迅速。对于要求“强逻辑推理”和“视觉细节捕捉”（例如看清远处的小信号灯）的列车驾驶场景，建议选择以下最新或主流的最强开源基座模型：

1.  **Qwen 系列 (如 Qwen3-VL 或 Qwen2.5-VL)**
    *   **推荐理由**：目前开源界的顶流之一，对中文和高分辨率图像细节理解极好。论文中评测采用的正是 `Qwen3-VL`，这说明该系列模型在铁路认知上表现出众。参数量推荐使用 7B/8B 左右的版本，对显存友好且能力强悍。
2.  **InternVL 系列 (如 InternVL 3.5 / InternVL 2.5)**
    *   **推荐理由**：上海AI实验室出品，采用动态分辨率技术，极其擅长处理高清图像和寻找图中的“小目标”（例如远处的行人或信号灯）。论文中也采用了 `InternVL 3.5` 进行评测。
3.  **STEP3-VL / MOSS-VL / Phi-4-reasoning-vision**
    *   **推荐理由**：2026年最新发布的前沿多模态模型代表，特别是在“多模态推理 (multimodal reasoning)”和“复杂视觉逻辑”方面实现了重大突破。例如 `Phi-4-reasoning-vision` 将高级推理能力引入视觉领域，非常契合您数据集中的 CoT (思维链) 任务。

**首选建议：直接采用论文验证过的 Qwen3-VL-8B 或 InternVL 3.5-8B 作为微调基座。**

## 4. 路线抉择：微调 YOLO 还是微调 VLM？

根据论文《RailVQA: A Benchmark and Framework...》的设计，实际上存在**两条不同的训练路线**，您可以根据自己的硬件资源和需求来选择：

### 路线 A：完全复现论文的“大小模型协同 (RailVQA-CoM)”框架
**操作方法**：
- **微调 YOLO**：只对轻量级的小模型（如论文中的 YOLO26m）进行微调，让它学会框出铁路场景中的人、车、障碍物。
- **冻结 VLM（不微调）**：不对大模型（如 Qwen3-VL）做任何训练（免训练推理 Training-free）。只通过写好的 Prompt，把 YOLO 吐出来的检测框、运动轨迹（日志文本）以及关键帧图片一起喂给大模型，让大模型根据文本日志做推理。
- **优点**：计算成本极低，只需要一张普通显卡微调 YOLO 即可；推理时因为大模型不用看完整的视频，只看关键帧，速度很快。
- **缺点**：需要自己写代码搭建“目标检测 -> 轨迹追踪 -> 文本日志生成 -> 大模型调用”这一整套复杂的工程管线。并且，**如果您只走这条路线，您手头这份包含了海量推理(Reasoning)和规划(Planning)文本的 `train.jsonl` 数据集就“无用武之地”了**，因为 YOLO 只能学习坐标框(x,y,w,h)，学不会文本逻辑。

### 路线 B：端到端微调 VLM (端到端视觉大模型路线)
**操作方法**：
- **抛弃 YOLO**，直接用您手头的 `train.jsonl`（包含了完整的图像和思维链问答对），对大型 VLM（如 Qwen3-VL-8B）进行 **LoRA 监督微调 (SFT)**。
- 让大模型自己用“眼睛”去看图片中的障碍物和信号灯，并自己输出 `Perception -> Reasoning -> Planning`。
- **优点**：工程极其简单，不需要挂载额外的 YOLO 和追踪算法；上限更高，因为大模型可以学习到数据集中复杂的端到端铁路驾驶逻辑。
- **缺点**：微调 VLM 需要较高的显存（通常需要 24GB 以上的显卡如 RTX 3090/4090），且推理整段视频时速度较慢。

### 从铁路“实际落地应用”的角度来看，哪种更合适？

**结论是：【路线 A（YOLO等小模型 + 大模型兜底校验）】远比【路线 B（纯端到端 VLM）】更适合当前的真实铁路落地。**

原因如下：

1. **极端的高频实时性要求 (Latency & FPS)**
   - **铁路现状**：高铁/动车运行速度极快（可达 350km/h，即近 100米/秒）。如果前方有突发障碍物，留给系统反应的时间通常只有毫秒级。
   - **VLM (路线B) 的致命伤**：即便是到了 2026 年，大模型（比如 8B 参数）处理一张高分辨率图片或视频流也需要数百毫秒到几秒钟的时间。这在真实的列车上是灾难性的延迟，极易导致刹车不及。
   - **小模型协同 (路线A) 的优势**：YOLO 跑在车载工控机上可以轻松达到 60~120 FPS。小模型可以极速、实时地扫描每一帧画面，遇到高危目标直接硬控刹车（或者输出运动日志给大模型做预警），**实时性得到了根本保障**。

2. **安全可追溯与确定性 (Safety & Determinism)**
   - **铁路现状**：铁路系统受到极度严格的安全监管（如 SIL4 级别认证）。任何系统作出的决策（加速/减速/紧急制动），都必须“说得清楚为什么”，算法不能是一个完全不可预测的黑盒。
   - **VLM (路线B) 的致命伤**：大模型本质上是概率模型（Next-token prediction），哪怕微调得再好，也存在一定的“幻觉 (Hallucination)”。它可能上一秒把铁轨看成了公路，或者无中生有地“推理”出一个实际上不存在的危险。
   - **小模型协同 (路线A) 的优势**：YOLO 的检测框、追踪库的轨迹、运动速度计算，这些都是**物理上明确、可测、可溯源的数据**。大模型只是作为一个“参谋”来看这份日志，即便大模型胡说八道，底层的规则引擎和小模型安全基线也能强制干预列车。

3. **车载计算资源限制 (Edge Computing Constraints)**
   - **铁路现状**：真实的列车车头（Cab）空间有限，电力系统、散热系统都不允许你装配像机房里那种重达几十斤、功耗几千瓦的 8 卡 H100 算力服务器。
   - **VLM (路线B) 的致命伤**：跑本地 VLM 极其耗费显存和算力，且在极端高温或震动的车载环境下非常脆弱。
   - **小模型协同 (路线A) 的优势**：小模型跑在几百块钱或几千块钱的车载边缘计算芯片（如 Nvidia Orin / 华为 Orin 等级）上绰绰有余。大模型可以部署在云端做延迟调度，或者选用极小规模参数在本地做后备判断（论文中正是为了降低算力压力，才让大模型只看抽取的少数“关键帧”）。

4. **长尾异常事件的兜底 (Corner Cases)**
   - 论文中提到“大模型主要用来解决长尾问题（比如一辆自行车掉铁轨上了，YOLO由于没见过所以漏检了）”。
   - **完美的工业闭环（路线A）**：日常 99.9% 的情况，YOLO 高效稳定地运行。当 YOLO 连续回报“啥也没有”，但列车雷达或摄像头有剧烈画面变动时（这就是论文里说的 *Defensive Prompting* 防御性提示），才唤醒大模型去做一次深度分析。这种“大小模型组合拳”才是自动驾驶界（不仅仅是火车，也包括汽车如特斯拉、华为ADS）目前最受推崇的实用落地架构。

**👉 工业落地总结**：
如果您是做**科研打比赛、发顶级AI论文、快速验证想法**，直接微调大模型（路线 B）最出效果。
如果您是要**交付给中车（CRRC）、国铁集团的真实车载系统**，绝对是选择**路线 A（YOLO高频检测 + 轨迹日志 + 大模型关键帧推理）**，这是目前兼顾“安全、实时、低算力”的唯一可行方案。

### 那么，如果是为了处理“异常情况 (Corner Cases)”偶尔用一次大模型，这份数据集有用武之地吗？

**答案是：有，而且这是这份数据集在工业界最核心的价值！**

在真实的“大小模型协同”架构中，小模型（YOLO）负责 99% 的常见情况，但它对没见过的东西（如掉在铁轨上的自行车、滑坡的泥石流）是“瞎子”。这个时候，系统会触发防御机制，**强行唤醒大模型来接管驾驶决策**。

要在这种危急时刻让大模型“靠谱”，大模型就必须经过专门的铁路知识微调。**您的这份 `train.jsonl` 就是用来训练这个“异常兜底大模型”的最佳教材！**
- 您可以通过微调，让大模型（如 Qwen3-VL）充分学习这套 `感知 -> 推理 -> 规划` 的铁路安全协议。
- 训练完成后，大模型日常处于休眠状态；一旦小模型判定出现异常，大模型立刻读取当前帧，给出符合铁路规范的高级推理决策。
- **结论**：这份 `train.jsonl` 不仅仅是评测试卷，更是用来**打造那个关键时刻能救命的“专家大脑”**的训练数据。

### 如果还要训练 YOLO，我还需要额外的数据吗？

**是的，绝对需要！**
如果您打算走**路线 A（大小模型协同）**，您手头这份 `train.jsonl` 只能用来训练/评测那个“大模型（专家大脑）”。
对于前面的“小模型（YOLO）”，您必须准备一份**完全不同格式**的数据集，通常是 COCO 或 YOLO 格式的 `.txt` 标注文件：
```text
# YOLO 格式示例：类别ID, 中心点X, 中心点Y, 宽度W, 高度H
0 0.45 0.60 0.10 0.20  # 比如 0 代表行人
1 0.70 0.80 0.05 0.15  # 比如 1 代表信号灯
```
**没有这种带有精准框坐标 (Bounding Box) 的标注数据，您是无法训练 YOLO 的。**（通常您需要用类似 LabelImg 或 CVAT 的工具，对视频帧一张张画框标注）。

---

## 5. 训练落地实操指南 (基于路线B：微调VLM专家大脑)

既然属于“微调阶段”，我们应当使用专门的微调框架，不需要写底层的 PyTorch/DeepSpeed 训练循环。

### 推荐工具
强烈推荐使用 **LLaMA-Factory** 或 **Swift (ModelScope)**。它们封装了底层的 `Transformers + PEFT (LoRA)`，支持“零代码” WebUI 启动，极大降低了多模态大模型的微调门槛。

### 步骤一：转换数据格式
框架通常不认识原生的自定义 JSONL。您需要写一段几十行的 Python 脚本，将您的 `train.jsonl` 转为框架支持的对话格式。
**转换核心**：将 `cot_perception`、`cot_reasoning`、`cot_planning` 拼接到一起，作为助手的最终回复。

**目标数据格式 (LLaVA 标准格式示例)：**
```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Please answer the question based on the image.\nQuestion: What should the train driver do when approaching a section with limited visibility due to fog, and no signals are clearly visible ahead?"
      },
      {
        "role": "assistant",
        "content": "Perception: The image shows a multi-track railway environment... visibility due to fog...\nReasoning: Given the reduced visibility, the driver must rely on onboard systems...\nPlanning: Reduce speed to a level appropriate...\nFinal Answer: The driver should reduce speed to a level that allows stopping within the visible distance..."
      }
    ],
    "images": ["/绝对路径/到/test_images/train_0001.jpg"]
  }
]
```

### 步骤二：环境准备与安装
以 LLaMA-Factory 为例：
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]
```

### 步骤三：注册与启动微调 (LoRA)
将转换好的 JSON 注册到框架的数据集配置中，然后通过 Web 界面 (`llamafactory-cli webui`) 勾选配置，或者使用命令行直接启动训练：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset railvqa_train_formatted \
    --template qwen2_vl \
    --finetuning_type lora \
    --output_dir saves/RailVQA-Qwen3-VL \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3
```

*(注：由于模型很大，采用 `finetuning_type lora` 可以大幅降低显存消耗，单张 24G/40G 显卡即可跑起来)*。

## 6. 总结

1.  **您的位置**：位于大模型生命周期的 **SFT (监督微调)** 阶段。
2.  **您的目标**：训练模型学会看图并输出“感知-推理-规划”的连贯思维链 (CoT)。
3.  **最优模型 (2026年视角)**：优先选择论文验证过的 **Qwen3-VL-8B** 或 **InternVL 3.5-8B**，或尝试最新一代的推理型多模态模型 (如 Phi-4-reasoning-vision)。
4.  **路线选择**：结合自身资源和应用目的，在 **微调 YOLO + 免训练大模型推理 (工程实用)** 与 **直接端到端微调大模型 (学术上限高)** 之间做出选择。
5.  **最佳实践**：如选择微调大模型，建议使用 **LLaMA-Factory** + **LoRA** 进行低成本高效微调，在此之前写个小脚本将您的原始 JSONL 数据打包为模型认识的标准多轮对话格式。

---

## 7. 附录：从源码 (`ours.py` vs `baseline.py`) 看两种路线的代码级实现

为了更直观地理解前面提到的 **路线 A（大小模型协同）** 与 **路线 B（纯大模型 Baseline）** 的区别，我们对比您课题组目录下的两个推理评测脚本：

### 1) `baseline.py`：传统的大模型直接看图（路线 B 的推理方式）
- **核心逻辑**：代码只加载了 `Qwen3-VL` 这一个模型。
- **输入处理**：在 `__getitem__` 中，直接读取原始的图像 `Image.open(image_file)`，然后丢给大模型。
- **Prompt 提示词**：系统提示词只说 `"Carefully analyze the provided railway scene image and answer the question..."`，要求它看原图并输出四步思维链。
- **总结**：这是非常标准的 VLM 玩法。如果要在真实列车上用这个方案，您**必须走路线 B**，提前用您的 `train.jsonl` 数据集去微调这个 `Qwen3-VL`，否则它可能根本不懂铁路规则，很容易胡说八道。

### 2) `ours.py`：论文提出的 RailVQA-CoM 大小模型协同（路线 A 的视觉提示机制）
- **核心逻辑**：代码**同时加载了两个模型**：`YOLO(yolo_path)` 和 `Qwen3-VL`。这就是典型的“小模型 + 大模型”。
- **输入处理 (非常巧妙)**：在 `__getitem__` 中，图片并没有直接丢给大模型！
  1. 首先把原图喂给小模型 YOLO：`results = self.yolo_model(image)`
  2. 利用 YOLO 的检测结果，**把检测框（Bounding Box）直接画在图片上**：`im_bgr = r.plot(line_width=2, font_size=10, labels=True)`
  3. 然后，把这张**带框的“加工图”**丢给大模型去推理。
- **Prompt 提示词**：系统提示词里特意加了一句非常关键的话：
  > *"The image contains visual bounding boxes highlighting potential objects (trains, signals, obstacles) generated by the AI. Use these boxes as references, visually verify their accuracy..."*
  翻译过来就是：**“图片上已经用框画出了列车、信号灯和障碍物，这是 AI 生成的，请你参考这些框来进行验证和答题。”**
- **总结**：这就是论文里提到的 **Visual Prompting（视觉提示）**。因为 YOLO 把难找的信号灯和行人都给框出来了，相当于给大模型“划了重点”，所以即使大模型没有做过专门的微调，也不太容易产生幻觉。在这个体系下，**你需要去微调 YOLO（让框画得更准），而大模型只是用来读带框的图并做逻辑推理（兜底专家）**。
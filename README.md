# PPO From Scratch

本项目基于 `ppo_train.py` 实现了一个面向大语言模型对齐训练的 PPO（Proximal Policy Optimization）最小可读版本。代码从 prompt 出发，使用 Actor 模型生成回答，通过 Reward Model 评估回答质量，并结合 Reference Model 的 KL 惩罚与 Critic 模型估计的价值函数，最终使用 PPO clipped objective 更新策略模型。

项目重点不是封装完整训练框架，而是把 RLHF/RLAIF 中 PPO 训练的关键数据流和数学逻辑显式写出来，便于理解每一步张量如何产生、如何参与损失计算，以及如何回传更新模型参数。

## 项目结构

```text
.
├── README.md
└── ppo_train.py
```

核心实现全部位于 `ppo_train.py`。

## 核心模块

- `PromptDataset`：构造 prompt 数据集，可选择使用 tokenizer 的 chat template。
- `Critic`：价值模型，基于 Actor 的 base model 加上线性 value head，输出每个生成 token 对应的价值估计。
- `Samples`：保存 Actor 采样生成后的序列、attention mask、action mask 等信息。
- `Experience`：保存 PPO 更新所需的旧策略 log probability、value、reward、advantage 和 return。
- `ExperienceBuffer`：暂存 rollout 得到的经验数据。
- `generate_samples()`：调用 Actor 生成 response。
- `generate_experiences()`：基于生成结果计算 reward、KL、advantage 和 return。
- `train_step()`：执行一次 Actor 和 Critic 的参数更新。
- `train()`：主训练循环，串联采样、经验构造、buffer、PPO 多轮训练。

## 模型角色

### Actor Model

Actor 是待训练的策略模型，代码中使用：

```python
actor_model = AutoModelForCausalLM.from_pretrained(...)
```

它定义当前策略：

```math
\pi_\theta(a_t \mid s_t)
```

其中 $s_t$ 表示当前上下文 token 序列，$a_t$ 表示下一步生成的 token。

### Reference Model

Reference Model 是 Actor 的初始副本，用于约束策略不要偏离初始语言模型太远：

```math
\pi_{ref}(a_t \mid s_t)
```

训练中不会用它反向传播，只用它计算 KL 惩罚。

### Reward Model

Reward Model 对 Actor 生成的完整文本打分：

```math
r_{rm} = R_\psi(x, y)
```

其中 $x$ 是 prompt，$y$ 是 response。当前实现中，Reward Model 输出的是结果奖励，会被加到 response 最后一个有效 token 上。

### Critic Model

Critic 用于估计每个生成 token 对应状态的价值：

```math
V_\phi(s_t)
```

代码中 `Critic` 复用 Actor 的 base model，并额外添加一个线性回归头：

```python
self.value_head = nn.Linear(base_model.config.hidden_size, 1)
```

其输出会被裁剪到 response 对应的 token 区间：

```python
values = value_model_output.squeeze(-1)[:, :-1][:, -num_actions:]
```

## PPO 数学原理

PPO 的目标是在提升奖励的同时限制策略更新幅度，避免一次更新导致策略分布发生过大变化。

设旧策略为 $\pi_{\theta_{\text{old}}}$，当前策略为 $\pi_\theta$。对于某个 response token $a_t$，策略概率比为：

```math
r_t(\theta)
= \frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
```

由于代码中使用 log probability 计算，因此实现形式为：

```math
r_t(\theta)
= \exp \left(
\log \pi_\theta(a_t \mid s_t)
- \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)
\right)
```

对应代码：

```python
ratio = (log_probs - old_log_probs).exp()
```

PPO clipped objective 为：

```math
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
```

训练时最小化负目标：

```math
L_{\pi}(\theta)
=
-\mathbb{E}_t
\left[
\min
\left(
r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
```

代码实现位于 `compute_policy_loss()`：

```python
surr1 = ratio * advantages
surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
loss = -torch.min(surr1, surr2)
```

其中 $\epsilon$ 对应 `clip_eps`，默认值为 `0.2`。

## KL 惩罚

在 RLHF/RLAIF 中，如果只最大化 Reward Model 的分数，Actor 可能会过度优化奖励模型，导致语言质量下降或偏离原模型分布。因此 PPO 训练通常会加入对 Reference Model 的 KL 约束。

当前代码使用 log probability 差值作为近似 KL：

```math
KL_t
\approx
\log \pi_\theta(a_t \mid s_t)
-
\log \pi_{ref}(a_t \mid s_t)
```

对应实现：

```python
log_ratio = log_probs.float() - ref_log_probs.float()
```

KL 惩罚被转换为 token-level reward：

```math
r^{kl}_t = -\beta \cdot KL_t
```

其中 $\beta$ 对应 `kl_ctl`。代码中默认：

```python
kl_ctl = 0.1
```

最终 token reward 由 KL 惩罚和 Reward Model 的结果奖励共同组成：

```math
r_t =
\begin{cases}
-\beta KL_t + \text{clip}(r_{rm}, -c, c), & t = T \\
-\beta KL_t, & t < T
\end{cases}
```

其中 $T$ 是 response 最后一个有效 token 的位置，$c$ 对应 `clip_reward_value`。

实现位于 `compute_rewards()`：

```python
kl_divergence_estimate = -kl_ctl * kl
rewards = kl_divergence_estimate
reward_clip = torch.clamp(r, -clip_reward_value, clip_reward_value)
rewards[j, :ends[j]][-1] += reward_clip[j, 0]
```

## GAE 优势估计

PPO 使用优势函数 $A_t$ 衡量某个动作相对于当前价值函数预期的好坏。

单步 TD error 为：

```math
\delta_t
=
r_t + \gamma V(s_{t+1}) - V(s_t)
```

GAE（Generalized Advantage Estimation）通过反向递推平衡 bias 和 variance：

```math
A_t
=
\delta_t + \gamma \lambda A_{t+1}
```

其中：

- $\gamma$ 是折扣因子。
- $\lambda$ 是 GAE 衰减系数。
- $A_{T+1}=0$。
- 最后一步默认 $V(s_{T+1})=0$。

回报由优势和价值相加得到：

```math
R_t = A_t + V(s_t)
```

代码实现位于 `get_advantages_and_returns()`：

```python
delta = rewards[:, t] + gamma * nextvalues - values[:, t]
lastgaelam = delta + gamma * lambd * lastgaelam
advantages = torch.stack(advantages_reversed[::-1], dim=1)
returns = advantages + values
```

当前脚本中使用：

```python
gamma = 0.1
lambd = 0.2
```

这两个值更偏教学演示。实际 RLHF 训练中常见设置通常会使用更大的 $\gamma$ 和 $\lambda$，例如 $\gamma = 1.0$、$\lambda = 0.95$。

## Value Loss

Critic 的训练目标是让价值估计 $V_\phi(s_t)$ 拟合 GAE 得到的 return：

```math
L_V(\phi)
=
\mathbb{E}_t
\left[
\left(
V_\phi(s_t) - R_t
\right)^2
\right]
```

代码实现位于 `compute_value_loss()`：

```python
loss = (values - returns) ** 2
```

函数也支持 value clipping。若启用 `clip_eps`，会先计算：

```math
V^{clip}_t
=
V^{old}_t
+
\text{clip}
\left(
V_t - V^{old}_t,
-\epsilon,
\epsilon
\right)
```

然后取 clipped value loss 和普通 value loss 的较大值：

```math
L_V
=
\max
\left(
(V^{clip}_t - R_t)^2,
(V_t - R_t)^2
\right)
```

当前 `train_step()` 调用时没有传入 `clip_eps`，因此实际使用的是普通 MSE value loss。

## 算法流程

完整训练流程如下：

1. 初始化 Actor、Reference、Reward Model 和 Critic。
2. 使用 `PromptDataset` 和 `DataLoader` 构造 prompt batch。
3. Actor 根据 prompt batch 生成 response。
4. 对生成序列重新计算 Actor 对每个 action token 的 log probability。
5. 使用 Reference Model 计算相同 action token 的 log probability。
6. 使用 Critic 预测每个 action token 对应的 value。
7. 将完整文本 decode 后送入 Reward Model，得到结果奖励。
8. 计算 Actor 与 Reference 的近似 KL。
9. 将 KL 惩罚和 Reward Model 分数合成为 token-level rewards。
10. 使用 GAE 反向计算 advantages 和 returns。
11. 将经验写入 `ExperienceBuffer`。
12. 从 buffer 构造训练 batch。
13. 多轮执行 PPO 更新：
    - 使用 clipped policy loss 更新 Actor。
    - 使用 value loss 更新 Critic。
14. 记录 TensorBoard 日志。
15. 清空 buffer，进入下一轮 rollout。

## 数据流

整体数据流：

```text
prompt_list
  -> PromptDataset
  -> prompts_dataloader
  -> generate_samples()
  -> Samples
  -> generate_experiences()
  -> Experience
  -> ExperienceBuffer
  -> DataLoader + collate_fn()
  -> BufferItem
  -> train_step()
  -> actor_model / critic_model 参数更新
```

训练中的主要张量如下：

```text
seqs:              [batch, max_length + max_new_tokens]
attention_mask:    [batch, max_length + max_new_tokens]
action_mask:       [batch, max_new_tokens]
action_log_probs:  [batch, num_actions]
ref_log_probs:     [batch, num_actions]
kl:                [batch, num_actions]
values:            [batch, num_actions]
rewards:           [batch, num_actions]
advantages:        [batch, num_actions]
returns:           [batch, num_actions]
```

其中：

- `seqs` 是 prompt 和 response 拼接后的完整 token 序列。
- `attention_mask` 标记非 padding token。
- `action_mask` 只标记 response 中的有效 token。
- `action_log_probs` 是 rollout 时 Actor 对生成 token 的 log probability。
- `values` 是 rollout 时 Critic 对 response token 的价值估计。
- `advantages` 和 `returns` 是 PPO 更新阶段直接使用的训练目标。

## 具体实现路径

### 1. Prompt 构造

入口数据是脚本中的 `prompt_list`：

```python
prompt_list = [
    '请问1+1等于多少？',
    'PowerShell，如何知道BIOS中的虚拟化是否已禁用',
    ...
]
```

随后构造 dataset：

```python
prompts_dataset = PromptDataset(
    prompt_list,
    actor_tokenizer,
    apply_chat_template=True,
)
```

当 `apply_chat_template=True` 时，prompt 会被包装成对话格式：

```python
content = [{"role": "user", "content": prompt}]
prompt = self.tokenizer.apply_chat_template(
    content,
    tokenize=False,
    add_generation_prompt=True,
)
```

### 2. 生成样本

`generate_samples()` 会将 prompt tokenize 后送入 Actor：

```python
inputs = actor_tokenizer(
    prompts,
    padding='max_length',
    max_length=max_length,
    truncation=True,
    return_tensors='pt',
)
```

然后调用：

```python
seqs = model.generate(
    **inputs.to(device),
    max_new_tokens=max_new_tokens,
    eos_token_id=eos_token_id,
    pad_token_id=pad_token_id,
)
```

生成后的序列会被统一处理成固定长度：

```text
total_length = max_length + max_new_tokens
```

prompt 之后的 token 被视为 action：

```python
ans = seqs[:, input_ids.size(1):]
action_mask = ans.ne(pad_token_id).to(dtype=torch.long)
```

### 3. 计算旧策略概率

PPO 更新需要旧策略概率。当前实现中，采样后立即用 Actor 对完整序列重新前向，得到 rollout 时的 `action_log_probs`：

```python
output = actor_model(seqs, attention_mask=attention_mask)
logits = output.logits
log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
log_probs_labels = log_probs.gather(
    dim=-1,
    index=seqs[:, 1:].unsqueeze(-1),
)
action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
```

这里使用 `logits[:, :-1, :]` 预测 `seqs[:, 1:]`，符合自回归语言模型的 next-token prediction 方式。

### 4. 计算 Reference 概率和 KL

Reference Model 对同一批 `seqs` 做前向：

```python
ref_output = ref_model(seqs, attention_mask=attention_mask)
ref_logits = ref_output.logits
ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
```

然后抽取生成 token 对应的 log probability：

```python
ref_action_log_probs = ref_log_probs_labels.squeeze(-1)[:, -num_actions:]
```

最后计算近似 KL：

```python
kl = compute_approx_kl(
    action_log_probs,
    ref_action_log_probs,
    action_mask=action_mask,
)
```

### 5. 计算 Reward

代码先把完整 token 序列 decode 为文本：

```python
seq_texts = actor_tokenizer.batch_decode(
    seqs,
    skip_special_tokens=True,
)
```

然后送入 Reward Model：

```python
reward_model_inputs = reward_tokenizer(
    seq_texts,
    return_tensors="pt",
    padding=True,
)
r = reward_model(**reward_model_inputs.to(device)).logits
```

最终调用 `compute_rewards()` 将结果奖励与 KL 惩罚合并。

### 6. 计算 Advantage 和 Return

Critic 先输出每个 action token 的 value：

```python
value = critic_model.forward(
    seqs,
    attention_mask,
    num_actions,
)
```

再通过 GAE 计算：

```python
advantages, returns = get_advantages_and_returns(
    value,
    rewards,
    action_mask,
    gamma=0.1,
    lambd=0.2,
)
```

这两个张量会被保存到 `Experience`，用于后续 PPO 更新。

### 7. Actor 更新

`train_step()` 中重新计算当前 Actor 的 `action_log_probs`，并与 rollout 时保存的 `old_action_log_probs` 比较：

```python
policy_loss = compute_policy_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    action_mask=action_mask,
)
```

随后反向传播并更新 Actor：

```python
policy_loss.backward()
optimizer_actor.step()
```

### 8. Critic 更新

Critic 重新预测 value：

```python
values = critic_model.forward(
    sequences,
    attention_mask,
    num_actions,
)
```

然后计算 value loss：

```python
value_loss = compute_value_loss(
    values,
    old_values,
    returns,
    action_mask,
)
```

最后更新 Critic：

```python
value_loss.backward()
optimizer_critic.step()
```

## 超参数说明

脚本入口中定义了主要训练超参数：

```python
episodes = 3
max_epochs = 5
rollout_batch_size = 8
micro_rollout_batch_size = 2
n_samples_per_prompt = 2
max_new_tokens = 50
max_length = 256
micro_train_batch_size = 2
```

含义如下：

- `episodes`：整体训练轮数。
- `max_epochs`：每次 rollout 后，对同一批经验重复训练的轮数。
- `rollout_batch_size`：每次从 prompt 数据集中取多少条 prompt。
- `micro_rollout_batch_size`：生成 response 时的小 batch，用于降低显存压力。
- `n_samples_per_prompt`：每个 prompt 采样多少个 response。
- `max_new_tokens`：每个 response 最多生成多少个 token。
- `max_length`：prompt tokenize 后的最大长度。
- `micro_train_batch_size`：PPO 参数更新时的小 batch。

优化器设置：

```python
optimizer_actor = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
optimizer_critic = torch.optim.Adam(critic_model.parameters(), lr=0.00005)
```

日志写入：

```python
writer = SummaryWriter('./runs')
```

## 运行方式

当前脚本中的模型路径是本地硬编码路径：

```python
actor_model = AutoModelForCausalLM.from_pretrained(
    '/home/user/Downloads/Qwen2.5-0.5B-Instruct'
)
ref_model = AutoModelForCausalLM.from_pretrained(
    '/home/user/Downloads/Qwen2.5-0.5B-Instruct'
)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    '/home/user/Downloads/reward-model-deberta-v3-large-v2'
)
```

运行前需要确保这些路径存在，或替换成你自己的本地模型路径 / Hugging Face 模型名称。

安装依赖后运行：

```bash
python ppo_train.py
```

查看 TensorBoard：

```bash
tensorboard --logdir ./runs
```

训练过程中会打印：

```text
step: <step>  policy_loss: <policy_loss>  value_loss: <value_loss>
```

同时 `policy_loss` 和 `value_loss` 会写入 `./runs`。

## 代码执行链路

从入口到训练的调用链路如下：

```text
if __name__ == "__main__"
  -> 初始化 device / 超参数
  -> 加载 actor_model / ref_model / reward_model
  -> 初始化 actor_tokenizer / reward_tokenizer
  -> 构造 critic_model
  -> 初始化 optimizer_actor / optimizer_critic
  -> 构造 PromptDataset
  -> 构造 prompts_dataloader
  -> train()
```

`train()` 内部链路如下：

```text
train()
  -> ExperienceBuffer(limit=100)
  -> for episode in range(episodes)
    -> for rand_prompts in prompts_dataloader
      -> generate_samples()
      -> generate_experiences()
      -> buffer.append()
      -> DataLoader(buffer, collate_fn=collate_fn)
      -> for epoch in range(max_epochs)
        -> for experience in dataloader
          -> train_step()
      -> buffer.clear()
```

## 实现特点

- 使用纯 PyTorch + Transformers 展示 PPO 训练核心逻辑。
- 将 response token 显式作为 action 处理。
- 使用 Reference Model 的 log probability 构造 KL 惩罚。
- 使用 Reward Model 的 scalar reward 作为结果奖励。
- 使用 GAE 计算 token-level advantage。
- 使用 PPO clipped objective 更新 Actor。
- 使用 value loss 更新 Critic。
- 使用 TensorBoard 记录训练损失。

## 注意事项与可改进方向

当前实现偏教学和实验性质，便于理解 PPO/RLHF 的核心路径。实际训练中可以继续改进：

- Reference Model 建议显式冻结参数，并在推理时保持 `eval()`。
- Reward Model 也应显式冻结参数。
- Actor 和 Critic 当前共享 `actor_model.base_model`，训练时需要注意参数共享带来的梯度影响。
- 当前没有 checkpoint 保存逻辑，可加入 `save_pretrained()` 保存 Actor 和 tokenizer。
- 当前 prompt 数据写死在脚本里，可扩展为读取 JSON/JSONL 数据集。
- 可以加入 advantage normalization，提升 PPO 训练稳定性。
- 可以加入 gradient clipping，降低梯度爆炸风险。
- 可以加入 entropy bonus，提升探索性。
- 可以将 `gamma`、`lambd`、`kl_ctl`、学习率等改为命令行参数。
- 可以使用更细粒度的 KL 统计、reward 统计和生成长度统计辅助调试。

## 训练轮数、经验训练、rollout

首先设置总迭代轮数 episodes。每个 episode 中，用当前策略模型 Actor 生成一批经验。由于生成经验成本较高，所以不会生成完只训练一次，而是把这批经验放入 buffer，用同一批经验训练 max_epochs 轮。
在生成经验时，每个 rollout 总共取 rollout_batch_size 条 prompt；但由于一次性处理这么多 prompt 会占用较大显存，所以实际会按照 micro_rollout_batch_size 拆分成若干个小批次依次生成。每个 prompt 会生成 n_samples_per_prompt 个回答，最后把这些小批次的结果合并成完整的一批 rollout 经验。

例如，结合参数：

episodes = 3

max_epochs = 5

rollout_batch_size = 8

micro_rollout_batch_size = 2

n_samples_per_prompt = 2

意思是：

总共做 3 个 episode。

每个 episode：
    
    取 8 条 prompt 生成经验。
    
    但生成时不是一次处理 8 条，而是每次处理 2 条。
    
    所以需要处理 8 / 2 = 4 次 micro rollout。

    每条 prompt 生成 2 个 response。
    
    所以最终得到 8 × 2 = 16 条 response 经验。

    然后用这 16 条经验训练 5 个 epoch。
    
    训练完后丢弃这批经验，进入下一个 episode，重新生成新经验。

## 总结

本项目展示了一个从零实现的语言模型 PPO 训练闭环：

```text
生成 response
  -> 计算奖励与 KL
  -> 估计优势和回报
  -> PPO 更新策略
  -> 训练 Critic
```

它覆盖了 RLHF/RLAIF 中最关键的组件：Actor、Reference、Reward Model、Critic、KL penalty、GAE 和 PPO clipped loss。通过阅读 `ppo_train.py` 和本 README，可以清楚追踪每个训练张量从哪里来、如何被使用，以及最终如何影响模型参数更新。



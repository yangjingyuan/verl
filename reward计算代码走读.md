# Reward计算代码走读

> 本文档详细梳理verl框架中reward计算的完整流程，包括架构设计、数据流、调用栈和关键实现。

## 目录

- [1. 概览](#1-概览)
- [2. 整体架构](#2-整体架构)
- [3. Reward Manager体系](#3-reward-manager体系)
- [4. Rule-Based Reward](#4-rule-based-reward)
- [5. Model-Based Reward](#5-model-based-reward)
- [6. 数据流与DataProto](#6-数据流与dataproto)
- [7. 训练循环集成](#7-训练循环集成)
- [8. 配置系统](#8-配置系统)
- [9. 关键文件索引](#9-关键文件索引)

---

## 1. 概览

### 1.1 Reward在RL训练中的作用

在PPO/GRPO等强化学习算法中，reward信号是驱动策略优化的核心：

```
训练迭代流程:
Rollout (生成) → Reward (评分) → Advantage (计算优势) → Update (更新策略)
```

verl支持两类reward计算方式：
- **Rule-Based Reward**: 基于规则的评分函数（如答案匹配、代码执行结果）
- **Model-Based Reward**: 使用神经网络reward model预测分数

### 1.2 核心设计理念

- **单控制器多工作器**: Controller调度，Worker分布式计算
- **DataProto抽象**: 统一的数据容器，支持tensor和非tensor数据传递
- **插件化扩展**: 通过Registry机制注册自定义reward manager
- **异步并行**: 支持异步计算reward以提升吞吐

---

## 2. 整体架构

### 2.1 核心流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      Training Loop                          │
│                  (ray_trainer.py:fit())                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   ┌─────────┐      ┌──────────────┐     ┌──────────┐
   │ Rollout │      │    Reward    │     │ Advantage│
   │ Worker  │      │   Manager    │     │ Compute  │
   └─────────┘      └──────────────┘     └──────────┘
        │                   │                   │
        │            ┌──────┴──────┐           │
        │            ↓             ↓           │
        │     ┌────────────┐ ┌──────────┐     │
        │     │ Rule-Based │ │  Model   │     │
        │     │   Reward   │ │  Reward  │     │
        │     └────────────┘ └──────────┘     │
        │                   │                   │
        └───────────────────┴───────────────────┘
                            ↓
                    ┌───────────────┐
                    │ Actor/Critic  │
                    │    Update     │
                    └───────────────┘
```

### 2.2 时序图

```
Controller          ActorRollout       RewardManager      RewardModelWorker
    │                    │                    │                   │
    │──generate─────→   │                    │                   │
    │←─responses────────│                    │                   │
    │                    │                    │                   │
    │─────────────compute_rm_score(batch)────────────────────→  │
    │←──────────────rm_scores────────────────────────────────────│
    │                    │                    │                   │
    │──compute_reward(batch)──────────────→  │                   │
    │                    │            ┌───────┴────────┐         │
    │                    │            │ for each sample│         │
    │                    │            │ · extract data │         │
    │                    │            │ · call scorer  │         │
    │                    │            │ · return score │         │
    │                    │            └───────┬────────┘         │
    │←─────reward_tensor─────────────────────│                   │
    │                    │                    │                   │
    │──apply_kl_penalty─→                    │                   │
    │──compute_advantage─→                   │                   │
    │                    │                    │                   │
    │──update_policy────→                    │                   │
```

---

## 3. Reward Manager体系

### 3.1 抽象基类: AbstractRewardManager

**文件位置**: `verl/workers/reward_manager/abstract.py:27-73`

```python
class AbstractRewardManager(ABC):
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Callable,
        reward_fn_key: str = "data_source"
    ):
        """
        Args:
            tokenizer: 分词器，用于解码token序列
            num_examine: 检查并打印的样本数（调试用）
            compute_score: 评分函数，计算单个样本的reward
            reward_fn_key: 从non_tensor_batch中提取数据源标识的键
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    @abstractmethod
    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        核心方法：将DataProto转换为reward张量

        Returns:
            - 如果return_dict=False: torch.Tensor (batch_size, seq_len)
            - 如果return_dict=True: {
                "reward_tensor": torch.Tensor,
                "reward_extra_info": dict  # 额外信息如acc, passed等
              }
        """
        pass
```

**关键方法**:

```python
def _extract_reward_from_rm_scores(self, data: DataProto, return_dict: bool = False):
    """
    检查DataProto中是否已有预计算的rm_scores

    如果存在，直接返回；否则返回None，表示需要计算
    """
    if "rm_scores" not in data.batch.keys():
        return None

    # 提取并处理rm_scores...
    return reward_tensor
```

### 3.2 具体实现类

#### A. NaiveRewardManager (逐样本计算)

**文件位置**: `verl/workers/reward_manager/naive.py:26-122`

**特点**:
- 逐样本串行计算reward
- 适用于简单的评分逻辑
- 默认使用的reward manager

**核心逻辑**:

```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __call__(self, data: DataProto, return_dict: bool = False):
        # 1. 检查是否已有rm_scores
        result = self._extract_reward_from_rm_scores(data, return_dict)
        if result is not None:
            return result

        # 2. 遍历batch中的每个样本
        all_reward = []
        all_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]  # 获取单个样本的DataProtoItem

            # 3. 提取关键信息
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]

            # 解码为文本
            prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # 提取ground truth和数据源
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # 4. 调用评分函数
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info
            )

            # 5. 处理返回值
            if isinstance(score, dict):
                reward_val = score["score"]
                for k, v in score.items():
                    if k != "score":
                        all_extra_info[k].append(v)
            else:
                reward_val = float(score)

            all_reward.append(reward_val)

        # 6. 转换为token级别的reward tensor
        reward_tensor = self._to_tensor(data, all_reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(all_extra_info)
            }
        return reward_tensor

    def _to_tensor(self, data: DataProto, scores: list[float]):
        """
        将标量score转换为token级别的reward

        策略：只在response的最后一个有效token处赋值
        """
        batch_size = len(data)
        response_length = data.batch["responses"].shape[1]

        reward_tensor = torch.zeros(batch_size, response_length)

        for i, score in enumerate(scores):
            # 找到最后一个有效token的位置
            attention_mask = data[i].batch["attention_mask"]
            last_valid_idx = attention_mask.sum() - 1

            # 只在最后位置赋值
            reward_tensor[i, last_valid_idx] = score

        return reward_tensor
```

**调用示例**:

```python
# 配置中指定
reward_model:
  reward_manager:
    name: "naive"  # 使用NaiveRewardManager

# 在训练中调用
reward_tensor = naive_manager(batch, return_dict=False)
# reward_tensor.shape: (batch_size, seq_len)
# 只有每个sequence最后一个token处有非零值
```

#### B. BatchRewardManager (批量计算)

**文件位置**: `verl/workers/reward_manager/batch.py:25-128`

**特点**:
- 批量并行计算reward
- compute_score函数接收整个batch的数据列表
- 适用于可向量化的评分逻辑

**核心差异**:

```python
@register("batch")
class BatchRewardManager(AbstractRewardManager):
    def __call__(self, data: DataProto, return_dict: bool = False):
        # 1. 批量提取所有样本的数据
        all_prompts = []
        all_responses = []
        all_ground_truths = []
        all_data_sources = []

        for i in range(len(data)):
            data_item = data[i]
            all_prompts.append(self.tokenizer.decode(...))
            all_responses.append(self.tokenizer.decode(...))
            all_ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
            all_data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])

        # 2. 批量调用compute_score
        scores = self.compute_score(
            data_sources=all_data_sources,      # 列表
            solution_strs=all_responses,        # 列表
            ground_truths=all_ground_truths,    # 列表
            extra_infos=[...]                   # 列表
        )
        # 返回：list[float] 或 list[dict]

        # 3. 转换为tensor
        reward_tensor = self._to_tensor(data, scores)
        return reward_tensor
```

**适用场景**:
- 需要批量调用外部API（如GPT评分）
- 可以利用GPU向量化加速的评分函数
- 需要跨样本进行归一化的场景

#### C. DAPORewardManager (带惩罚)

**文件位置**: `verl/workers/reward_manager/dapo.py:25-149`

**特点**:
- 继承自NaiveRewardManager
- 对过长的response施加长度惩罚
- 专为DAPO算法设计

**核心逻辑**:

```python
@register("dapo")
class DAPORewardManager(NaiveRewardManager):
    def __init__(self, max_response_length: int = 512, length_penalty: float = -1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_response_length = max_response_length
        self.length_penalty = length_penalty

    def __call__(self, data: DataProto, return_dict: bool = False):
        # 1. 调用父类获取基础reward
        result = super().__call__(data, return_dict=True)
        reward_tensor = result["reward_tensor"]

        # 2. 应用长度惩罚
        for i in range(len(data)):
            response_length = data[i].batch["responses"].shape[0]

            if response_length > self.max_response_length:
                # 对所有token应用惩罚
                reward_tensor[i, :] += self.length_penalty

        if return_dict:
            result["reward_tensor"] = reward_tensor
            return result
        return reward_tensor
```

#### D. PrimeRewardManager (异步并行)

**文件位置**: `verl/workers/reward_manager/prime.py`

**特点**:
- 使用ProcessPoolExecutor实现多进程并行
- 适用于compute_score计算密集的场景
- 显著提升大batch的处理速度

**实现概要**:

```python
@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    def __init__(self, max_workers: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

    def __call__(self, data: DataProto, return_dict: bool = False):
        # 1. 提交所有样本到进程池
        futures = []
        for i in range(len(data)):
            future = self.executor.submit(
                self._compute_single_sample,
                data[i], self.compute_score
            )
            futures.append(future)

        # 2. 收集结果
        scores = [future.result() for future in futures]

        # 3. 转换为tensor
        reward_tensor = self._to_tensor(data, scores)
        return reward_tensor
```

### 3.3 Registry机制

**文件位置**: `verl/workers/reward_manager/registry.py:20-56`

```python
# 全局注册表
REWARD_MANAGER_REGISTRY: dict[str, type[AbstractRewardManager]] = {}

def register(name: str) -> Callable:
    """
    装饰器：注册reward manager类到全局registry

    使用示例:
        @register("my_custom_manager")
        class MyCustomRewardManager(AbstractRewardManager):
            ...
    """
    def decorator(cls: type[AbstractRewardManager]) -> type[AbstractRewardManager]:
        if name in REWARD_MANAGER_REGISTRY:
            raise ValueError(f"Reward manager '{name}' already registered")

        REWARD_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator

def get_reward_manager_cls(name: str) -> type[AbstractRewardManager]:
    """
    从registry获取reward manager类

    Args:
        name: 注册名称（如 "naive", "batch", "dapo"）

    Returns:
        RewardManager类（未实例化）

    Raises:
        ValueError: 如果name未注册
    """
    if name not in REWARD_MANAGER_REGISTRY:
        available = ", ".join(REWARD_MANAGER_REGISTRY.keys())
        raise ValueError(
            f"Unknown reward manager: '{name}'. "
            f"Available: {available}"
        )

    return REWARD_MANAGER_REGISTRY[name]
```

**当前已注册的managers**:
- `"naive"` → NaiveRewardManager
- `"batch"` → BatchRewardManager
- `"dapo"` → DAPORewardManager
- `"prime"` → PrimeRewardManager

---

## 4. Rule-Based Reward

### 4.1 核心入口: default_compute_score

**文件位置**: `verl/utils/reward_score/__init__.py:19-114`

```python
def default_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    sandbox_fusion_url: Optional[str] = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = 1024
) -> float | dict[str, Any]:
    """
    根据data_source路由到对应的评分函数

    Args:
        data_source: 数据集标识（如 "openai/gsm8k", "lighteval/MATH"）
        solution_str: 模型生成的答案文本
        ground_truth: 真实答案
        extra_info: 额外信息（如rollout_reward_scores, num_turns等）
        sandbox_fusion_url: 代码沙箱URL（用于代码评分）
        concurrent_semaphore: 并发控制信号量
        memory_limit_mb: 沙箱内存限制

    Returns:
        - float: 标量分数（如 0.0 或 1.0）
        - dict: 包含 "score" 键及其他额外信息
    """

    # 路由逻辑
    if data_source == "openai/gsm8k":
        return compute_score_for_gsm8k(solution_str, ground_truth)

    elif data_source == "lighteval/MATH":
        return compute_score_for_math(solution_str, ground_truth)

    elif data_source.startswith("numina_"):
        return compute_score_for_prime_math(solution_str, ground_truth, data_source)

    elif data_source in ["codecontests", "apps", "taco"]:
        # 代码评分，需要沙箱执行
        if sandbox_fusion_url is None:
            raise ValueError(f"Sandbox URL required for {data_source}")

        return compute_score_with_sandbox(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            sandbox_url=sandbox_fusion_url,
            semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb
        )

    elif data_source == "geometry3k":
        return compute_score_for_geo3k(solution_str, ground_truth)

    elif data_source.startswith("searchR1_"):
        return compute_score_for_qa_em(solution_str, ground_truth)

    else:
        raise ValueError(f"Unsupported data_source: {data_source}")
```

### 4.2 具体评分函数示例

#### A. GSM8K数学评分

**文件位置**: `verl/utils/reward_score/gsm8k.py:25-68`

```python
def compute_score_for_gsm8k(solution_str: str, ground_truth: str) -> float:
    """
    GSM8K数据集的评分函数

    策略：
    1. 从solution_str中提取最终答案数字
    2. 与ground_truth进行数值比较
    3. 返回1.0（正确）或0.0（错误）
    """

    # 1. 提取答案
    # 查找 "####" 标记后的数字（GSM8K标准格式）
    if "####" in solution_str:
        answer_str = solution_str.split("####")[-1].strip()
    else:
        # 尝试提取最后出现的数字
        answer_str = extract_last_number(solution_str)

    # 2. 清理和规范化
    answer_str = answer_str.replace(",", "")  # 移除千位分隔符
    ground_truth = ground_truth.replace(",", "")

    # 3. 比较
    try:
        answer_num = float(answer_str)
        gt_num = float(ground_truth)

        # 允许小的浮点误差
        if abs(answer_num - gt_num) < 1e-6:
            return 1.0
        else:
            return 0.0
    except (ValueError, TypeError):
        return 0.0
```

**输入输出示例**:

```python
# 正确答案
solution = "First, we calculate 5 * 3 = 15. Then add 2 to get #### 17"
ground_truth = "17"
score = compute_score_for_gsm8k(solution, ground_truth)  # 返回 1.0

# 错误答案
solution = "The answer is #### 20"
ground_truth = "17"
score = compute_score_for_gsm8k(solution, ground_truth)  # 返回 0.0
```

#### B. MATH数据集评分

**文件位置**: `verl/utils/reward_score/math_reward.py:24-89`

```python
def compute_score_for_math(solution_str: str, ground_truth: str) -> dict[str, float]:
    """
    MATH数据集的评分函数

    特点：
    - 支持LaTeX格式的数学表达式
    - 使用符号数学库进行等价性检查
    - 返回详细的评分信息
    """

    # 1. 提取\\boxed{}中的答案（MATH标准格式）
    pred_answer = extract_boxed_answer(solution_str)

    # 2. 规范化数学表达式
    pred_normalized = normalize_math_expr(pred_answer)
    gt_normalized = normalize_math_expr(ground_truth)

    # 3. 符号数学比较
    try:
        from sympy import simplify, sympify

        pred_expr = sympify(pred_normalized)
        gt_expr = sympify(gt_normalized)

        # 检查等价性
        is_equivalent = simplify(pred_expr - gt_expr) == 0

        score = 1.0 if is_equivalent else 0.0
    except Exception:
        # 降级为字符串比较
        score = 1.0 if pred_normalized == gt_normalized else 0.0

    return {
        "score": score,
        "pred_answer": pred_answer,
        "gt_answer": ground_truth
    }
```

#### C. 代码评分（沙箱执行）

**文件位置**: `verl/utils/reward_score/sandbox_fusion.py:25-156`

```python
def compute_score_with_sandbox(
    data_source: str,
    solution_str: str,
    ground_truth: dict,  # 包含test_cases等信息
    sandbox_url: str,
    semaphore: Optional[threading.Semaphore] = None,
    memory_limit_mb: int = 1024,
    timeout_sec: int = 10
) -> dict[str, Any]:
    """
    在安全沙箱中执行代码并评分

    流程：
    1. 提取代码块
    2. 构造测试用例
    3. 在沙箱中执行
    4. 检查输出是否匹配
    """

    # 1. 提取代码
    code = extract_code_block(solution_str)

    # 2. 构造测试请求
    test_cases = ground_truth.get("test_cases", [])

    # 3. 并发控制
    if semaphore is not None:
        semaphore.acquire()

    try:
        # 4. 调用沙箱服务
        response = requests.post(
            f"{sandbox_url}/execute",
            json={
                "code": code,
                "test_cases": test_cases,
                "language": "python",
                "memory_limit_mb": memory_limit_mb,
                "timeout_sec": timeout_sec
            },
            timeout=timeout_sec + 5
        )

        result = response.json()

        # 5. 评分
        passed = result.get("all_passed", False)
        num_passed = result.get("num_passed", 0)
        num_total = result.get("num_total", len(test_cases))

        score = 1.0 if passed else 0.0

        return {
            "score": score,
            "passed": passed,
            "num_passed": num_passed,
            "num_total": num_total,
            "errors": result.get("errors", [])
        }

    finally:
        if semaphore is not None:
            semaphore.release()
```

**配置沙箱**:

```yaml
reward_model:
  sandbox_fusion:
    url: "http://localhost:8000"
    max_concurrent: 64
    memory_limit_mb: 1024
```

### 4.3 支持的数据集总览

| 数据集 | data_source标识 | 评分模块 | 评分方式 |
|--------|----------------|----------|----------|
| GSM8K | `openai/gsm8k` | `gsm8k.py` | 数值匹配 |
| MATH | `lighteval/MATH` | `math_reward.py` | 符号数学等价 |
| MATH DAPO | `math_dapo` | `math_dapo.py` | 数值+长度惩罚 |
| Numina Math | `numina_*` | `prime_math.py` | 综合数学评分 |
| CodeContests | `codecontests` | `sandbox_fusion.py` | 沙箱执行 |
| APPS | `apps` | `sandbox_fusion.py` | 沙箱执行 |
| TACO | `taco` | `sandbox_fusion.py` | 沙箱执行 |
| Geometry3K | `geometry3k` | `geo3k.py` | 几何答案匹配 |
| QA任务 | `searchR1_*` | `search_r1_like_qa_em.py` | 精确匹配 |

---

## 5. Model-Based Reward

### 5.1 RewardModelWorker

**文件位置**: `verl/workers/fsdp_workers.py:1654-2009`

**架构图**:

```
RewardModelWorker
│
├─ __init__()                 # 初始化配置
│
├─ _build_model()             # 构建reward model
│  ├─ AutoModelForTokenClassification.from_pretrained()
│  ├─ apply_fsdp_wrapping()  # FSDP/FSDP2包装
│  └─ model.eval()           # 设置为评估模式
│
├─ compute_rm_score()         # 主入口（注册为分布式dispatch）
│  ├─ _switch_chat_template() # 切换tokenizer模板（可选）
│  ├─ _forward_micro_batch()  # 微批次前向传播
│  │  ├─ 数据加载到GPU
│  │  ├─ model.forward()
│  │  ├─ 提取最后有效token的logits
│  │  └─ 返回scores (micro_batch_size,)
│  ├─ _expand_to_token_level() # 扩展为token级别
│  └─ 返回 DataProto({"rm_scores": ...})
│
└─ _switch_chat_template()    # 处理不同tokenizer格式
```

### 5.2 核心实现

#### A. 模型构建

```python
def _build_model(self):
    """
    加载并包装reward model

    支持的模型类型：
    - AutoModelForTokenClassification (HuggingFace)
    - 自定义reward head
    """

    # 1. 加载预训练模型
    model = AutoModelForTokenClassification.from_pretrained(
        self.config.model.path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 2. 配置FSDP包装
    if self.config.fsdp_config.fsdp_type == "fsdp2":
        # 使用FSDP2（PyTorch 2.5+）
        from torch.distributed._composable.fsdp import fully_shard_module

        fully_shard_module(
            model,
            mesh=self.mesh,
            reshard_after_forward=True,
            offload_policy=self.config.fsdp_config.offload_policy
        )
    else:
        # 使用FSDP1
        from torch.distributed.fsdp import FullyShardedDataParallel

        model = FullyShardedDataParallel(
            model,
            sharding_strategy=self.config.fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device()
        )

    # 3. 设置为评估模式
    model.eval()

    return model
```

#### B. 分布式计算

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
def compute_rm_score(self, data: DataProto) -> DataProto:
    """
    计算reward model的分数

    Dispatch模式说明：
    - make_nd_compute_dataproto_dispatch_fn: 自动处理分布式数据分片
    - mesh_name="reward": 使用名为"reward"的设备mesh

    流程：
    1. Controller调用此方法
    2. 框架自动将data按DP维度切分
    3. 分发到各个worker执行
    4. 收集结果并拼接回controller
    """

    # 1. 可选：切换chat template
    if self.config.get("switch_chat_template", False):
        data = self._switch_chat_template(data)

    # 2. 移动数据到GPU
    data = data.to(torch.cuda.current_device())

    # 3. 微批次处理
    micro_batch_size = self.config.get("rm_micro_batch_size", 4)
    all_scores = []

    for i in range(0, len(data), micro_batch_size):
        micro_batch = data[i:i + micro_batch_size]
        scores = self._forward_micro_batch(micro_batch)
        all_scores.append(scores)

    all_scores = torch.cat(all_scores, dim=0)  # (batch_size,)

    # 4. 扩展为token级别
    token_level_scores = self._expand_to_token_level(data, all_scores)
    # token_level_scores.shape: (batch_size, response_length)

    # 5. 返回DataProto
    return DataProto.from_dict(
        tensors={"rm_scores": token_level_scores}
    )
```

#### C. 微批次前向传播

```python
def _forward_micro_batch(self, micro_batch: DataProto) -> torch.Tensor:
    """
    对微批次执行前向传播

    优化：
    - remove_padding: 移除padding以节省计算
    - Ulysses序列并行: 处理超长序列
    """

    # 1. 提取输入
    input_ids = micro_batch.batch["input_ids"]
    attention_mask = micro_batch.batch["attention_mask"]
    position_ids = micro_batch.batch.get("position_ids", None)

    # 2. 可选：移除padding
    if self.config.get("remove_padding", False):
        input_ids, attention_mask, position_ids = remove_padding_for_batch(
            input_ids, attention_mask, position_ids
        )

    # 3. 前向传播
    with torch.no_grad():
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

    # 4. 提取logits
    logits = outputs.logits  # (batch_size, seq_len, num_classes)

    # 5. 提取最后有效token的分数
    scores = []
    for i in range(len(micro_batch)):
        # 找到最后一个有效token
        seq_len = attention_mask[i].sum().item()
        last_token_idx = seq_len - 1

        # 提取该位置的logit
        score = logits[i, last_token_idx, 0].item()  # 假设分数在class 0
        scores.append(score)

    return torch.tensor(scores, device=input_ids.device)
```

#### D. 扩展为Token级别

```python
def _expand_to_token_level(
    self,
    data: DataProto,
    scores: torch.Tensor
) -> torch.Tensor:
    """
    将标量score扩展为token级别的reward

    策略：只在response的最后一个token处赋值
    （与rule-based reward保持一致）
    """
    batch_size = len(data)
    response_length = data.batch["responses"].shape[1]

    token_level_scores = torch.zeros(
        batch_size, response_length,
        device=scores.device,
        dtype=scores.dtype
    )

    for i in range(batch_size):
        # 找到response的最后一个有效token
        response_mask = data[i].batch["attention_mask"]
        last_idx = response_mask.sum() - 1

        # 赋值
        token_level_scores[i, last_idx] = scores[i]

    return token_level_scores
```

### 5.3 配置示例

```yaml
reward_model:
  enable: true
  enable_resource_pool: false  # 独立资源池

  # 分布式配置
  n_gpus_per_node: 4
  nnodes: 1

  # 模型配置
  model:
    path: "OpenAssistant/reward-model-deberta-v3-large-v2"

  # FSDP配置
  fsdp_config:
    fsdp_type: "fsdp2"
    sharding_strategy: "FULL_SHARD"
    offload_policy: false

  # 计算配置
  rm_micro_batch_size: 4
  remove_padding: true
  switch_chat_template: false
```

### 5.4 与Rule-Based Reward的集成

在训练循环中，两种reward可以组合使用：

```python
# 1. 先计算model-based reward
if self.use_rm:
    rm_scores = self.rm_wg.compute_rm_score(batch)
    batch = batch.union(rm_scores)  # 添加rm_scores到batch

# 2. 再计算rule-based reward
reward_tensor, extra_info = self.reward_fn(batch, return_dict=True)

# 3. 融合策略（在reward manager中实现）
# 如果batch中已有rm_scores，可以直接使用，或与rule-based结合
```

---

## 6. 数据流与DataProto

### 6.1 DataProto结构

**文件位置**: `verl/protocol.py`

```python
@dataclass
class DataProto:
    """
    统一的数据容器，支持tensor和非tensor数据传递

    在reward计算中的作用：
    - 封装prompt、response、ground_truth等信息
    - 在controller和worker之间高效传递
    - 支持索引、切片、拼接等操作
    """

    batch: TensorDict                # tensor数据
    non_tensor_batch: dict           # 非tensor数据
    meta_info: dict = field(default_factory=dict)  # 元信息
```

### 6.2 Reward相关的DataProto键

#### A. 输入键（reward manager消费）

```python
# Tensor数据
batch.batch = {
    "prompts": torch.Tensor,              # (batch_size, prompt_len)
    "responses": torch.Tensor,            # (batch_size, response_len)
    "input_ids": torch.Tensor,            # (batch_size, seq_len) = prompts + responses
    "attention_mask": torch.Tensor,       # (batch_size, seq_len)
    "position_ids": torch.Tensor,         # (batch_size, seq_len)
}

# 非Tensor数据
batch.non_tensor_batch = {
    "data_source": list[str],             # 数据集标识
    # 示例: ["openai/gsm8k", "openai/gsm8k", "lighteval/MATH", ...]

    "reward_model": list[dict],           # reward配置
    # 示例: [
    #   {"ground_truth": "42", "style": "rule"},
    #   {"ground_truth": "17", "style": "rule"},
    #   ...
    # ]

    "extra_info": list[dict],             # 额外信息
    # 示例: [
    #   {"num_turns": 1, "difficulty": "easy"},
    #   {"num_turns": 2, "difficulty": "hard"},
    #   ...
    # ]

    "uid": list[str],                     # 样本唯一标识
    # 示例: ["sample_001", "sample_002", ...]

    "rollout_reward_scores": list[dict],  # 已有的reward分数（可选）
    # 示例: [
    #   {"intermediate_reward": 0.5},
    #   {"intermediate_reward": 0.8},
    #   ...
    # ]
}

# 元信息
batch.meta_info = {
    "global_steps": int,                  # 全局训练步数
    "is_validation": bool,                # 是否验证集
    "epoch": int,                         # 当前epoch
}
```

#### B. 输出键（reward manager产生）

```python
# 从reward manager返回
{
    "reward_tensor": torch.Tensor,        # (batch_size, seq_len)
    # token级别的reward，通常只在最后一个token处有非零值

    "reward_extra_info": dict,            # 额外评分信息
    # 示例: {
    #   "acc": [1, 0, 1, 1, 0, ...],       # 每个样本的准确率
    #   "passed": [True, False, True, ...],# 是否通过测试
    #   "num_passed": [10, 5, 10, ...],    # 通过的测试用例数
    # }
}

# 后续在训练循环中添加
batch.batch["token_level_scores"] = reward_tensor      # Line 1588
batch.batch["rm_scores"] = ...                         # model-based reward
batch.batch["token_level_rewards"] = ...               # KL惩罚后的reward
batch.batch["advantages"] = ...                        # GAE计算的优势
batch.batch["returns"] = ...                           # 用于critic的return
```

### 6.3 DataProto的关键操作

#### A. 创建DataProto

```python
# 从字典创建
batch = DataProto.from_single_dict({
    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    "data_source": ["openai/gsm8k", "openai/gsm8k"],
    "reward_model": [
        {"ground_truth": "42"},
        {"ground_truth": "17"}
    ]
})
```

#### B. 索引和切片

```python
# 单个样本
data_item = batch[0]  # 返回 DataProtoItem
print(data_item.batch["input_ids"])  # 单个样本的input_ids
print(data_item.non_tensor_batch["data_source"])  # "openai/gsm8k"

# 批量切片
sub_batch = batch[10:20]  # 返回 DataProto，包含10个样本
sub_batch = batch[[1, 3, 5]]  # 选择特定样本
```

#### C. 合并DataProto

```python
# Union操作（用于添加新字段）
reward_result = DataProto.from_dict(
    tensors={"rm_scores": torch.tensor([...])}
)
batch = batch.union(reward_result)
# 现在 batch.batch["rm_scores"] 可用
```

#### D. 分块处理

```python
# 将batch分为8块（用于分布式计算）
chunks = batch.chunk(num_chunks=8)

# 每个worker处理一块
for i, chunk in enumerate(chunks):
    result = worker_group.compute_rm_score(chunk)
```

### 6.4 数据流完整示例

```
初始化 (Controller)
├─ 从dataloader加载batch_dict
├─ batch = DataProto.from_single_dict(batch_dict)
└─ batch包含：prompts, responses, data_source, reward_model, etc.

↓

生成阶段 (Rollout Workers)
├─ gen_output = rollout_wg.generate_sequences(batch)
├─ gen_output包含：responses, log_probs, attention_mask
└─ batch = batch.union(gen_output)

↓

Model-Based Reward (可选)
├─ rm_output = rm_wg.compute_rm_score(batch)
├─ rm_output包含：rm_scores (batch_size, seq_len)
└─ batch = batch.union(rm_output)

↓

Rule-Based Reward (Controller)
├─ reward_tensor, extra_info = compute_reward(batch, reward_fn)
├─ reward_tensor: (batch_size, seq_len)
├─ extra_info: {"acc": [...], "passed": [...]}
└─ batch.batch["token_level_scores"] = reward_tensor
   batch.non_tensor_batch.update(extra_info)

↓

KL惩罚 (Controller)
├─ old_log_probs = batch.batch["old_log_probs"]
├─ new_log_probs = batch.batch["log_probs"]
├─ kl_divergence = old_log_probs - new_log_probs
├─ token_level_rewards = token_level_scores - beta * kl_divergence
└─ batch.batch["token_level_rewards"] = token_level_rewards

↓

Advantage计算 (Controller)
├─ values = batch.batch["values"]  # 从critic获取
├─ advantages, returns = compute_gae(
│     rewards=token_level_rewards,
│     values=values,
│     gamma=0.99, lam=0.95
│  )
└─ batch.batch["advantages"] = advantages
   batch.batch["returns"] = returns

↓

策略更新 (Actor/Critic Workers)
├─ actor_wg.update_policy(batch)  # 使用advantages
└─ critic_wg.update_value(batch)  # 使用returns
```

---

## 7. 训练循环集成

### 7.1 Reward Manager加载

**文件位置**: `verl/trainer/ppo/reward.py:99-176`

```python
def load_reward_manager(
    config,
    tokenizer,
    num_examine: int = 0,
    **reward_kwargs
) -> AbstractRewardManager:
    """
    根据配置加载reward manager

    Args:
        config: 训练配置对象
        tokenizer: 分词器
        num_examine: 打印检查的样本数
        **reward_kwargs: 额外的reward manager参数

    Returns:
        实例化的reward manager对象
    """

    reward_manager_cfg = config.reward_model.reward_manager

    # 1. 获取自定义reward函数（如果存在）
    custom_reward_fn = None
    if hasattr(config, "custom_reward_function") and config.custom_reward_function:
        custom_reward_fn = load_extern_object(
            module_path=config.custom_reward_function.module,
            object_name=config.custom_reward_function.name
        )

    # 2. 获取reward manager类
    if reward_manager_cfg.source == "register":
        # 从registry获取
        reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        # 从外部模块导入
        reward_manager_cls = load_extern_object(
            module_path=reward_manager_cfg.module.path,
            object_name=reward_manager_cfg.module.name
        )
    else:
        raise ValueError(f"Unknown source: {reward_manager_cfg.source}")

    # 3. 处理sandbox fusion配置（代码评分）
    sandbox_cfg = config.reward_model.get("sandbox_fusion", {})
    sandbox_url = sandbox_cfg.get("url", None)

    if sandbox_url:
        # 创建并发控制信号量
        max_concurrent = sandbox_cfg.get("max_concurrent", 64)
        _concurrent_semaphore = threading.Semaphore(max_concurrent)

        # 包装compute_score函数
        final_compute_score = partial(
            default_compute_score,
            sandbox_fusion_url=sandbox_url,
            concurrent_semaphore=_concurrent_semaphore,
            memory_limit_mb=sandbox_cfg.get("memory_limit_mb", 1024)
        )
    else:
        final_compute_score = default_compute_score

    # 4. 使用自定义reward函数（如果提供）
    if custom_reward_fn is not None:
        final_compute_score = custom_reward_fn

    # 5. 实例化reward manager
    reward_manager = reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs
    )

    return reward_manager
```

**在main_ppo.py中的调用**:

```python
# verl/trainer/main_ppo.py:318-323

# 加载训练用的reward manager
reward_fn = load_reward_manager(
    config,
    tokenizer,
    num_examine=0,
    **config.reward_model.get("reward_kwargs", {})
)

# 加载验证用的reward manager（会打印1个样本）
val_reward_fn = load_reward_manager(
    config,
    tokenizer,
    num_examine=1,
    **config.reward_model.get("reward_kwargs", {})
)
```

### 7.2 训练循环中的Reward计算

**文件位置**: `verl/trainer/ppo/ray_trainer.py:1349-1749`

#### A. 主训练循环结构

```python
def fit(self):
    """PPO/GRPO训练的主循环"""

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:

            # ==== 1. ROLLOUT: 生成序列 ====
            batch = DataProto.from_single_dict(batch_dict)
            gen_output = self.actor_rollout_wg.generate_sequences(batch)
            batch = batch.union(gen_output)

            # ==== 2. COMPUTE: 获取log probs和values ====
            old_log_probs = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_probs)

            if self.use_critic:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

            # ==== 3. REWARD: 计算reward (核心部分) ====
            with marked_timer("reward", timing_raw, color="yellow"):

                # 3.1 Model-based reward (如果启用)
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    if not self.use_reward_loop:
                        rm_scores = self.rm_wg.compute_rm_score(batch)
                    else:
                        rm_scores = self.reward_loop_manager.compute_rm_score(batch)

                    batch = batch.union(rm_scores)

                # 3.2 Rule-based reward
                if self.config.reward_model.launch_reward_fn_async:
                    # 异步计算（不阻塞后续操作）
                    future_reward = compute_reward_async.remote(
                        data=batch,
                        config=self.config,
                        tokenizer=self.tokenizer,
                        reward_fn=None  # 会在worker中重新加载
                    )
                else:
                    # 同步计算
                    reward_tensor, reward_extra_infos = self._compute_or_extract_reward(
                        batch,
                        reward_fn=self.reward_fn,
                        reward_for_val=False
                    )

            # ==== 4. ADVANTAGE: 计算优势函数 ====
            with marked_timer("adv", timing_raw, color="brown"):

                # 4.1 等待异步reward结果（如果使用异步模式）
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos = ray.get(future_reward)

                # 4.2 设置token_level_scores
                batch.batch["token_level_scores"] = reward_tensor

                # 4.3 更新额外信息
                if reward_extra_infos:
                    batch.non_tensor_batch.update({
                        k: np.array(v) for k, v in reward_extra_infos.items()
                    })

                # 4.4 应用KL惩罚（如果启用）
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch,
                        kl_ctrl=self.kl_ctrl,
                        kl_penalty=self.config.algorithm.kl_penalty
                    )
                else:
                    # 直接使用scores作为rewards
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # 4.5 计算advantages
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam
                )

            # ==== 5. UPDATE: 更新actor和critic ====
            with marked_timer("update", timing_raw, color="green"):

                for ppo_epoch in range(self.config.algorithm.ppo_epochs):
                    # 更新actor
                    actor_output = self.actor_wg.update_actor(batch)

                    # 更新critic（如果启用）
                    if self.use_critic:
                        critic_output = self.critic_wg.update_critic(batch)

            # ==== 6. LOGGING: 记录指标 ====
            self._log_metrics(batch, reward_extra_infos, actor_output, critic_output)
```

#### B. _compute_or_extract_reward方法

```python
def _compute_or_extract_reward(
    self,
    batch: DataProto,
    reward_fn: AbstractRewardManager,
    reward_for_val: bool = False
) -> tuple[torch.Tensor, dict]:
    """
    计算或提取reward

    Args:
        batch: 数据批次
        reward_fn: reward manager实例
        reward_for_val: 是否用于验证（影响打印行为）

    Returns:
        (reward_tensor, reward_extra_infos)
    """

    # 调用reward manager
    try:
        result = reward_fn(batch, return_dict=True)
        reward_tensor = result["reward_tensor"]
        reward_extra_infos = result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        # 降级处理：不返回额外信息
        reward_tensor = reward_fn(batch, return_dict=False)
        reward_extra_infos = {}

    return reward_tensor, reward_extra_infos
```

#### C. 异步Reward计算

```python
# verl/trainer/ppo/reward.py:206-217

@ray.remote(num_cpus=1)
def compute_reward_async(
    data: DataProto,
    config=None,
    tokenizer=None,
    reward_fn=None
):
    """
    在单独的Ray worker中异步计算reward

    优点：
    - 不阻塞主训练流程
    - 可以在计算advantage时并行执行
    - 适用于compute_score计算密集的场景
    """

    # 如果没有提供reward_fn，重新加载
    if reward_fn is None:
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0)

    return compute_reward(data, reward_fn)


# 使用方式
future_reward = compute_reward_async.remote(
    data=batch,
    config=self.config,
    tokenizer=self.tokenizer
)

# ... 执行其他操作 ...

# 等待结果
reward_tensor, reward_extra_infos = ray.get(future_reward)
```

### 7.3 验证循环中的Reward计算

**文件位置**: `verl/trainer/ppo/ray_trainer.py:1751-1892`

```python
def _validate(self):
    """验证循环"""

    all_scores = []
    all_extra_infos = defaultdict(list)

    for test_batch_dict in self.val_dataloader:
        # 1. 创建DataProto
        test_batch = DataProto.from_single_dict(test_batch_dict)

        # 2. 重复样本（增加生成多样性）
        n_repeats = self.config.trainer.get("val_n_repeats", 1)
        if n_repeats > 1:
            test_batch = test_batch.repeat(n_repeats, interleave=True)

        # 3. 生成序列
        test_output = self.actor_rollout_wg.generate_sequences(test_batch)
        test_batch = test_batch.union(test_output)

        # 4. 计算reward
        reward_tensor, reward_extra_infos = self._compute_or_extract_reward(
            test_batch,
            reward_fn=self.val_reward_fn,  # 使用验证reward manager
            reward_for_val=True
        )

        # 5. 聚合分数
        scores = reward_tensor.sum(dim=-1)  # 对token维度求和
        all_scores.extend(scores.tolist())

        for k, v in reward_extra_infos.items():
            all_extra_infos[k].extend(v)

    # 6. 计算平均指标
    avg_score = np.mean(all_scores)
    avg_acc = np.mean(all_extra_infos.get("acc", []))

    print(f"Validation - Avg Score: {avg_score:.4f}, Avg Acc: {avg_acc:.4f}")

    return {
        "val/avg_score": avg_score,
        "val/avg_acc": avg_acc,
        **{f"val/{k}": np.mean(v) for k, v in all_extra_infos.items()}
    }
```

### 7.4 完整调用栈（同步模式）

```
verl/trainer/main_ppo.py:318
└─ load_reward_manager(config, tokenizer)
   └─ reward_fn = NaiveRewardManager(...)

训练时:
verl/trainer/ppo/ray_trainer.py:1526
└─ _compute_or_extract_reward(batch, reward_fn)
   └─ verl/trainer/ppo/reward.py:189
      └─ compute_reward(batch, reward_fn)
         └─ reward_fn(data, return_dict=True)
            └─ verl/workers/reward_manager/naive.py:46
               └─ NaiveRewardManager.__call__(data)
                  ├─ 遍历batch中的每个样本
                  ├─ 提取prompt/response/ground_truth
                  ├─ 调用self.compute_score()
                  │  └─ verl/utils/reward_score/__init__.py:61
                     └─ default_compute_score(data_source, ...)
                        └─ 根据data_source路由到具体评分函数
                           └─ verl/utils/reward_score/gsm8k.py:25
                              └─ compute_score_for_gsm8k(...)
                                 └─ 返回 1.0 或 0.0
                  └─ _to_tensor(data, all_reward)
                     └─ 返回 (batch_size, seq_len) 的tensor
```

### 7.5 完整调用栈（异步模式）

```
verl/trainer/ppo/ray_trainer.py:1520
└─ future_reward = compute_reward_async.remote(data=batch, ...)
   └─ Ray创建远程任务
      └─ verl/trainer/ppo/reward.py:206
         └─ compute_reward_async函数在Ray worker中执行
            ├─ load_reward_manager(config, tokenizer)
            └─ compute_reward(data, reward_fn)
               └─ [同上，与同步模式相同的调用栈]

... 主线程继续执行其他操作 ...

verl/trainer/ppo/ray_trainer.py:1584
└─ reward_tensor, reward_extra_infos = ray.get(future_reward)
   └─ 阻塞等待远程任务完成，获取结果
```

---

## 8. 配置系统

### 8.1 配置文件结构

**文件位置**: `verl/trainer/config/`

```
config/
├── ppo_trainer/
│   ├── default.yaml              # 基础配置
│   ├── algorithm/
│   │   ├── gae.yaml              # GAE优势估计
│   │   ├── grpo.yaml             # GRPO算法
│   │   └── reinforce.yaml        # REINFORCE
│   ├── actor_rollout_ref/        # Actor和rollout配置
│   ├── critic/                   # Critic配置
│   ├── data/                     # 数据配置
│   └── reward_model/             # Reward配置
│       ├── default.yaml
│       ├── gsm8k.yaml
│       ├── math.yaml
│       └── code.yaml
└── config.py                     # 配置数据类定义
```

### 8.2 核心配置类

**文件位置**: `verl/trainer/config/config.py:98-130`

```python
@dataclass
class RewardManagerConfig(BaseConfig):
    """Reward Manager配置"""

    source: str = "register"
    # 来源类型：
    #   - "register": 从REWARD_MANAGER_REGISTRY获取
    #   - "importlib": 从外部模块导入

    name: str = "naive"
    # Reward manager名称（当source="register"时）
    # 可选值: "naive", "batch", "dapo", "prime"

    module: Optional[ModuleConfig] = None
    # 外部模块配置（当source="importlib"时）
    # 示例: ModuleConfig(path="my_module.reward", name="MyRewardManager")


@dataclass
class RewardModelConfig(BaseConfig):
    """Reward Model配置"""

    # ==== 基础配置 ====
    enable: bool = False
    # 是否启用model-based reward

    launch_reward_fn_async: bool = False
    # 是否异步计算rule-based reward

    # ==== 分布式配置 ====
    enable_resource_pool: bool = False
    # 是否使用独立的资源池（与actor/critic隔离）

    n_gpus_per_node: int = 4
    # 每个节点的GPU数量

    nnodes: int = 1
    # 节点数量

    # ==== Reward Manager配置 ====
    reward_manager: RewardManagerConfig = field(
        default_factory=lambda: RewardManagerConfig(
            source="register",
            name="naive"
        )
    )

    reward_kwargs: dict = field(default_factory=dict)
    # 传递给reward manager的额外参数
    # 示例: {"max_response_length": 512, "length_penalty": -1.0}

    # ==== Model配置（用于model-based reward）====
    model: Optional[ModelConfig] = None
    # 示例: ModelConfig(path="OpenAssistant/reward-model-deberta-v3-large")

    # ==== Sandbox配置（用于代码评分）====
    sandbox_fusion: dict = field(default_factory=dict)
    # 示例: {
    #   "url": "http://localhost:8000",
    #   "max_concurrent": 64,
    #   "memory_limit_mb": 1024
    # }


@dataclass
class DataConfig(BaseConfig):
    """数据配置"""

    reward_fn_key: str = "data_source"
    # 从non_tensor_batch中提取数据源标识的键名

    train_files: str = ""
    # 训练数据文件路径（支持glob）

    val_files: str = ""
    # 验证数据文件路径
```

### 8.3 配置示例

#### A. GSM8K配置

```yaml
# verl/trainer/config/reward_model/gsm8k.yaml

reward_model:
  # 不启用model-based reward
  enable: false

  # 使用同步计算
  launch_reward_fn_async: false

  # 使用naive reward manager
  reward_manager:
    source: "register"
    name: "naive"

  # 无需sandbox
  sandbox_fusion: {}

data:
  reward_fn_key: "data_source"
  train_files: "${HOME}/data/gsm8k/train.parquet"
  val_files: "${HOME}/data/gsm8k/test.parquet"
```

**使用方式**:

```bash
python -m verl.trainer.main_ppo \
    --config-name=default \
    reward_model=gsm8k \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
```

#### B. 代码评分配置

```yaml
# verl/trainer/config/reward_model/code.yaml

reward_model:
  # 不启用model-based reward
  enable: false

  # 使用异步计算（代码执行较慢）
  launch_reward_fn_async: true

  # 使用prime reward manager（多进程并行）
  reward_manager:
    source: "register"
    name: "prime"

  reward_kwargs:
    max_workers: 8  # 8个进程并行

  # 配置sandbox
  sandbox_fusion:
    url: "http://localhost:8000"
    max_concurrent: 64
    memory_limit_mb: 1024

data:
  reward_fn_key: "data_source"
  train_files: "${HOME}/data/codecontests/train.parquet"
  val_files: "${HOME}/data/codecontests/test.parquet"
```

#### C. Model-Based Reward配置

```yaml
# verl/trainer/config/reward_model/model_rm.yaml

reward_model:
  # 启用model-based reward
  enable: true

  # 使用独立资源池
  enable_resource_pool: true

  # 分布式配置
  n_gpus_per_node: 4
  nnodes: 1

  # Reward model配置
  model:
    path: "OpenAssistant/reward-model-deberta-v3-large-v2"

  # FSDP配置
  fsdp_config:
    fsdp_type: "fsdp2"
    sharding_strategy: "FULL_SHARD"

  # 计算配置
  rm_micro_batch_size: 4
  remove_padding: true

  # 仍然使用rule-based reward作为辅助
  reward_manager:
    source: "register"
    name: "naive"
```

#### D. 组合配置（Model + Rule）

```yaml
# 同时使用model-based和rule-based reward

reward_model:
  # 启用model-based reward
  enable: true

  model:
    path: "OpenAssistant/reward-model-deberta-v3-large-v2"

  # 同时使用rule-based reward
  reward_manager:
    source: "register"
    name: "naive"

# 在训练循环中的行为：
# 1. 先计算rm_scores（model-based）
# 2. 再计算rule-based reward
# 3. 在reward manager中可以访问rm_scores进行融合
```

### 8.4 自定义Reward函数

#### A. 通过配置指定

```yaml
# config.yaml

custom_reward_function:
  module:
    path: "my_project.rewards"
    name: "compute_custom_score"

reward_model:
  reward_manager:
    source: "register"
    name: "naive"  # 使用naive manager，但compute_score是自定义的
```

#### B. 自定义函数实现

```python
# my_project/rewards.py

def compute_custom_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs
) -> float | dict:
    """
    自定义评分函数

    要求：
    - 签名必须匹配default_compute_score
    - 返回float或包含"score"键的dict
    """

    # 自定义逻辑
    if data_source == "my_custom_task":
        # 实现自定义评分
        score = my_custom_logic(solution_str, ground_truth)
        return {
            "score": score,
            "custom_metric": some_value
        }
    else:
        # 降级到默认评分
        from verl.utils.reward_score import default_compute_score
        return default_compute_score(data_source, solution_str, ground_truth, extra_info)
```

#### C. 自定义Reward Manager

```python
# my_project/custom_manager.py

from verl.workers.reward_manager import AbstractRewardManager, register

@register("my_custom_manager")
class MyCustomRewardManager(AbstractRewardManager):
    def __init__(self, special_param: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.special_param = special_param

    def __call__(self, data: DataProto, return_dict: bool = False):
        # 实现自定义逻辑
        # 可以访问self.tokenizer, self.compute_score等

        # 示例：先检查rm_scores
        result = self._extract_reward_from_rm_scores(data, return_dict)
        if result is not None:
            return result

        # 实现自定义计算
        # ...

        return reward_tensor
```

**使用方式**:

```yaml
# config.yaml

reward_model:
  reward_manager:
    source: "register"
    name: "my_custom_manager"  # 使用自定义manager

  reward_kwargs:
    special_param: 2.0  # 传递给__init__的参数
```

---

## 9. 关键文件索引

### 9.1 核心文件列表

| 功能模块 | 文件路径 | 关键内容 |
|---------|---------|---------|
| **训练入口** | `verl/trainer/main_ppo.py` | 主训练脚本，L318加载reward manager |
| **训练循环** | `verl/trainer/ppo/ray_trainer.py` | PPO训练器，L1510-1528 reward计算 |
| **Reward加载** | `verl/trainer/ppo/reward.py` | L99-176加载，L189-205计算，L206-217异步 |
| **抽象基类** | `verl/workers/reward_manager/abstract.py` | AbstractRewardManager定义 |
| **Naive实现** | `verl/workers/reward_manager/naive.py` | 逐样本计算，L46-122 |
| **Batch实现** | `verl/workers/reward_manager/batch.py` | 批量计算，L25-128 |
| **DAPO实现** | `verl/workers/reward_manager/dapo.py` | 带长度惩罚，L25-149 |
| **Prime实现** | `verl/workers/reward_manager/prime.py` | 异步并行计算 |
| **Registry** | `verl/workers/reward_manager/registry.py` | 注册机制，L20-56 |
| **默认评分** | `verl/utils/reward_score/__init__.py` | default_compute_score，L19-114 |
| **GSM8K** | `verl/utils/reward_score/gsm8k.py` | GSM8K评分，L25-68 |
| **MATH** | `verl/utils/reward_score/math_reward.py` | MATH评分，L24-89 |
| **代码沙箱** | `verl/utils/reward_score/sandbox_fusion.py` | 代码执行评分，L25-156 |
| **RM Worker** | `verl/workers/fsdp_workers.py` | RewardModelWorker，L1654-2009 |
| **DataProto** | `verl/protocol.py` | 数据容器定义 |
| **配置类** | `verl/trainer/config/config.py` | L98-130 reward配置 |
| **配置文件** | `verl/trainer/config/reward_model/` | 各种预设配置 |

### 9.2 关键代码位置

#### A. 训练循环中的Reward计算

```python
# verl/trainer/ppo/ray_trainer.py

# Line 1510: Reward计算阶段开始
with marked_timer("reward", timing_raw, color="yellow"):

    # Line 1512-1517: Model-based reward
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        if not self.use_reward_loop:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
        else:
            reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

    # Line 1520-1528: Rule-based reward
    if self.config.reward_model.launch_reward_fn_async:
        future_reward = compute_reward_async.remote(data=batch, ...)
    else:
        reward_tensor, reward_extra_infos = self._compute_or_extract_reward(
            batch, reward_fn=self.reward_fn
        )
```

#### B. Advantage计算

```python
# verl/trainer/ppo/ray_trainer.py

# Line 1583-1600: 使用reward计算advantage
with marked_timer("adv", timing_raw, color="brown"):

    # Line 1584: 获取异步reward结果
    if self.config.reward_model.launch_reward_fn_async:
        reward_tensor, reward_extra_infos = ray.get(future_reward)

    # Line 1588: 设置token_level_scores
    batch.batch["token_level_scores"] = reward_tensor

    # Line 1591-1594: 应用KL惩罚
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl, ...)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    # Line 1597: 计算advantages
    batch = compute_advantage(batch, adv_estimator=..., gamma=..., lam=...)
```

#### C. NaiveRewardManager核心逻辑

```python
# verl/workers/reward_manager/naive.py

# Line 46-122: __call__方法
def __call__(self, data: DataProto, return_dict: bool = False):

    # Line 50-52: 检查rm_scores
    result = self._extract_reward_from_rm_scores(data, return_dict)
    if result is not None:
        return result

    # Line 56-87: 遍历计算
    all_reward = []
    for i in range(len(data)):
        data_item = data[i]

        # Line 60-65: 提取数据
        prompt_str = self.tokenizer.decode(data_item.batch["prompts"], ...)
        response_str = self.tokenizer.decode(data_item.batch["responses"], ...)
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]

        # Line 70: 调用评分函数
        score = self.compute_score(data_source, response_str, ground_truth, extra_info)

        # Line 75-81: 处理返回值
        if isinstance(score, dict):
            reward_val = score["score"]
        else:
            reward_val = float(score)
        all_reward.append(reward_val)

    # Line 90: 转换为tensor
    reward_tensor = self._to_tensor(data, all_reward)
    return reward_tensor
```

#### D. default_compute_score路由

```python
# verl/utils/reward_score/__init__.py

# Line 19-114: 根据data_source路由
def default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):

    # Line 30-32: GSM8K
    if data_source == "openai/gsm8k":
        return compute_score_for_gsm8k(solution_str, ground_truth)

    # Line 34-36: MATH
    elif data_source == "lighteval/MATH":
        return compute_score_for_math(solution_str, ground_truth)

    # Line 45-52: 代码评分
    elif data_source in ["codecontests", "apps", "taco"]:
        return compute_score_with_sandbox(
            data_source, solution_str, ground_truth,
            sandbox_url=kwargs["sandbox_fusion_url"],
            semaphore=kwargs["concurrent_semaphore"],
            memory_limit_mb=kwargs["memory_limit_mb"]
        )

    # Line 110-112: 未知数据源
    else:
        raise ValueError(f"Unsupported data_source: {data_source}")
```

#### E. RewardModelWorker计算

```python
# verl/workers/fsdp_workers.py

# Line 1956-2009: compute_rm_score方法
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
def compute_rm_score(self, data: DataProto) -> DataProto:

    # Line 1965-1967: 切换chat template
    if self.config.get("switch_chat_template", False):
        data = self._switch_chat_template(data)

    # Line 1970: 移动到GPU
    data = data.to(torch.cuda.current_device())

    # Line 1973-1978: 微批次处理
    micro_batch_size = self.config.get("rm_micro_batch_size", 4)
    all_scores = []
    for i in range(0, len(data), micro_batch_size):
        micro_batch = data[i:i + micro_batch_size]
        scores = self._forward_micro_batch(micro_batch)
        all_scores.append(scores)

    # Line 1981: 拼接结果
    all_scores = torch.cat(all_scores, dim=0)

    # Line 1984: 扩展为token级别
    token_level_scores = self._expand_to_token_level(data, all_scores)

    # Line 1987: 返回DataProto
    return DataProto.from_dict(tensors={"rm_scores": token_level_scores})
```

---

## 附录A: 快速参考

### A.1 常用命令

```bash
# 查看训练日志中的reward信息
cat logs/train.log | grep "reward"

# 运行GSM8K训练
python -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    reward_model=gsm8k

# 运行代码评分训练（需要先启动sandbox）
docker run -p 8000:8000 sandbox-fusion:latest
python -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/codecontests/train.parquet \
    reward_model=code

# 调试reward计算（打印1个样本）
python -m verl.trainer.main_ppo \
    reward_model.reward_kwargs.num_examine=1
```

### A.2 常见问题

**Q: Reward总是0怎么办？**

A: 检查以下几点：
1. `data_source`是否正确设置（在dataloader中）
2. `ground_truth`是否存在于`reward_model`字段
3. 评分函数的返回值格式是否正确
4. 查看日志中是否有异常信息

**Q: 如何添加自定义数据集的评分？**

A: 有两种方式：
1. 在`verl/utils/reward_score/__init__.py`的`default_compute_score`中添加新分支
2. 通过配置指定自定义reward函数（见8.4节）

**Q: Model-based reward和rule-based reward如何选择？**

A:
- Rule-based: 适用于有明确正确答案的任务（数学、代码）
- Model-based: 适用于主观评价任务（对话质量、创意写作）
- 组合使用: 可以同时启用，在reward manager中融合

**Q: 异步reward计算什么时候使用？**

A:
- 评分函数计算密集（如代码沙箱执行）
- 需要调用外部API（如GPT评分）
- 希望与其他操作并行以提升吞吐

### A.3 性能优化建议

1. **使用BatchRewardManager**: 如果评分可以向量化
2. **启用异步计算**: 对于耗时的评分函数
3. **调整micro_batch_size**: Model-based reward的批大小
4. **使用PrimeRewardManager**: 多进程并行评分
5. **缓存评分结果**: 对于重复的样本（需自行实现）

---

## 附录B: 扩展阅读

- **verl文档**: https://verl.readthedocs.io/
- **HybridFlow论文**: https://arxiv.org/abs/2409.19256
- **PPO算法**: Proximal Policy Optimization
- **GRPO算法**: Group Relative Policy Optimization
- **DataProto设计**: `verl/protocol.py`
- **Ray框架**: https://docs.ray.io/

---

**文档版本**: v1.0
**最后更新**: 2026-01-22
**维护者**: verl团队

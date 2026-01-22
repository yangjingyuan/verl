# RewardLoop代码走读

> 本文档详细梳理verl框架中新版Reward Loop系统的完整流程，包括架构设计、组件关系、调用链路和关键实现。

## 目录

- [1. 概览](#1-概览)
- [2. 整体架构](#2-整体架构)
- [3. RewardLoopManager](#3-rewardloopmanager)
- [4. RewardLoopWorker](#4-rewardloopworker)
- [5. 训练循环集成](#5-训练循环集成)
- [6. 奖励计算逻辑](#6-奖励计算逻辑)
- [7. 配置系统](#7-配置系统)
- [8. 关键文件索引](#8-关键文件索引)

---

## 1. 概览

### 1.1 Reward Loop的定位

Reward Loop是verl框架中**实验性**的新版奖励计算系统，位于`verl/experimental/reward_loop/`目录。它将取代旧版的`RewardModelWorker`，提供更灵活、高效的奖励计算能力。

**两套系统对比**：

| 特性 | 旧版 (RewardModelWorker) | 新版 (RewardLoop) |
|-----|------------------------|------------------|
| 位置 | `verl/workers/fsdp_workers.py` | `verl/experimental/reward_loop/` |
| 架构 | Worker Group模式 | Manager + Ray Workers |
| 并行方式 | 数据并行 (DP) | 异步并行 (asyncio) |
| 支持的Reward | 仅Model-based | Rule/DisRM/GenRM/自定义 |
| 灵活性 | 较低 | 高 |

### 1.2 核心设计理念

- **异步并行**: 使用`asyncio`实现样本级别的异步并发
- **统一接口**: 支持规则奖励、判别式RM、生成式RM等多种模式
- **资源隔离**: 可独立配置RM资源池，与Actor/Critic隔离
- **可扩展性**: 通过Registry机制注册自定义RewardManager

### 1.3 主要组件

```
RewardLoop系统组件:
├── RewardLoopManager (Controller端管理器)
│   ├── 管理多个RewardLoopWorker
│   ├── 协调RewardModelManager (可选)
│   └── 提供compute_rm_score接口
│
├── RewardLoopWorker (Ray远程Worker)
│   ├── 执行实际的奖励计算
│   ├── 支持异步批量处理
│   └── 可访问Reward Model服务
│
├── RewardModelManager (RM服务管理)
│   ├── 管理vLLM/SGLang推理服务
│   └── 提供HTTP接口供Worker调用
│
└── RewardManagerBase (奖励管理器基类)
    ├── NaiveRewardManager
    ├── DAPORewardManager
    ├── RateLimitedRewardManager
    └── RemoteRewardManager
```

---

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                   RayPPOTrainer (Controller)                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RewardLoopManager                       │   │
│  │  - 初始化和管理Workers                               │   │
│  │  - 数据分块和结果合并                               │   │
│  │  - 协调RewardModelManager                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│           ┌───────────────┼───────────────┐                │
│           ▼               ▼               ▼                │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│    │ Worker 0 │    │ Worker 1 │    │ Worker N │           │
│    │ (Ray)    │    │ (Ray)    │    │ (Ray)    │           │
│    │          │    │          │    │          │           │
│    │ asyncio  │    │ asyncio  │    │ asyncio  │           │
│    │ 并发处理 │    │ 并发处理 │    │ 并发处理 │           │
│    └──────────┘    └──────────┘    └──────────┘           │
│           │               │               │                │
│           └───────────────┼───────────────┘                │
│                           ▼                                 │
│                ┌─────────────────────┐                     │
│                │  RewardModelManager │ (可选)              │
│                │  - vLLM/SGLang服务  │                     │
│                │  - HTTP Router      │                     │
│                └─────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 调用时序图

```
RayPPOTrainer       RewardLoopManager    RewardLoopWorker[N]    RewardModelManager
     │                     │                     │                      │
     │──compute_rm_score──→│                     │                      │
     │                     │                     │                      │
     │                     │──wake_up()──────────────────────────────→ │
     │                     │                     │                      │
     │                     │──data.chunk(N)──→  │                      │
     │                     │                     │                      │
     │                     │──[并行] compute_score_batch.remote()──→  │
     │                     │                     │                      │
     │                     │                     │──[异步] compute_score()
     │                     │                     │        │
     │                     │                     │        ├─[自定义] reward_loop.run_single()
     │                     │                     │        │
     │                     │                     │        └─[RM模式] compute_score_disrm()
     │                     │                     │                 │
     │                     │                     │                 │──POST /classify
     │                     │                     │                 │      or
     │                     │                     │                 │──POST /v1/embeddings
     │                     │                     │                 │
     │                     │                     │←────────────────┘
     │                     │                     │
     │                     │←───list[dict]──────│
     │                     │                     │
     │                     │──合并结果→rm_scores │
     │                     │                     │
     │                     │──sleep()────────────────────────────────→│
     │                     │                     │                      │
     │←──DataProto(rm_scores)                   │                      │
     │                     │                     │                      │
```

### 2.3 与旧版系统的切换

**文件位置**: `verl/trainer/ppo/ray_trainer.py:856-883`

```python
if not self.use_reward_loop:
    # 旧版模式：使用RewardModelWorker
    if self.use_rm:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        rm_cls = RayClassWithInitArgs(
            self.role_worker_mapping[Role.RewardModel],
            config=self.config.reward_model
        )
        self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
else:
    # 新版 Reward Loop 模式
    can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool

    if not can_reward_loop_parallelize:
        # 同步模式：在Controller中创建RewardLoopManager
        from verl.experimental.reward_loop import RewardLoopManager

        self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )
```

**切换条件**:
- `use_reward_loop=True`: 启用新版Reward Loop
- `can_reward_loop_parallelize`: 判断是否可以与Actor并行
  - 条件1: 不使用RM（纯规则奖励）
  - 条件2: 使用RM但启用了独立资源池

---

## 3. RewardLoopManager

### 3.1 类定义

**文件位置**: `verl/experimental/reward_loop/reward_loop.py:227-305`

```python
class RewardLoopManager:
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    RewardLoopManager will deprecate fsdp/megatron RewardModelWorker in the future.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config

        # 1. 初始化RewardModelManager（如果启用RM）
        if self.config.reward_model.enable:
            self.reward_model_manager = RewardModelManager(
                config.reward_model,
                rm_resource_pool
            )
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        # 2. 初始化RewardLoopWorkers
        self._init_reward_loop_workers()
```

### 3.2 Worker初始化

**文件位置**: `reward_loop.py:245-261`

```python
def _init_reward_loop_workers(self):
    self.reward_loop_workers = []
    num_workers = self.config.reward_model.num_workers

    # 获取所有存活节点
    node_ids = [
        node["NodeID"]
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("CPU", 0) > 0
    ]

    for i in range(num_workers):
        # Round-robin方式分配到各节点
        node_id = node_ids[i % len(node_ids)]

        self.reward_loop_workers.append(
            RewardLoopWorker.options(
                name=f"reward_loop_worker_{i}",
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=True,  # 软亲和，允许调度到其他节点
                ),
            ).remote(self.config, self.reward_router_address)
        )
```

**关键设计**:
- 使用Round-robin将Workers均匀分布到各节点
- 使用NodeAffinitySchedulingStrategy控制调度
- `soft=True`允许在资源不足时调度到其他节点

### 3.3 核心方法: compute_rm_score

**文件位置**: `reward_loop.py:264-298`

```python
def compute_rm_score(self, data: DataProto) -> DataProto:
    """
    计算Reward Model分数

    此方法用于替代旧版RewardModelWorker.compute_rm_score

    Returns:
        DataProto: 包含rm_scores的数据容器
    """

    # 1. 唤醒Reward Model服务（如果有）
    if self.reward_model_manager is not None:
        self.reward_model_manager.wake_up()

    # 2. 将数据分块，分发给各Worker并行处理
    chunks = data.chunk(len(self.reward_loop_workers))

    # 3. 并行调用所有Workers
    outputs = ray.get([
        worker.compute_score_batch.remote(chunk)
        for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
    ])

    # 4. 展平结果列表
    outputs_flat = [item for sublist in outputs for item in sublist]

    # 5. 提取分数并构建rm_scores tensor
    scores = [item["reward_score"] for item in outputs_flat]

    prompt_length = data.batch["prompts"].size(1)
    valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)

    # 创建token级别的rm_scores（只在最后一个token处赋值）
    rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    rm_scores[
        torch.arange(rm_scores.size(0)),
        valid_response_length - 1
    ] = torch.tensor(scores, dtype=torch.float32)

    batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

    # 6. 处理额外信息
    reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
    reward_extra_keys = list(reward_extra_infos[0].keys())

    non_tensor_batch = {}
    for key in reward_extra_keys:
        non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

    # 7. 休眠Reward Model服务
    if self.reward_model_manager is not None:
        self.reward_model_manager.sleep()

    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={"reward_extra_keys": reward_extra_keys}
    )
```

**关键流程**:
1. **Wake up**: 唤醒RM推理服务（节省空闲时资源）
2. **Chunk**: 按Worker数量分块数据
3. **Parallel**: Ray并行调用各Worker
4. **Flatten**: 合并所有结果
5. **To Tensor**: 将标量分数转为token级别tensor
6. **Sleep**: 休眠RM服务

---

## 4. RewardLoopWorker

### 4.1 类定义

**文件位置**: `verl/experimental/reward_loop/reward_loop.py:40-225`

```python
@ray.remote
class RewardLoopWorker:
    """
    Ray远程Actor，执行实际的奖励计算

    支持三种奖励计算方式:
    (1) rule-based reward computation - 基于规则的奖励
    (2) reward model-based reward computation (both disrm and genrm) - 基于模型的奖励
    (3) high-flexible user-customized reward function - 用户自定义奖励函数

    计算逻辑:
    - 如果提供了自定义reward函数 → 直接使用
    - 如果没有自定义函数:
        - RM未启用 → 使用默认规则奖励
        - RM是DisRM → 计算判别式RM分数
        - RM是GenRM → 必须提供自定义函数（否则报错）
    """

    def __init__(self, config: DictConfig, reward_router_address: str = None):
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()
```

### 4.2 初始化流程

**文件位置**: `reward_loop.py:64-102`

```python
def _init_reward_fn(self):
    # 1. 加载输入Tokenizer（Actor模型的tokenizer）
    input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
    self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)

    # 2. 加载RM Tokenizer（如果启用RM）
    self.reward_model_tokenizer = None
    if self.config.reward_model.enable:
        reward_model_tokenizer_local_path = copy_to_local(self.config.reward_model.model.path)
        self.reward_model_tokenizer = hf_tokenizer(
            reward_model_tokenizer_local_path,
            trust_remote_code=True
        )

    # 3. 加载自定义Reward函数（如果配置了）
    self.reward_fn = get_custom_reward_fn(self.config)

    # 4. 加载RewardManager类
    reward_loop_source = self.config.reward_model.get("reward_loop_source", "register")

    if reward_loop_source == "register":
        # 从Registry加载（默认方式）
        reward_manager_cls = get_reward_manager_cls(self.config.reward_model.reward_manager)
    elif reward_loop_source == "importlib":
        # 从外部模块加载
        from verl.utils.import_utils import load_extern_object

        reward_loop_module_path = self.config.reward_model.get("reward_loop_module_path")
        reward_loop_class_name = self.config.reward_model.get("reward_loop_class_name")

        reward_manager_cls = load_extern_object(
            module_path=reward_loop_module_path,
            object_name=reward_loop_class_name
        )
    else:
        raise ValueError(f"Unknown reward_loop_source: {reward_loop_source}")

    # 5. 实例化RewardManager
    self.reward_loop = reward_manager_cls(
        self.config,
        self.input_tokenizer,
        self.reward_fn,
        self.reward_router_address,
        self.reward_model_tokenizer
    )
```

### 4.3 批量计算接口

**文件位置**: `reward_loop.py:104-109`

```python
async def compute_score_batch(self, data: DataProto) -> list[dict]:
    """
    批量计算奖励分数

    对batch中的每个样本创建异步任务并发执行

    Args:
        data: 包含多个样本的DataProto

    Returns:
        list[dict]: 每个样本的奖励结果
    """
    tasks = []
    for i in range(len(data)):
        # 为每个样本创建异步任务
        tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))

    # 并发执行所有任务
    outputs = await asyncio.gather(*tasks)
    return outputs
```

**关键设计**:
- 使用`asyncio.gather`实现样本级别的并发
- 每个样本独立计算，互不阻塞
- 适合IO密集型的奖励计算（如调用外部API）

### 4.4 单样本计算逻辑

**文件位置**: `reward_loop.py:111-122`

```python
async def compute_score(self, data: DataProto) -> dict:
    """
    计算单个样本的奖励分数

    Args:
        data: 包含单个样本的DataProto

    Returns:
        dict: 包含"reward_score"键的字典
    """
    assert len(data) == 1, "RewardLoopWorker only support single data item"

    if self.config.custom_reward_function.path is not None:
        # 使用自定义reward函数
        return await self.reward_loop.run_single(data)
    else:
        if self.config.reward_model.enable:
            # 使用判别式Reward Model
            return await self.compute_score_disrm(data)
        else:
            # 使用基于规则的奖励
            return await self.reward_loop.run_single(data)
```

**计算路径**:
```
compute_score(data)
    │
    ├─[自定义函数存在]→ reward_loop.run_single(data)
    │
    └─[自定义函数不存在]
        │
        ├─[RM启用]→ compute_score_disrm(data) → POST到RM服务
        │
        └─[RM未启用]→ reward_loop.run_single(data) → 规则奖励
```

### 4.5 判别式RM计算

**文件位置**: `reward_loop.py:200-224`

```python
async def compute_score_disrm(self, data: DataProto) -> dict:
    """
    使用判别式Reward Model计算分数

    流程:
    1. 预处理输入（应用chat template）
    2. 发送HTTP请求到RM服务
    3. 解析返回的分数
    """
    # 1. 预处理
    disrm_prompt = await self._preprocess_reward_inputs(data)

    engine_name = self.config.reward_model.rollout.name
    model_name = self.config.reward_model.model.path

    if engine_name == "vllm":
        # vLLM引擎
        payloads = {
            "model": model_name,
            "input": disrm_prompt,
            "activation": False,
        }
        output = await self._post_request(payloads, "classify")
        rm_score = output["data"][-1]["probs"][-1]

    elif engine_name == "sglang":
        # SGLang引擎
        payloads = {
            "model": model_name,
            "input": disrm_prompt,
        }
        output = await self._post_request(payloads, "v1/embeddings")
        rm_score = output["data"][-1]["embedding"][-1]
    else:
        raise NotImplementedError(f"RewardLoopManager does not support {engine_name}")

    return {"reward_score": rm_score}
```

### 4.6 输入预处理

**文件位置**: `reward_loop.py:164-198`

```python
async def _preprocess_reward_inputs(self, data: DataProto) -> str:
    """
    预处理输入数据，构造RM所需的prompt格式

    流程:
    1. 提取原始prompt（对话历史）
    2. 解码response tokens
    3. 应用RM的chat template
    """
    assert len(data) == 1, "RewardLoopWorker only support single data item"
    data_item = data[0]
    assert "raw_prompt" in data_item.non_tensor_batch

    # 1. 提取原始prompt（对话历史）
    chat: list = list(data_item.non_tensor_batch["raw_prompt"])

    # 2. 提取并解码response
    response_ids = data_item.batch["responses"]
    response_length = response_ids.shape[-1]
    valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    # 解码
    rollout_response = self.input_tokenizer.decode(valid_response_ids)
    # 移除特殊token
    rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")

    # 3. 添加assistant回复
    chat.append({"role": "assistant", "content": rollout_response})

    # 4. 应用RM的chat template
    rm_prompt = self.reward_model_tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=False,
        tokenize=False,
    )

    # 5. 移除BOS token（某些tokenizer会自动添加）
    if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(
        self.reward_model_tokenizer.bos_token
    ):
        rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token):]

    return rm_prompt
```

### 4.7 HTTP请求处理

**文件位置**: `reward_loop.py:124-162`

```python
async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
    """
    发送HTTP POST请求到Reward Model服务

    特性:
    - 指数退避重试
    - 区分4xx和5xx错误
    - 无限超时（适合长时间推理）
    """
    url = f"http://{self.reward_router_address}/{endpoint}"
    last_exception = None

    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=None)  # 无超时限制
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()

        except aiohttp.ClientResponseError as e:
            # 4xx错误不重试（客户端问题）
            if 400 <= e.status < 500:
                logger.error(f"Request to {url} failed with client error HTTP {e.status}")
                raise
            # 5xx错误重试（服务端问题）
            last_exception = e
            logger.warning(f"[Attempt {attempt + 1}/{max_retries}] HTTP {e.status}. Retrying...")

        except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
            last_exception = e
            logger.warning(f"[Attempt {attempt + 1}/{max_retries}] {e}. Retrying...")

        if attempt < max_retries - 1:
            # 指数退避，最大30秒
            backoff_seconds = 2 ** attempt
            await asyncio.sleep(min(backoff_seconds, 30))

    logger.error(f"Max retries ({max_retries}) reached for {url}")
    if last_exception:
        raise last_exception
```

---

## 5. 训练循环集成

### 5.1 训练循环中的调用点

**文件位置**: `verl/trainer/ppo/ray_trainer.py:1510-1528`

```python
with marked_timer("reward", timing_raw, color="yellow"):
    # Step 1: 计算Reward Model分数（如果启用RM且batch中还没有）
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        if not self.use_reward_loop:
            # 旧版：使用RewardModelWorker
            reward_tensor = self.rm_wg.compute_rm_score(batch)
        else:
            # 新版：使用RewardLoopManager
            assert self.reward_loop_manager is not None, "RewardLoopManager is None"
            reward_tensor = self.reward_loop_manager.compute_rm_score(batch)

        batch = batch.union(reward_tensor)

    # Step 2: 计算最终奖励（规则奖励 + 可能的RM分数融合）
    if self.config.reward_model.launch_reward_fn_async:
        # 异步模式
        future_reward = compute_reward_async.remote(
            data=batch,
            config=self.config,
            tokenizer=self.tokenizer
        )
    else:
        # 同步模式
        reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
            batch, reward_fn=self.reward_fn, reward_for_val=False
        )
```

### 5.2 完整调用链路

```
Training Loop (ray_trainer.py:fit)
    │
    ├──[1] rollout_wg.generate_sequences(prompts)  # 生成响应
    │
    ├──[2] reward计算阶段
    │       │
    │       ├── [2.1] reward_loop_manager.compute_rm_score(batch)
    │       │           │
    │       │           ├── reward_model_manager.wake_up()
    │       │           │
    │       │           ├── data.chunk(num_workers)
    │       │           │
    │       │           ├── [并行] workers[i].compute_score_batch.remote(chunk)
    │       │           │              │
    │       │           │              ├── [异步] compute_score(data_item)
    │       │           │              │       │
    │       │           │              │       ├── [自定义] reward_loop.run_single()
    │       │           │              │       │
    │       │           │              │       └── [RM模式] compute_score_disrm()
    │       │           │              │                 │
    │       │           │              │                 └── POST /classify
    │       │           │              │
    │       │           │              └── return list[dict]
    │       │           │
    │       │           ├── 合并结果 → rm_scores tensor
    │       │           │
    │       │           └── reward_model_manager.sleep()
    │       │
    │       └── [2.2] self.reward_fn(batch)  # 规则奖励计算
    │
    ├──[3] compute_advantage()  # 计算优势函数
    │
    └──[4] actor_wg.update_policy()  # 更新策略
```

### 5.3 数据流转

```
输入 DataProto:
├── batch (TensorDict)
│   ├── prompts: (batch_size, prompt_len)
│   ├── responses: (batch_size, response_len)
│   ├── input_ids: (batch_size, seq_len)
│   └── attention_mask: (batch_size, seq_len)
│
└── non_tensor_batch (dict)
    ├── raw_prompt: list[list[dict]]  # 对话历史
    ├── data_source: list[str]        # 数据集标识
    └── reward_model: list[dict]      # ground_truth等
            ↓
    RewardLoopManager.compute_rm_score()
            ↓
输出 DataProto:
├── batch (TensorDict)
│   └── rm_scores: (batch_size, response_len)  # token级别分数
│
└── non_tensor_batch (dict)
    └── [reward_extra_keys]: 额外信息
```

---

## 6. 奖励计算逻辑

### 6.1 RewardManager体系

**Experimental RewardManager**位于`verl/experimental/reward_loop/reward_manager/`：

```
reward_manager/
├── __init__.py         # 导出和Registry
├── base.py             # RewardManagerBase基类
├── registry.py         # 注册机制
├── naive.py            # NaiveRewardManager
├── dapo.py             # DAPORewardManager
├── limited.py          # RateLimitedRewardManager
└── remote.py           # RemoteRewardManager
```

### 6.2 基类定义

**文件位置**: `verl/experimental/reward_loop/reward_manager/base.py`

```python
class RewardManagerBase(ABC):
    """
    Experimental RewardManager基类

    与旧版AbstractRewardManager的区别:
    - 使用异步接口 (async def)
    - 简化的构造函数签名
    - 不接收num_examine参数
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        compute_score: Callable,
        reward_router_address: str = None,
        reward_model_tokenizer = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    @abstractmethod
    async def run_single(self, data: DataProto) -> dict:
        """
        计算单个样本的奖励

        Args:
            data: 包含单个样本的DataProto

        Returns:
            dict: 必须包含"reward_score"键
        """
        pass
```

### 6.3 NaiveRewardManager实现

**文件位置**: `verl/experimental/reward_loop/reward_manager/naive.py`

```python
@register("naive")
class NaiveRewardManager(RewardManagerBase):
    """
    简单的规则奖励管理器

    直接调用compute_score函数计算奖励
    """

    async def run_single(self, data: DataProto) -> dict:
        """计算单个样本的规则奖励"""
        assert len(data) == 1
        data_item = data[0]

        # 1. 提取数据
        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]

        # 2. 解码
        prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 3. 获取ground_truth和data_source
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch.get("data_source", "default")
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # 4. 调用评分函数
        score = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info
        )

        # 5. 处理返回值
        if isinstance(score, dict):
            reward_score = score.get("score", score.get("reward_score", 0.0))
            reward_extra_info = {k: v for k, v in score.items() if k not in ["score", "reward_score"]}
        else:
            reward_score = float(score)
            reward_extra_info = {}

        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info
        }
```

### 6.4 注册机制

**文件位置**: `verl/experimental/reward_loop/reward_manager/registry.py`

```python
REWARD_LOOP_MANAGER_REGISTRY: dict[str, type[RewardManagerBase]] = {}

def register(name: str):
    """
    装饰器：注册RewardManager到Registry

    使用示例:
        @register("my_manager")
        class MyRewardManager(RewardManagerBase):
            ...
    """
    def decorator(cls):
        if name in REWARD_LOOP_MANAGER_REGISTRY:
            raise ValueError(f"Reward manager '{name}' already registered")
        REWARD_LOOP_MANAGER_REGISTRY[name] = cls
        return cls
    return decorator

def get_reward_manager_cls(name: str) -> type[RewardManagerBase]:
    """从Registry获取RewardManager类"""
    if name not in REWARD_LOOP_MANAGER_REGISTRY:
        available = ", ".join(REWARD_LOOP_MANAGER_REGISTRY.keys())
        raise ValueError(f"Unknown reward manager: '{name}'. Available: {available}")
    return REWARD_LOOP_MANAGER_REGISTRY[name]
```

**已注册的Managers**:
- `"naive"` → NaiveRewardManager
- `"dapo"` → DAPORewardManager
- `"limited"` → RateLimitedRewardManager
- `"remote"` → RemoteRewardManager

---

## 7. 配置系统

### 7.1 关键配置项

```yaml
reward_model:
  # ==== 基础配置 ====
  enable: true                    # 是否启用Reward Model
  use_reward_loop: true           # 是否使用新版Reward Loop

  # ==== Worker配置 ====
  num_workers: 4                  # RewardLoopWorker数量

  # ==== RewardManager配置 ====
  reward_manager: naive           # Manager类型: naive/dapo/limited/remote
  reward_loop_source: register    # 加载方式: register/importlib

  # 当reward_loop_source=importlib时使用
  reward_loop_module_path: "my_module.reward"
  reward_loop_class_name: "MyRewardManager"

  # ==== 资源池配置 ====
  enable_resource_pool: false     # 是否为RM使用独立资源池
  n_gpus_per_node: 4              # 每节点GPU数
  nnodes: 1                       # 节点数

  # ==== 模型配置 ====
  model:
    path: "path/to/reward/model"

  rollout:
    name: vllm                    # 推理引擎: vllm/sglang

# ==== 自定义Reward函数 ====
custom_reward_function:
  path: "path/to/reward.py"
  name: "my_reward_fn"
  reward_kwargs:
    param1: value1
```

### 7.2 配置文件示例

**DP Reward Loop配置**: `verl/trainer/config/reward_model/dp_reward_loop.yaml`

```yaml
reward_model:
  enable: true
  use_reward_loop: true

  # Worker配置
  num_workers: 8
  reward_manager: naive

  # RM模型
  model:
    path: "OpenAssistant/reward-model-deberta-v3-large-v2"

  rollout:
    name: vllm

  # 资源配置
  enable_resource_pool: false
  n_gpus_per_node: 4
```

**Megatron Reward Loop配置**: `verl/trainer/config/reward_model/megatron_reward_loop.yaml`

```yaml
reward_model:
  enable: true
  use_reward_loop: true

  # Worker配置
  num_workers: 16
  reward_manager: naive

  # 使用独立资源池
  enable_resource_pool: true
  n_gpus_per_node: 8
  nnodes: 2
```

### 7.3 使用方式

```bash
# 使用Reward Loop的PPO训练
python -m verl.trainer.main_ppo \
    reward_model.use_reward_loop=true \
    reward_model.num_workers=8 \
    reward_model.reward_manager=naive \
    reward_model.enable=true \
    reward_model.model.path=OpenAssistant/reward-model-deberta-v3-large-v2

# 纯规则奖励（不使用RM）
python -m verl.trainer.main_ppo \
    reward_model.use_reward_loop=true \
    reward_model.enable=false \
    reward_model.num_workers=4 \
    reward_model.reward_manager=naive
```

---

## 8. 关键文件索引

### 8.1 核心文件列表

| 功能模块 | 文件路径 | 关键内容 |
|---------|---------|---------|
| **RewardLoopManager** | `verl/experimental/reward_loop/reward_loop.py:227-305` | Manager类，管理Workers |
| **RewardLoopWorker** | `verl/experimental/reward_loop/reward_loop.py:40-225` | Ray Worker，执行计算 |
| **RewardModelManager** | `verl/experimental/reward_loop/reward_model.py` | RM服务管理 |
| **RewardManager基类** | `verl/experimental/reward_loop/reward_manager/base.py` | 异步基类定义 |
| **NaiveRewardManager** | `verl/experimental/reward_loop/reward_manager/naive.py` | 规则奖励实现 |
| **Registry** | `verl/experimental/reward_loop/reward_manager/registry.py` | 注册机制 |
| **训练循环集成** | `verl/trainer/ppo/ray_trainer.py:1510-1528` | reward计算调用点 |
| **初始化集成** | `verl/trainer/ppo/ray_trainer.py:856-883` | Manager创建 |
| **Reward加载** | `verl/trainer/ppo/reward.py:60-96` | get_custom_reward_fn |
| **配置文件** | `verl/trainer/config/reward_model/` | 各种预设配置 |

### 8.2 关键代码位置

#### A. RewardLoopManager初始化

```python
# verl/trainer/ppo/ray_trainer.py:876-883

if not can_reward_loop_parallelize:
    from verl.experimental.reward_loop import RewardLoopManager

    self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
    self.reward_loop_manager = RewardLoopManager(
        config=self.config,
        rm_resource_pool=resource_pool,
    )
```

#### B. 训练循环中的调用

```python
# verl/trainer/ppo/ray_trainer.py:1512-1518

if self.use_rm and "rm_scores" not in batch.batch.keys():
    if not self.use_reward_loop:
        reward_tensor = self.rm_wg.compute_rm_score(batch)
    else:
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
    batch = batch.union(reward_tensor)
```

#### C. Worker并行计算

```python
# verl/experimental/reward_loop/reward_loop.py:268-274

chunks = data.chunk(len(self.reward_loop_workers))
outputs = ray.get([
    worker.compute_score_batch.remote(chunk)
    for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
])
```

#### D. 异步样本计算

```python
# verl/experimental/reward_loop/reward_loop.py:104-109

async def compute_score_batch(self, data: DataProto) -> list[dict]:
    tasks = []
    for i in range(len(data)):
        tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
    outputs = await asyncio.gather(*tasks)
    return outputs
```

---

## 附录A: 与旧版系统的对比

### A.1 架构对比

| 特性 | 旧版 RewardModelWorker | 新版 RewardLoop |
|-----|----------------------|-----------------|
| 位置 | `verl/workers/fsdp_workers.py` | `verl/experimental/reward_loop/` |
| 类型 | WorkerGroup成员 | 独立的Manager+Workers |
| 并行模式 | 数据并行(DP) | Ray远程 + asyncio |
| 调度 | 与Actor/Critic共享资源池 | 可独立资源池 |
| 模型加载 | Worker内部加载 | RewardModelManager管理 |
| 通信方式 | Ray ObjectRef | HTTP API |

### A.2 接口对比

**旧版**:
```python
# Worker Group模式
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
def compute_rm_score(self, data: DataProto) -> DataProto:
    # 在Worker进程内部执行
    ...
```

**新版**:
```python
# Manager模式
def compute_rm_score(self, data: DataProto) -> DataProto:
    # Manager分发任务给Ray Workers
    outputs = ray.get([
        worker.compute_score_batch.remote(chunk)
        for worker, chunk in zip(self.workers, chunks)
    ])
    ...
```

### A.3 迁移指南

从旧版迁移到新版只需修改配置:

```yaml
# 旧版配置
reward_model:
  enable: true
  # use_reward_loop默认为false

# 新版配置
reward_model:
  enable: true
  use_reward_loop: true
  num_workers: 8
```

---

## 附录B: 常见问题

**Q1: Reward Loop和RewardModelWorker可以同时使用吗？**

A: 不可以。通过`use_reward_loop`配置项二选一：
- `use_reward_loop=false`: 使用旧版RewardModelWorker
- `use_reward_loop=true`: 使用新版RewardLoop

**Q2: 为什么Worker使用asyncio而不是多线程？**

A: asyncio更适合IO密集型场景（如HTTP请求）：
- 避免线程切换开销
- 更高的并发数
- 更好的资源利用率

**Q3: RewardModelManager和RewardLoopManager的关系？**

A:
- RewardModelManager: 管理RM推理服务（vLLM/SGLang）
- RewardLoopManager: 管理RewardLoopWorkers
- 关系: RewardLoopManager可选地包含RewardModelManager

**Q4: 如何调试Reward Loop？**

A:
```python
# 1. 设置日志级别
import os
os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"

# 2. 查看Ray Dashboard
ray start --head --dashboard-host=0.0.0.0
# 访问 http://localhost:8265

# 3. 检查Worker状态
ray.get([w.ping.remote() for w in manager.reward_loop_workers])
```

---

**文档版本**: v1.0
**最后更新**: 2026-01-22
**维护者**: verl团队

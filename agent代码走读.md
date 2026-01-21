# VERL Agent Loop 代码走读

本文档详细讲解 VERL 中 multi-turn agent loop 的完整实现，包括初始化流程、状态机设计、工具调用和环境交互机制。

---

## 目录

- [一、整体架构](#一整体架构)
- [二、核心组件详解](#二核心组件详解)
  - [2.1 AgentLoopManager](#21-agentloopmanager)
  - [2.2 AsyncLLMServerManager](#22-asyncllmservermanager)
  - [2.3 AgentLoopWorker](#23-agentloopworker)
  - [2.4 Agent Loop 状态机](#24-agent-loop-状态机)
- [三、初始化流程详解](#三初始化流程详解)
  - [3.1 初始化LLM Servers](#31-初始化llm-servers)
  - [3.2 初始化Agent Loop Workers](#32-初始化agent-loop-workers)
- [四、状态转换详解](#四状态转换详解)
  - [4.1 PENDING → GENERATING](#41-pending--generating)
  - [4.2 GENERATING → PROCESSING_TOOLS/INTERACTING/TERMINATED](#42-generating--processing_toolsinteractingterminated)
  - [4.3 PROCESSING_TOOLS → GENERATING](#43-processing_tools--generating)
  - [4.4 INTERACTING → GENERATING/TERMINATED](#44-interacting--generatingterminated)
- [五、工具调用机制](#五工具调用机制)
- [六、环境交互机制](#六环境交互机制)
- [七、完整执行流程示例](#七完整执行流程示例)
- [八、关键设计特点](#八关键设计特点)

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                   AgentLoopManager                       │
│  - 管理多个AgentLoopWorker                               │
│  - 管理LLM Server Replicas (vLLM/SGLang)               │
│  - 负载均衡和任务分发                                     │
└────────────┬────────────────────────────────────────────┘
             │
             ├──────────┬──────────┬──────────┐
             ▼          ▼          ▼          ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │AgentLoop     │ │AgentLoop     │ │AgentLoop     │
    │Worker 0      │ │Worker 1      │ │Worker N      │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           └────────────────┴────────────────┘
                         │
                 异步并发处理batch
                         │
           ┌─────────────┴─────────────┐
           │                           │
      ┌────▼────┐              ┌──────▼──────┐
      │ Single  │              │    Tool     │
      │ Turn    │              │   Agent     │
      │ Agent   │              │   Loop      │
      └─────────┘              └─────────────┘
```

**核心文件位置**：
- `verl/experimental/agent_loop/agent_loop.py` - 管理类和基础组件
- `verl/experimental/agent_loop/tool_agent_loop.py` - 工具调用Agent
- `verl/experimental/agent_loop/single_turn_agent_loop.py` - 单轮Agent
- `verl/workers/rollout/replica.py` - LLM Server Replica抽象

---

## 二、核心组件详解

### 2.1 AgentLoopManager

**位置**: `verl/experimental/agent_loop/agent_loop.py:837-1031`

**职责**:
- 管理整个agent loop系统的生命周期
- 初始化LLM server replicas (vLLM/SGLang/TensorRT-LLM)
- 管理AgentLoopWorker池
- 实现任务分发和结果聚合

**关键方法**:

```python
class AgentLoopManager:
    def __init__(
        config: DictConfig,
        worker_group: RayWorkerGroup = None,           # hybrid模式的worker group
        rollout_resource_pool: RayResourcePool = None, # colocated/standalone资源池
        rm_resource_pool: RayResourcePool = None       # reward model资源池
    ):
        # 1. 初始化reward model manager (可选)
        if config.reward_model.enable:
            self.reward_model_manager = RewardModelManager(...)

        # 2. 确定rollout replica类型
        self.rollout_replica_class = get_rollout_replica_class(config.rollout.name)

        # 3. 初始化LLM servers
        self._initialize_llm_servers(rollout_resource_pool)

        # 4. 初始化agent loop workers
        self._init_agent_loop_workers()

        # 5. 初始sleep (如果启用free_cache_engine)
        if config.rollout.free_cache_engine:
            self.sleep()

    def generate_sequences(prompts: DataProto) -> DataProto:
        """主入口: 处理一个batch的生成请求"""
        # 1. wake_up所有replica (同步weights)
        self.wake_up()

        # 2. 将batch切分成chunks
        chunks = prompts.chunk(len(self.agent_loop_workers))

        # 3. 并发调用所有workers
        outputs = ray.get([
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ])

        # 4. 合并结果
        output = DataProto.concat(outputs)

        # 5. sleep所有replica
        self.sleep()

        return output
```

### 2.2 AsyncLLMServerManager

**位置**: `verl/experimental/agent_loop/agent_loop.py:57-123`

**职责**:
- 管理多个OpenAI兼容的LLM server handles
- 实现负载均衡 (least requests)
- 实现sticky session (多轮对话路由到同一server以利用prefix caching)

**核心机制**:

```python
class AsyncLLMServerManager:
    def __init__(config, server_handles, max_cache_size=10000):
        self.server_handles = server_handles

        # 小顶堆: [请求数, idx, server_handle]
        self.weighted_serveres = [[0, idx, server]
                                   for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU缓存: request_id -> server_handle
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(request_id: str) -> ray.actor.ActorHandle:
        """选择server: 优先sticky session, 否则least requests"""
        # Sticky session: 同一request_id路由到之前的server
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        # Load balancing: 选择请求数最少的server
        _, _, server = self.weighted_serveres[0]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])

        # 缓存映射
        self.request_id_to_server[request_id] = server
        return server

    async def generate(request_id, *, prompt_ids, sampling_params,
                       image_data=None, video_data=None) -> TokenOutput:
        """生成tokens"""
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=uuid4().hex,  # 每turn使用新的request_id
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data
        )
        return output
```

**关键设计**:
- **Sticky Session**: 同一trajectory的多轮对话路由到同一server，最大化prefix caching效果
- **Load Balancing**: 新trajectory选择负载最低的server
- **LRU Cache**: 限制缓存大小，自动淘汰旧的映射

### 2.3 AgentLoopWorker

**位置**: `verl/experimental/agent_loop/agent_loop.py:345-813`

**职责**:
- 处理一个batch chunk的prompts
- 为每个sample创建对应的AgentLoop实例
- 异步并发执行所有samples
- 后处理: padding, masking, position_ids计算

**核心流程**:

```python
class AgentLoopWorker:
    def __init__(config, server_handles, reward_router_address):
        # 1. 创建AsyncLLMServerManager
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        # 2. 初始化tokenizer和processor
        self.tokenizer = hf_tokenizer(model_path)
        self.processor = hf_processor(model_path)

        # 3. 加载agent loop registry
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for cfg in agent_loop_configs:
                _agent_loop_registry[cfg.name] = cfg

        # 4. 初始化reward loop worker (可选)
        if use_reward_loop:
            self.reward_loop_worker = RewardLoopWorker.remote(...)

    async def generate_sequences(batch: DataProto) -> DataProto:
        """处理一个batch chunk"""
        # 1. 准备sampling params
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            logprobs=config.calculate_log_probs
        )

        # 2. 确定每个sample的agent类型
        if "agent_name" not in batch.non_tensor_batch:
            agent_name = default_agent_loop

        # 3. 为每个sample创建异步任务
        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(
                        sampling_params,
                        trajectory_info[i],
                        agent_name=kwargs['agent_name'],
                        **kwargs
                    )
                )
            )

        # 4. 并发执行所有agent loop
        outputs = await asyncio.gather(*tasks)

        # 5. 后处理
        output = self._postprocess(outputs)

        return output

    async def _run_agent_loop(sampling_params, trajectory, *, agent_name, **kwargs):
        """运行单个agent loop"""
        # 1. 实例化agent loop
        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            ...
        )

        # 2. 运行agent loop
        output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

        # 3. 后处理
        return await self._agent_loop_postprocess(output, **kwargs)
```

**后处理机制** (`_agent_loop_postprocess`):

```python
# Padding策略:
# - prompt_ids: 左padding (用0填充)
#   例: [0,0,0,0,1,2,3,4]
#
# - response_ids: 右padding (用0填充)
#   例: [5,6,7,8,0,0,0,0]
#
# - attention_mask: 0表示padding, 1表示有效token
#   例: [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,0,0,0(response)]
#
# - response_mask: 1表示LLM生成的token, 0表示tool response/padding
#   例: [1,1,1,(tool start),0,0(tool end),1,1,0,0,0]
#
# - position_ids: 顺序递增的位置编码
#   例: [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,0,0,0]

async def _agent_loop_postprocess(output, **kwargs):
    # 1. Padding prompt (左padding)
    tokenizer.padding_side = "left"
    prompt_output = tokenizer.pad(
        {"input_ids": output.prompt_ids},
        padding="max_length",
        max_length=prompt_length,
        return_attention_mask=True
    )

    # 2. Padding response (右padding)
    tokenizer.padding_side = "right"
    response_output = tokenizer.pad(
        {"input_ids": output.response_ids},
        padding="max_length",
        max_length=response_length,
        return_attention_mask=True
    )

    # 3. Padding response_mask
    response_mask_output = tokenizer.pad(
        {"input_ids": output.response_mask},
        padding="max_length",
        max_length=response_length
    )

    # 4. 计算最终mask
    response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
    attention_mask = torch.cat([prompt_output["attention_mask"],
                                response_output["attention_mask"]], dim=1)
    input_ids = torch.cat([prompt_output["input_ids"],
                           response_output["input_ids"]], dim=1)

    # 5. 计算position_ids (考虑multi-modal)
    position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)

    # 6. 可选: 计算reward score
    await self._compute_score(output, prompts, responses, ...)

    return _InternalAgentLoopOutput(
        prompt_ids=prompt_output["input_ids"],
        response_ids=response_output["input_ids"],
        input_ids=input_ids,
        position_ids=position_ids,
        response_mask=response_mask,
        attention_mask=attention_mask,
        ...
    )
```

### 2.4 Agent Loop 状态机

**位置**: `verl/experimental/agent_loop/tool_agent_loop.py`

**状态定义**:

```python
class AgentState(Enum):
    PENDING = "pending"             # 准备初始prompt
    GENERATING = "generating"        # LLM生成中
    PROCESSING_TOOLS = "processing_tools"  # 执行工具调用
    INTERACTING = "interacting"     # 与环境交互
    TERMINATED = "terminated"       # 终止
```

**AgentData状态容器**:

```python
class AgentData:
    """封装agent loop的所有状态变量"""
    # 对话历史
    messages: list[dict]            # chat messages
    image_data: list[Image]         # 图像数据
    video_data: list[tuple]         # 视频数据

    # Token序列 (累积的)
    prompt_ids: list[int]           # 整个prompt token序列
    response_ids: list[int]         # 当前turn的response
    response_mask: list[int]        # 1=LLM生成, 0=tool/env response
    response_logprobs: list[float]  # log probabilities

    # 计数器
    user_turns: int                 # 用户轮次
    assistant_turns: int            # assistant轮次

    # 工具相关
    tool_calls: list[FunctionCall]  # 待执行的工具调用

    # 环境交互
    interaction: BaseInteraction    # 交互环境实例
    interaction_kwargs: dict        # 交互配置

    # 奖励和指标
    turn_scores: list[float]        # 每轮得分(来自interaction)
    tool_rewards: list[float]       # 工具调用奖励
    metrics: dict                   # 性能指标

    # 扩展字段
    extra_fields: dict              # 动态添加的字段
```

**主循环**:

```python
@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    async def run(sampling_params, **kwargs) -> AgentLoopOutput:
        # 1. 初始化AgentData
        agent_data = AgentData(
            messages=kwargs["raw_prompt"],
            image_data=images,
            video_data=videos,
            request_id=uuid4().hex,
            interaction=interaction,
            ...
        )

        # 2. 状态机循环
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)

        # 3. 构造最终输出
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=agent_data.response_mask,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            extra_fields={"turn_scores": agent_data.turn_scores,
                         "tool_rewards": agent_data.tool_rewards}
        )
```

---

## 三、初始化流程详解

### 3.1 初始化LLM Servers

**方法**: `AgentLoopManager._initialize_llm_servers()` (lines 878-924)

#### 步骤1: 计算Replica数量

```python
def _initialize_llm_servers(rollout_resource_pool):
    # 每个replica需要的GPU数量
    rollout_world_size = (
        config.tensor_model_parallel_size   # TP
        * config.data_parallel_size          # DP
        * config.pipeline_model_parallel_size # PP
    )
    # 例如: TP=2, DP=4, PP=1 → rollout_world_size = 8

    # 总的可用GPU数量
    world_size = (
        worker_group.world_size if worker_group
        else config.n_gpus_per_node * config.nnodes
    )
    # 例如: 4 nodes × 8 GPUs = 32 GPUs

    # 可以创建的replica数量
    num_replicas = world_size // rollout_world_size
    # 例如: 32 / 8 = 4 replicas
```

**示意图**:
```
总GPU资源: 32 GPUs (4 nodes × 8 GPUs/node)
每个replica: 8 GPUs (TP=2, DP=4)

Node 0        Node 1        Node 2        Node 3
[0,1,2,3,    [8,9,10,11,   [16,17,18,19, [24,25,
 4,5,6,7]     12,13,14,15]  20,21,22,23]  26,27,28,29,30,31]
    ↓             ↓             ↓             ↓
Replica 0     Replica 1     Replica 2     Replica 3
GPUs 0-7      GPUs 8-15     GPUs 16-23    GPUs 24-31
```

#### 步骤2: 创建RolloutReplica实例

```python
self.rollout_replicas = [
    self.rollout_replica_class(
        replica_rank=replica_rank,
        config=rollout_config,
        model_config=model_config,
        gpus_per_node=config.n_gpus_per_node
    )
    for replica_rank in range(num_replicas)
]
```

#### 步骤3: 根据模式初始化

verl支持三种rollout模式:

**模式1: HYBRID - 训练和推理共用进程**

```python
if worker_group and rollout_config.name != "trtllm":
    self._run_all([
        server.init_hybrid(worker_group)
        for server in self.rollout_replicas
    ])

# RolloutReplica.init_hybrid():
async def init_hybrid(worker_group):
    self.rollout_mode = RolloutMode.HYBRID

    # 从worker_group中分配workers
    self.workers = worker_group.workers[
        self.world_size * self.replica_rank :
        self.world_size * (self.replica_rank + 1)
    ]

    # 在这些workers上启动LLM server
    await self.launch_servers()
```

**特点**:
- 共享进程和GPU
- 通过`wake_up()`/`sleep()`切换训练/推理
- 适用于on-policy训练 (PPO, GRPO)

**模式2: COLOCATED - 同一placement group，不同进程**

```python
elif worker_group and rollout_config.name == "trtllm":
    self._run_all([
        server.init_hybrid_colocated(worker_group, rollout_resource_pool)
        for server in self.rollout_replicas
    ])
```

**特点**:
- 独立进程但共享GPU
- 无需频繁weight sync
- 适用于TensorRT-LLM, GRM场景

**模式3: STANDALONE - 完全独立资源**

```python
else:
    self._run_all([
        server.init_standalone()
        for server in self.rollout_replicas
    ])

# RolloutReplica.init_standalone():
async def init_standalone():
    # 1. 创建独立的resource pool
    resource_pool_spec = {
        f"rollout_pool_{replica_rank}": [gpus_per_node] * nnodes
    }
    resource_pool_manager.create_resource_pool()

    # 2. 创建worker group
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=self.get_ray_class_with_init_args(),
        name_prefix=f"rollout_standalone_{replica_rank}"
    )

    # 3. 启动servers
    await self.launch_servers()
```

**特点**:
- 拥有专属GPU资源
- 与training完全解耦
- 适用于off-policy训练

#### 步骤4: 获取Server地址

```python
self.server_handles = [server._server_handle for server in self.rollout_replicas]
self.server_addresses = [server._server_address for server in self.rollout_replicas]

print(f"AgentLoopManager: {self.server_addresses}")
# 输出:
# ['http://10.0.0.1:8000/v1',
#  'http://10.0.0.2:8000/v1',
#  'http://10.0.0.3:8000/v1',
#  'http://10.0.0.4:8000/v1']
```

### 3.2 初始化Agent Loop Workers

**方法**: `AgentLoopManager._init_agent_loop_workers()` (lines 926-941)

```python
def _init_agent_loop_workers():
    num_workers = config.agent.num_workers

    # 1. 获取所有可用的Ray nodes
    node_ids = [
        node["NodeID"]
        for node in ray.nodes()
        if node["Alive"] and node["Resources"].get("CPU", 0) > 0
    ]

    # 2. Round-robin创建workers
    for i in range(num_workers):
        node_id = node_ids[i % len(node_ids)]

        worker = agent_loop_workers_class.options(
            name=f"agent_loop_worker_{i}_{uuid4().hex[:8]}",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=True  # 软亲和性
            )
        ).remote(
            self.config,
            self.server_handles,      # 所有replica的handles
            self.reward_router_address
        )

        self.agent_loop_workers.append(worker)
```

**Worker分布示意**:
```
假设: 4 nodes, 8 agent loop workers

Node 1: worker_0, worker_4
Node 2: worker_1, worker_5
Node 3: worker_2, worker_6
Node 4: worker_3, worker_7

每个worker通过AsyncLLMServerManager访问所有4个LLM server replicas
```

---

## 四、状态转换详解

### 4.1 PENDING → GENERATING

```python
async def _handle_pending_state(agent_data, sampling_params):
    """准备初始prompt"""
    prompt_ids = await self.apply_chat_template(
        agent_data.messages,
        tools=self.tool_schemas,      # 注入工具schema
        images=agent_data.image_data,
        videos=agent_data.video_data
    )
    agent_data.prompt_ids = prompt_ids
    return AgentState.GENERATING
```

### 4.2 GENERATING → PROCESSING_TOOLS/INTERACTING/TERMINATED

```python
async def _handle_generating_state(agent_data, sampling_params):
    """LLM生成，提取tool calls，决定下一状态"""

    # 1. 调用LLM生成
    output = await self.server_manager.generate(
        request_id=agent_data.request_id,  # sticky session
        prompt_ids=agent_data.prompt_ids,
        sampling_params=sampling_params,
        image_data=agent_data.image_data,
        video_data=agent_data.video_data
    )

    # 2. 更新状态
    agent_data.assistant_turns += 1
    agent_data.response_ids = output.token_ids
    agent_data.prompt_ids += agent_data.response_ids  # 累积
    agent_data.response_mask += [1] * len(agent_data.response_ids)
    if output.log_probs:
        agent_data.response_logprobs += output.log_probs

    # 3. 检查终止条件
    if len(agent_data.response_mask) >= self.response_length:
        return AgentState.TERMINATED
    if agent_data.assistant_turns >= self.max_assistant_turns:
        return AgentState.TERMINATED
    if agent_data.user_turns >= self.max_user_turns:
        return AgentState.TERMINATED

    # 4. 提取工具调用
    _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
        agent_data.response_ids
    )

    # 5. 添加assistant message (如果有interaction)
    if self.interaction_config_file:
        assistant_message = tokenizer.decode(agent_data.response_ids)
        agent_data.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

    # 6. 决定下一状态
    if agent_data.tool_calls:
        return AgentState.PROCESSING_TOOLS
    elif self.interaction_config_file:
        return AgentState.INTERACTING
    else:
        return AgentState.TERMINATED
```

### 4.3 PROCESSING_TOOLS → GENERATING

```python
async def _handle_processing_tools_state(agent_data):
    """执行工具调用，将结果添加到prompt"""

    # 1. 并发执行工具调用
    tasks = []
    tool_call_names = []
    for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
        tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
        tool_call_names.append(tool_call.name)

    responses = await asyncio.gather(*tasks)

    # 2. 处理工具返回
    add_messages = []
    new_images_this_turn = []

    for tool_response, tool_reward, _ in responses:
        # 构造tool message
        if tool_response.image or tool_response.video:
            # Multi-modal content
            content = []
            if tool_response.image:
                content.append({"type": "image"})
                new_images_this_turn.append(tool_response.image)
            if tool_response.text:
                content.append({"type": "text", "text": tool_response.text})
            message = {"role": "tool", "content": content}
        else:
            # Text-only content
            message = {"role": "tool", "content": tool_response.text or ""}

        add_messages.append(message)

        if tool_reward is not None:
            agent_data.tool_rewards.append(tool_reward)

    # 3. 更新messages
    agent_data.messages.extend(add_messages)

    # 4. 将tool responses转为token ids
    if self.tool_parser_name == "gpt-oss":
        # 手动格式化
        tool_response_text = build_gpt_oss_tool_response_text(
            add_messages, tool_call_names
        )
        response_ids = tokenizer.encode(tool_response_text, add_special_tokens=False)
    else:
        response_ids = await self.apply_chat_template(
            add_messages,
            images=new_images_this_turn,
            remove_system_prompt=True
        )

    # 5. 检查是否超长
    if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
        return AgentState.TERMINATED

    # 6. 更新状态
    if new_images_this_turn:
        agent_data.image_data.extend(new_images_this_turn)

    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)  # tool response标记为0
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(response_ids)

    agent_data.user_turns += 1

    return AgentState.GENERATING
```

### 4.4 INTERACTING → GENERATING/TERMINATED

```python
async def _handle_interacting_state(agent_data):
    """从环境获取响应"""

    # 1. 调用Interaction.generate_response
    (
        should_terminate_sequence,  # 环境决定是否终止
        interaction_responses,      # 环境的文本响应
        reward,                     # 当前turn的得分
        metrics,                    # 额外指标
    ) = await agent_data.interaction.generate_response(
        agent_data.request_id,      # instance_id
        agent_data.messages,        # 完整对话历史
        **agent_data.interaction_kwargs
    )

    # 2. 更新状态
    agent_data.user_turns += 1

    # 3. 添加环境响应到对话历史
    add_messages = [{"role": "user", "content": interaction_responses}]
    agent_data.messages.extend(add_messages)

    # 4. 记录reward
    if reward is not None:
        agent_data.turn_scores.append(reward)

    # 5. 将环境响应转换为tokens
    response_ids = await self.apply_chat_template(
        add_messages,
        remove_system_prompt=True
    )

    # 6. 更新prompt和mask
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)  # 环境响应标记为0
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(response_ids)

    # 7. 决定下一状态
    if should_terminate_sequence:
        return AgentState.TERMINATED
    else:
        return AgentState.GENERATING
```

---

## 五、工具调用机制

### 5.1 ToolParser

**位置**: `verl/experimental/agent_loop/tool_parser.py`

**职责**: 从LLM输出中提取工具调用

```python
class ToolParser(ABC):
    @abstractmethod
    async def extract_tool_calls(responses_ids: list[int])
        -> tuple[str, list[FunctionCall]]:
        """返回: (剩余文本, 工具调用列表)"""
        pass
```

**支持的格式**:

1. **Hermes格式**:
```xml
<tool_call>
{"name": "get_temperature", "arguments": {"location": "Beijing"}}
</tool_call>
```

2. **GPT-OSS格式**:
```
<|start|>assistant<|channel|>analysis<|message|>...thinking...<|end|>
<|start|>assistant<|channel|> to=functions.get_temperature
<|constrain|>json<|message|>{"location": "Beijing"}<|call|>
```

### 5.2 Tool执行

```python
async def _call_tool(tool_call, tools_kwargs, agent_data):
    """执行单个工具调用"""
    try:
        # 1. 解析tool call
        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments)
        tool = self.tools[tool_name]

        # 2. 创建tool instance
        kwargs = tools_kwargs.get(tool_name, {})
        instance_id, _ = await tool.create(
            create_kwargs=kwargs.get("create_kwargs", {})
        )

        # 3. 执行tool
        tool_response, tool_reward, res = await tool.execute(
            instance_id,
            tool_args,
            agent_data=agent_data  # 传入完整状态
        )

        # 4. 截断过长的响应
        if len(tool_response.text) > max_tool_response_length:
            tool_response.text = truncate(
                tool_response.text,
                side=truncate_side
            )

        return ToolResponse(
            text=tool_response.text,
            image=tool_response.image,  # optional
            video=tool_response.video   # optional
        ), tool_reward, res

    except Exception as e:
        logger.warning(f"Error executing tool: {e}")
        return ToolResponse(text=f"Error: {e}"), 0.0, {}

    finally:
        if tool and instance_id:
            await tool.release(instance_id)
```

---

## 六、环境交互机制

### 6.1 BaseInteraction

**位置**: `verl/interactions/base.py`

```python
class BaseInteraction:
    """所有交互环境的基类"""

    async def start_interaction(instance_id=None, **kwargs) -> str:
        """初始化交互实例"""
        if instance_id is None:
            return str(uuid4())
        return instance_id

    async def generate_response(
        instance_id: str,
        messages: list[dict],
        **kwargs
    ) -> tuple[bool, str, float, dict]:
        """生成环境响应

        Returns:
            should_terminate_sequence (bool): 是否终止
            response_content (str): 环境响应文本
            current_turn_score (float): 当前轮得分
            additional_data (dict): 额外元数据
        """
        return False, "Continue working.", 0.8, {}

    async def calculate_score(**kwargs) -> float:
        """计算得分"""
        return 0.0

    async def finalize_interaction(**kwargs) -> None:
        """清理资源"""
        pass
```

### 6.2 Gsm8kInteraction示例

**位置**: `verl/interactions/gsm8k_interaction.py`

```python
class Gsm8kInteraction(BaseInteraction):
    """GSM8K数学问题交互环境"""

    def __init__(self, config):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(instance_id=None, ground_truth=None, **kwargs):
        """保存ground_truth"""
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0
        }
        return instance_id

    async def generate_response(instance_id, messages, **kwargs):
        """判断LLM答案是否正确"""
        # 1. 提取LLM最新回答
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content")
                break

        self._instance_dict[instance_id]["response"] = content

        # 2. 计算得分
        reward = await self.calculate_score(instance_id)

        # 3. 生成反馈
        if reward == 1.0:
            response = "Your response is correct!"
            should_terminate = True
        else:
            response = "Your response is incorrect! Reflect and try again."
            should_terminate = False

        return should_terminate, response, reward, {}

    async def calculate_score(instance_id, **kwargs):
        """使用gsm8k评分函数"""
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="strict"
        )

    async def finalize_interaction(instance_id, **kwargs):
        del self._instance_dict[instance_id]
```

### 6.3 配置和初始化

**配置文件** (`gsm8k_interaction_config.yaml`):
```yaml
interaction:
  - name: "gsm8k"
    class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
    config: {}
```

**初始化**:
```python
# ToolAgentLoop.__init__:
self.interaction_config_file = config.multi_turn.interaction_config_path
if self.interaction_config_file:
    self.interaction_map = initialize_interactions_from_config(
        self.interaction_config_file
    )
    # 返回: {"gsm8k": Gsm8kInteraction(...)}

# ToolAgentLoop.run:
if self.interaction_config_file:
    interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
    # 例如: {"name": "gsm8k", "ground_truth": "540"}

    interaction_name = interaction_kwargs["name"]
    interaction = self.interaction_map[interaction_name]

    await interaction.start_interaction(
        request_id,
        **interaction_kwargs
    )
```

---

## 七、完整执行流程示例

### 场景: GSM8K数学问题 + 工具调用

```
问题: "James decides to run 3 sprints 3 times a week.
      He runs 60 meters each sprint. How many meters does he run a week?"
正确答案: 540
工具: calculator
```

### 流程Trace

```
1. AgentLoopManager.generate_sequences(batch)
   ├─ wake_up all replicas (同步weights)
   ├─ chunk batch: [sample_0, sample_1, ...] → [chunk_0, chunk_1, ...]
   └─ ray.get([worker.generate_sequences(chunk) for worker, chunk])

2. AgentLoopWorker.generate_sequences(chunk)
   ├─ sampling_params = {temperature: 1.0, top_p: 1.0, ...}
   ├─ for each sample:
   │   └─ asyncio.create_task(_run_agent_loop(...))
   └─ outputs = await asyncio.gather(*tasks)

3. AgentLoopWorker._run_agent_loop(sample)
   ├─ agent_loop = hydra.utils.instantiate(
   │       config=_agent_loop_registry["tool_agent"],
   │       server_manager=AsyncLLMServerManager(...),
   │       ...
   │  )
   └─ output = await agent_loop.run(sampling_params, **kwargs)

4. ToolAgentLoop.run()
   ├─ messages = [{"role": "user", "content": "James decides..."}]
   ├─ interaction = Gsm8kInteraction(...)
   ├─ await interaction.start_interaction(request_id, ground_truth="540")
   ├─ agent_data = AgentData(messages, interaction, ...)
   └─ state = PENDING

5. _handle_pending_state()
   ├─ prompt_ids = apply_chat_template(messages, tools=[calculator_schema])
   │   # → "<|user|>James decides...<|tools|>[{calculator}]<|assistant|>"
   └─ return GENERATING

6. _handle_generating_state() - Turn 1
   ├─ output = server_manager.generate(prompt_ids)
   │   # LLM: "Let me calculate. 3 sprints × 3 days = <tool_call>..."
   ├─ response_ids = [token_ids]
   ├─ prompt_ids += response_ids
   ├─ response_mask += [1,1,1,...]
   ├─ tool_calls = extract_tool_calls(response_ids)
   │   # → [FunctionCall(name="calculator", arguments='{"expr":"3*3*60"}')]
   └─ return PROCESSING_TOOLS

7. _handle_processing_tools_state()
   ├─ tool_response = await calculator.execute("3*3*60")
   │   # → ToolResponse(text="540")
   ├─ add_messages = [{"role": "tool", "content": "540"}]
   ├─ response_ids = apply_chat_template(add_messages)
   ├─ prompt_ids += response_ids
   ├─ response_mask += [0,0,0,...]  # tool response
   └─ return GENERATING

8. _handle_generating_state() - Turn 2
   ├─ output = server_manager.generate(prompt_ids)
   │   # LLM: "The answer is 540 meters."
   ├─ response_ids = [token_ids]
   ├─ prompt_ids += response_ids
   ├─ response_mask += [1,1,1,...]
   ├─ tool_calls = []
   ├─ add_messages = [{"role": "assistant", "content": "The answer is 540 meters."}]
   └─ return INTERACTING

9. _handle_interacting_state() - Turn 1
   ├─ (should_terminate, response, reward, _) =
   │   interaction.generate_response(request_id, messages)
   │   ├─ extract LLM answer: "540 meters"
   │   ├─ compute_score("540 meters", "540") → 1.0
   │   └─ return (True, "Your response is correct!", 1.0, {})
   ├─ turn_scores.append(1.0)
   ├─ add_messages = [{"role": "user", "content": "Your response is correct!"}]
   ├─ response_ids = apply_chat_template(add_messages)
   ├─ prompt_ids += response_ids
   ├─ response_mask += [0,0,0,...]  # env response
   └─ return TERMINATED (should_terminate=True)

10. 构造AgentLoopOutput
    ├─ prompt_ids = [initial_prompt_tokens]
    ├─ response_ids = [
    │     llm_turn1_tokens,           # "Let me calculate..."
    │     0,0,0...tool_response...0,  # "540"
    │     llm_turn2_tokens,           # "The answer is 540 meters"
    │     0,0,0...env_feedback...0    # "Your response is correct!"
    │  ]
    ├─ response_mask = [
    │     1,1,1...turn1,       # LLM
    │     0,0,0...tool,        # Tool
    │     1,1,1...turn2,       # LLM
    │     0,0,0...env          # Env
    │  ]
    ├─ num_turns = 5  # user + assistant + tool + assistant + env
    └─ extra_fields = {
          "turn_scores": [1.0],
          "tool_rewards": []
       }

11. AgentLoopWorker._agent_loop_postprocess(output)
    ├─ padding (左padding prompt, 右padding response)
    ├─ compute attention_mask, position_ids
    ├─ optional: compute_score via reward_loop_worker
    └─ return _InternalAgentLoopOutput(...)

12. AgentLoopWorker._postprocess(outputs)
    ├─ stack所有samples
    ├─ batch = TensorDict({prompts, responses, response_mask, ...})
    ├─ non_tensor_batch = {__num_turns__, reward_extra_info, ...}
    └─ return DataProto(batch, non_tensor_batch, meta_info)

13. AgentLoopManager.generate_sequences() 完成
    ├─ output = DataProto.concat(outputs)
    ├─ sleep all replicas
    ├─ compute metrics
    └─ return output
```

### 最终输出格式

```python
DataProto(
    batch=TensorDict({
        "prompts": [bsz, prompt_length],           # 左padding
        "responses": [bsz, response_length],       # 右padding
        "response_mask": [bsz, response_length],   # 1=LLM, 0=tool/env
        "input_ids": [bsz, prompt_length + response_length],
        "attention_mask": [bsz, prompt_length + response_length],
        "position_ids": [bsz, prompt_length + response_length],
        "rollout_log_probs": [bsz, response_length],  # optional
        "rm_scores": [bsz, response_length]           # optional
    }),
    non_tensor_batch={
        "__num_turns__": np.array([5, 4, 6, ...]),  # 每个sample的turn数
        "reward_extra_info": {...},
        "multi_modal_inputs": np.array([...], dtype=object),  # optional
        "raw_prompt": np.array([...], dtype=object)
    },
    meta_info={
        "timing": {
            "agent_loop/generate_sequences/mean": 2.5,
            "agent_loop/tool_calls/mean": 0.3,
            ...
        },
        "reward_extra_keys": [...]
    }
)
```

---

## 八、关键设计特点

### 8.1 异步并发

- **Batch内并发**: 所有samples在worker内异步并发执行
- **工具并发**: 同一turn内多个tool calls并发执行
- **Worker并发**: 多个workers并行处理不同chunks

### 8.2 Sticky Session + Prefix Caching

```python
# 同一trajectory的多轮对话路由到同一server
server = AsyncLLMServerManager._choose_server(request_id)
# 利用vLLM/SGLang的prefix caching减少重复计算
```

### 8.3 Response Mask机制

```python
# 精确区分LLM生成的token和外部输入的token
response_mask = [
    1,1,1,  # LLM generated
    0,0,0,  # tool response
    1,1,1,  # LLM generated
    0,0,0   # env response
]
# 只有标记为1的token参与loss计算
```

### 8.4 三种Rollout模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **HYBRID** | 训练和推理共享进程 | On-policy训练 (PPO, GRPO) |
| **COLOCATED** | 独立进程，共享GPU | TensorRT-LLM, GRM |
| **STANDALONE** | 独立资源池 | Off-policy训练 |

### 8.5 灵活的扩展机制

- **Agent Loop Registry**: 动态注册和切换agent类型
- **Tool Registry**: 动态加载工具
- **Interaction Registry**: 动态加载交互环境
- **ToolParser Registry**: 支持多种tool calling格式

### 8.6 完善的状态管理

- **AgentData**: 封装所有状态变量
- **Instance-level管理**: 每个trajectory独立的interaction instance
- **Turn-level Reward**: 即时反馈用于RL训练
- **Multi-modal支持**: 统一处理文本、图像、视频

### 8.7 性能优化

- **Free Cache Engine**: 训练时释放推理cache
- **Load Balancing**: 基于请求数的动态负载均衡
- **Round-robin调度**: Worker均匀分布到所有节点
- **Prometheus监控**: 实时性能指标收集

---

## 文件导航

**核心实现**:
- `verl/experimental/agent_loop/agent_loop.py` - 管理器和基础组件
- `verl/experimental/agent_loop/tool_agent_loop.py` - 工具调用Agent (476行)
- `verl/experimental/agent_loop/single_turn_agent_loop.py` - 单轮Agent (85行)
- `verl/experimental/agent_loop/tool_parser.py` - 工具解析器 (162行)

**交互系统**:
- `verl/interactions/base.py` - 交互基类 (73行)
- `verl/interactions/gsm8k_interaction.py` - GSM8K示例 (88行)
- `verl/interactions/utils/interaction_registry.py` - 交互注册器 (86行)

**Rollout后端**:
- `verl/workers/rollout/replica.py` - Replica抽象 (300行)
- `verl/workers/rollout/vllm_rollout/` - vLLM实现
- `verl/workers/rollout/sglang_rollout/` - SGLang实现

**测试**:
- `tests/experimental/agent_loop/test_basic_agent_loop.py` - 基础测试

**示例**:
- `examples/sglang_multiturn/` - SGLang多轮示例
- `examples/tutorial/agent_loop_get_started/` - 入门教程

---

## 总结

VERL的agent loop系统是一个设计精巧的多轮对话和工具调用框架，核心特点包括：

1. **分层架构**: Manager → Workers → Agent Loops，职责清晰
2. **状态机模式**: PENDING → GENERATING → PROCESSING_TOOLS/INTERACTING → TERMINATED
3. **异步并发**: 充分利用asyncio实现高并发
4. **灵活扩展**: 支持多种agent类型、工具、交互环境
5. **性能优化**: Sticky session, prefix caching, load balancing
6. **RL友好**: Turn-level reward, response mask, 支持on/off-policy训练

这个系统为训练multi-turn, tool-using的LLM agent提供了完整的基础设施。

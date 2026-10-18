<p align="center">
  <img src="docs/assets/banner.jpg" alt="tsuzuri" width="100%">
</p>

<p align="center">
  <a href="docs/README.md">文档</a> ·
  <a href="examples/">示例</a> ·
  <a href="docs/evaluation.md">评测</a> ·
  <a href="README.md">English</a> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <a href="https://github.com/puresky271/tsuzuri/actions/workflows/ci.yml"><img src="https://github.com/puresky271/tsuzuri/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11 | 3.12 | 3.13">
  <img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Status: alpha">
  <img src="https://img.shields.io/badge/runs-offline%20by%20default-2ea44f.svg" alt="Runs offline by default">
  <img src="https://img.shields.io/badge/LLM-any%20OpenAI--compatible-6f42c1.svg" alt="Any OpenAI-compatible LLM">
  <img src="https://img.shields.io/badge/prompts-%E4%B8%AD%E6%96%87%20first-red.svg" alt="Prompts: Chinese first">
</p>

> [!NOTE]
> **tsuzuri 目前是 alpha（0.1）**，小版本之间 API 可能变化。prompt 与文本启发式**以中文为主**：其他语言可以正常存取和召回，但抽取质量按中文调优。

tsuzuri（綴り，"缀连"）记住用户分别对每个角色说过的话，判断哪些与下一句回复相关，并把它们编进 prompt——一周后角色可以自然地问一句"你周三那场面试怎么样了？"，而不需要把全部历史塞进上下文。

它为这样的产品设计：几个长期存在的角色（agent）和同一个用户一对一或多人聊天。

- **按角色分层的记忆**：working → episodic → semantic，另有 procedural 规则和全体共享的 *collective* 作用域。写入由规则规划（LLM 不直接写主表）；召回是混合检索（词法/FTS、可选稠密与稀疏向量、结构化槽位、重排、MMR），带 ACT-R 式衰减。
- **轮次事实卡（TurnFact）**：每轮对话提炼成一张小卡片——事实、状态变化、承诺、纠正、未完话题，跨会话延续并让过时的说法失效。
- **上下文编译器**：`compile_fact_workspace` 等函数负责挑选、去重并控制预算：当前状态、相关事实、召回记忆、对话脉络、按需证据，以及放在模型注意力最强位置的本轮执行包。
- **离线巩固（"做梦"）**：晋升与衰减记忆、用新证据重审语义事实、把情景聚成事件/话题叙事、重建核心摘要与实体投影、扫描遗忘候选。
- **多人记忆**：多个角色共享一段带说话人归属的历史、房间级事实卡、共享的集体作用域，以及在角色之间按置信度衰减传播的传闻。
- **角色自定义**：角色名、别名、群体称呼、用户称呼、每个角色的风格规则和词表都是配置，不写在代码里。

## 安装

```bash
pip install -e .                 # 核心依赖：numpy、scipy、jieba
pip install -e ".[llm]"          # OpenAI 兼容客户端（抽取 / rethink / 回复）
pip install -e ".[embeddings]"   # 稠密召回（sentence-transformers / FlagEmbedding，下载较大）
pip install -e ".[dev]"          # pytest、ruff、build
```

需要 Python 3.11+。

## 快速开始（离线、无需 API key）

```python
from tsuzuri import Tsuzuri

memory = Tsuzuri("./.tsuzuri")
memory.chat("林栖", "我下周三上午要去面试，有点紧张")
memory.chat("林栖", "对了，我不太能吃辣")

turn = memory.chat("林栖", "你还记得我下周三的面试吗？")
print(turn.reply)        # 离线应答器根据编译后的工作区作答
print(turn.recalled)     # 本轮浮现的记忆行
print(turn.fact_card)    # 本轮提炼的事实卡
```

> [!TIP]
> `tsuzuri demo` 会跑一段脚本化对话、一次召回检查和一个多人房间——不需要 API key，也不需要下载模型。

没有 LLM 时，事实抽取走规则，回复来自确定性的替身（`tsuzuri.offline.offline_reply`）——足够观察记忆闭环、跑测试和评测。正式使用请接入模型。

## 接入模型

```python
from tsuzuri import Tsuzuri, openai_compatible

llm, chat = openai_compatible(model="your-model", base_url="https://...", api_key="...")
memory = Tsuzuri("./.tsuzuri", llm=llm, reply=chat)
```

`llm`（`prompt -> str`）用于事实抽取、语义重审和巩固；`reply`（`messages -> str`）用编译好的消息生成角色回复。也可以只把 tsuzuri 当记忆层：调用 `memory.build_messages(agent, text)`，把结果交给你自己的模型栈。

## 自定义角色

见 [docs/configuration.md](docs/configuration.md) 与 `examples/custom_cast.json`；通过 `personas=`、`PERSONA_CONFIG_PATH` 或 `configure_personas()` 加载。

## 多人对话

```python
room = memory.room(["林栖", "苏芮", "叶澄"], shared=True)
turn = room.say("周六我们一起去爬山吧，七点楼下集合")
room.share("苏芮", "阿澈下个月要去参加摄影比赛")   # 让其他成员之后"听说"
```

## 📊 评测

`tsuzuri eval` 在隔离的临时存储里回放短的合成历史，检查该出现的信息在需要时是否对模型可见。

| 类别 | 离线、词法召回 | + 稠密向量（bge-small） |
|---|---|---|
| 偏好 | 1/3 | 1/3 |
| 将来的计划 | 0/3 | 1/3 |
| 稳定事实 | 1/2 | 2/2 |
| 纠正（以最新为准） | 1/2 | 1/2 |
| 隔离（不泄露，且本人仍能召回） | 1/2 | 2/2 |
| 多人房间 | 2/2 | 2/2 |
| 传闻 | 1/1 | 1/1 |
| 多轮线索 | 0/1 | 1/1 |
| **合计** | **7/16 (44%)** | **11/16 (69%)** |

内置合成集 v1，提问在**新会话**中进行（没有原文、没有会话事实卡，只能靠长期记忆作答），规则抽取，未接 LLM；LLM 抽取不在上表之内。复现：`tsuzuri eval`、`tsuzuri eval --embeddings`。用例格式见 [docs/evaluation.md](docs/evaluation.md)。

> [!NOTE]
> 离线数字只衡量词法召回：问题和记忆有共同词才能召回（"面试" ↔ "面试"），只在语义上相关时召不回（"我下周三要做什么" ↔ "面试"）。稠密向量和 LLM 抽取能补上大部分差距；请用你自己的配置和用例评测。

> [!WARNING]
> 这是 16 条用例的**可见性**检查（期望文本是否出现在模型读到的内容里），不是端到端回答正确率，不能与 LoCoMo / LongMemEval 分数比较。persona 提示词不计入；传闻必须进入编译后的上下文；隔离用例同时要求本人能召回（阳性对照）。

**已知局限**：几天后的自然追问（不说"你还记得"）常被判为静默接地而不浮现；离线规则抽取只是原话回声，不抽承诺/待办，房间事实不写入长期记忆；一个存储只服务一个用户；暂无硬删除/导出 API（删除数据目录即可清除）。

> [!IMPORTANT]
> 用户说的话都可能存进数据目录（`memory.db`、`sessions/`）。请像保护个人数据一样保护它；API key 放环境变量或不入库的 `tsuzuri.toml`；召回的文本对模型而言是不可信输入。

## 与应用集成

引擎从不 import 你的应用。需要只有宿主知道的东西（角色完整 system prompt、设定事实、OOC 分类器、反思投递面……）时，它通过 `tsuzuri.memory_host` 请求可选的具名钩子；缺失的钩子会跳过对应步骤。见 [docs/hosting.md](docs/hosting.md)。

## 贡献与许可

欢迎 issue 与 PR，请先读 [CONTRIBUTING.md](CONTRIBUTING.md)；安全问题见 [SECURITY.md](SECURITY.md)。Apache License 2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

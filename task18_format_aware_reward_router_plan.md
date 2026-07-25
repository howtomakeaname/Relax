# Task 18 Plan: Format-aware Reward Router

## 问题

任务来源：

- 认领 Issue: https://github.com/redai-infra/Relax/issues/86
- 正式任务文档: `community/contributor-program/2026-cohort-1/official-task.md`
- 任务编号: 18
- 任务名称: Format-aware reward router
- Relax 基线 commit: `039ce876d25540adad847d4223b4de4722d8f425`
- 开发分支: `task/18-format-aware-reward-router`
- 提交目标: `howtomakeaname/Relax:task/18-format-aware-reward-router` -> `redai-infra/Relax:main`

任务原文要求：

> 单一 `--rm-type` 无法支持一个 batch 内的 math、multiple-choice 等混合任务。目标是按 metadata/label 自动分发，支持 fallback 和一行式 registry。验收标准是混合 math + multiple-choice batch 逐样本命中正确 reward；未知/缺失/冲突类型走配置的 fallback 或 0 分并记录 warning；新增 reward 只需注册一行，不改路由分支；保留显式 `--rm-type` 的兼容策略；单测覆盖路由、fallback、registry 和混合 batch。

当前源码结论：

- `relax/utils/data/data_utils.py::process_raw_sample()` 已把 `data[metadata_key]` 写入 `Sample.metadata`，并会补充顶层 `data_source`。
- `relax/engine/rewards/__init__.py::RewardExecutor.execute()` 已有 `metadata["rm_type"]` 覆盖 `args.rm_type` 的雏形。
- `RewardWorker.compute()` 仍是 `if/elif` 分支，新增 reward 必须改路由分支，不满足 registry 要求。
- `dapo` 返回 dict: `{"score": 1.0/-1.0, "acc": bool, "pred": str}`；`multiple_choice` 返回 float。混合 batch 若继续依赖调用方手动设置统一 `reward_key`，容易在训练后处理和 eval 聚合处出错。
- 普通 rollout 和 agentic reward 最终都调用 `async_rm()` / `batched_async_rm()`，所以路由应收敛在 reward 层，不应在 rollout 层复制分支。
- 顶层 `relax.engine.rewards` 当前导入 `ray`、`aiohttp`、`math_verify` 相关模块，快速单测需要尽量隔离纯路由逻辑，避免无关运行时依赖。

不做范围：

- 不重写 rollout 调度、agentic pipeline、数据集加载器。
- 不改变现有训练脚本的显式 `--rm-type dapo` / `--rm-type multiple_choice` 行为。
- 不要求 GPU、不跑端到端训练。
- 不把自定义 reward worker 池并发重构并入本题；那是第 19 题范围。

## 方案

目标文件结构草案：

```diff
+ relax/engine/rewards/registry.py
+ relax/engine/rewards/router.py
  relax/engine/rewards/__init__.py
  relax/utils/arguments.py
  relax/utils/types.py
  relax/utils/utils.py
+ tests/engine/rewards/test_reward_registry.py
+ tests/engine/rewards/test_reward_router.py
  tests/engine/rewards/test_reward_worker.py
```

### 1. Reward registry

新增 `relax/engine/rewards/registry.py`，职责是维护 reward 名称到 handler 的注册表。

设计要点：

- 注册表是 O(1) 字典查找。
- handler 用轻量 wrapper 或 lazy loader，避免 import registry 时就加载所有重依赖。
- 内置 reward 在模块底部集中注册，新增 reward 只需要一行注册。
- `RewardWorker.compute()` 只做 registry 查找和执行，不再维护 `if/elif` 链。

目标形态：

```diff
- if rm_type == "multiple_choice":
-     return get_multiple_choice_reward(response, label)
- elif rm_type == "dapo":
-     return compute_score_dapo(response, label)
- ...
+ handler = get_sync_reward_handler(rm_type)
+ return handler(response=response, label=label, metadata=metadata)
```

### 2. Format-aware router

新增 `relax/engine/rewards/router.py`，职责是解析单个 sample 应使用的 reward 类型。

解析优先级：

```diff
+ 1. sample.metadata["rm_type"] / reward_type / reward_format / task_type
+ 2. 显式 args.rm_type，保持历史兼容
+ 3. metadata 可识别字段，例如 choices、valid_letters、data_source、source
+ 4. label 格式启发，例如 <answer>A</answer> 识别为 multiple_choice，普通 math 标准答案识别为 dapo
+ 5. fallback rm_type
+ 6. 0 分并记录 warning
```

冲突处理：

- metadata 中多个类型字段若指向不同 reward，判定为冲突。
- 冲突时优先走配置的 fallback。
- 没有 fallback 时返回 0 分并记录 warning，不让 batch 因单条脏样本整体失败。

显式 `--rm-type` 兼容策略：

- 用户显式设置 `--rm-type dapo` 时，仍按旧行为作为全局默认。
- 样本 metadata 中显式 `rm_type` 仍可覆盖全局默认，保留当前能力。
- 用户不设置 `--rm-type` 时，启用 metadata/label 自动解析。
- 可增加 `--rm-type auto` 作为显式自动路由入口，但不依赖它完成兼容。

fallback 参数：

```diff
+ --reward-router-fallback-rm-type
```

含义：

- 设置后，未知、缺失、冲突类型走该 reward。
- 不设置时，未知、缺失、冲突类型返回 0 分并 warning。

### 3. Reward 返回值数值化

当前 `Sample.get_reward_value()` 在 `args.reward_key` 为空时直接返回 `sample.reward`。如果 `dapo` 返回 dict 且混入 float reward，会影响 `post_process_rewards()`、eval 聚合和 `torch.tensor(raw_rewards)`。

处理方式：

- 保留 `sample.reward` 原始返回形态，避免破坏现有显式 reward 指标。
- 新增或增强统一取值 helper：

```diff
- return self.reward if not args.reward_key else self.reward[args.reward_key]
+ if args.reward_key:
+     return self.reward[args.reward_key]
+ if isinstance(self.reward, dict) and "score" in self.reward:
+     return self.reward["score"]
+ return self.reward
```

- 普通训练后处理继续通过 `Sample.get_reward_value(args)` 取数值。
- eval 输出中涉及 reward 数值列表的地方改走同一 helper，避免 `reward_key` 为空时输出 dict。

### 4. Warning 与效率

- warning 使用 logger，包含 sample index、metadata keys、label 摘要、解析原因。
- 对重复类型错误做限频或缓存，避免大 batch 日志刷屏。
- 自动解析只做轻量字符串和 dict 检查，不调用 reward 函数、不做正则大扫描。
- registry handler 加载一次后缓存，避免每个样本重复 import。

### 5. 测试矩阵

快速测试，纯 CPU，不启动 Ray：

```bash
python -m pytest -v \
  tests/engine/rewards/test_reward_registry.py \
  tests/engine/rewards/test_reward_router.py
```

覆盖内容：

- registry 注册、重复注册、未知类型错误、内置类型可查。
- metadata `rm_type` 精确路由。
- metadata alias 路由：`reward_type` / `reward_format` / `task_type`。
- label 自动识别：math 样本命中 `dapo`，`<answer>B</answer>` 命中 `multiple_choice`。
- 混合 math + multiple-choice batch 每条样本取到正确 reward。
- 未知类型无 fallback 返回 0 分并记录 warning。
- 未知类型有 fallback 时命中 fallback。
- 冲突 metadata 走 fallback 或 0 分。
- `Sample.get_reward_value()` 对 float、dict + `score`、dict + 显式 `reward_key` 都返回正确数值。

可选集成测试，验证执行器链路：

```bash
python -m pytest -v tests/engine/rewards/test_reward_worker.py
```

若环境未安装 Ray 或 math_verify，此测试可沿用现有 skip 策略；新 router 的核心正确性不能依赖该测试。

回归测试：

```bash
python -m pytest -v tests/engine/rewards
python -m pytest -v tests/utils/test_multimodal_rollout_stats.py
```

格式检查：

```bash
pre-commit run --all-files
```

PR 模板和 CI 要求：

- PR 模板路径：`.github/PULL_REQUEST_TEMPLATE.md`。
- PR 正文必须覆盖 `What`、`Why`、`How`、`Testing`、`Type of Change`、`Screenshots / Logs`。
- PR Testing 清单要求：
  - `pre-commit run --all-files` passes
  - Tests pass (`pytest tests/`)
  - New tests added (if applicable)
  - Documentation updated (if applicable)
- CI 包含三类检查：
  - Pre-commit Checks：Python 3.10 下运行全部 pre-commit hooks。
  - Lint：`ruff check relax/ tests/` 与 `ruff format --check relax/ tests/`。
  - Tests：Python 3.10 / 3.11 / 3.12 矩阵运行 `pytest tests/ -v --tb=short -x --ignore=tests/autoscale`。
- CI 依赖安装策略：
  - 先安装 CPU-only torch。
  - 安装 `pybase64`、`tensordict`。
  - `pip install --no-deps -e .`。
  - `pip install -r requirements.txt || true`，允许 GPU 或不可用包安装失败。
  - 创建 `transfer_queue.py` stub 处理内部包缺失。

本地最终验证命令：

```bash
source /home/yangruitao/projects/rinfra-test/.venv-task18/bin/activate
python -m pytest -v tests/engine/rewards
python -m pytest -v tests/utils/test_multimodal_rollout_stats.py
ruff check relax/ tests/
ruff format --check relax/ tests/
pre-commit run --all-files
```

若完整 `pytest tests/` 耗时可控，提交 PR 前补跑：

```bash
python -m pytest tests/ -v --tb=short -x --ignore=tests/autoscale
```

## 评估

正确性保障：

- 路由逻辑用纯函数测试覆盖，每个输入样本能明确断言解析到的 reward 类型。
- reward 执行测试覆盖实际 `dapo` 与 `multiple_choice` 返回结果，避免只测路由名。
- 数值提取测试覆盖 dict/float 混合返回，防止训练后处理阶段才暴露问题。
- fallback 和 warning 单独测试，覆盖未知、缺失、冲突三类异常输入。

可维护性保障：

- registry 和 router 分层：registry 管“有哪些 reward”，router 管“这个 sample 选哪个 reward”。
- 新增 reward 不改 `RewardWorker.compute()` 分支。
- 自动识别规则集中在 router，后续新增格式只改一处。
- 保留原始 `sample.reward`，不破坏已有指标字典和 `reward_key` 机制。

运行效率保障：

- 每样本路由是 O(1) 字典查找 + 少量 metadata/label 检查。
- handler lazy load 后缓存，不重复加载 reward 函数。
- fallback warning 做限频，避免异常数据放大日志开销。
- 快速单测不启动 Ray，开发循环控制在秒级。

踩坑清单：

- 不把 `metadata["rm_type"]` 已有能力当成完整答案；题目还要求 registry、fallback、自动分发。
- 不在 rollout 层复制 router，否则普通 rollout 和 agentic 可能分叉。
- 不让 `dapo` dict 和 `multiple_choice` float 在 `torch.tensor(raw_rewards)` 处混类型失败。
- 不用裸 `"A"` / `"B"` 作为 multiple-choice 标准答案测试，项目文档里的标准格式是 `<answer>B</answer>`。
- 不让纯测试直接依赖 `relax.engine.rewards` 顶层重导入导致 Ray/math_verify 成为硬门槛。
- 不改变现有脚本显式 `--rm-type` 的默认行为。

当前状态：

- 已在 `task/18-format-aware-reward-router` 分支完成实现、文档和测试。
- 已配置开发环境：`/home/yangruitao/projects/rinfra-test/.venv-task18`。
- venv 的 pip 源已配置为阿里云镜像：`https://mirrors.aliyun.com/pypi/simple/`。
- 清华源 `https://pypi.tuna.tsinghua.edu.cn/simple` 已尝试，但下载 `numpy` wheel 返回 HTTP 403，因此切换到阿里云镜像。
- 环境内工具版本：`pytest 9.1.1`、`ruff 0.16.0`、`pre-commit 4.6.1`。
- 已安装 CPU-only `torch 2.13.0+cpu`、`numpy 1.26.4`、`ray 2.56.1`、`aiohttp 3.14.3`、`math_verify 0.8.0`、`httpx 0.28.1`、`datasets 2.19.2`、`pyarrow 14.0.2`、`transformers 5.14.1`、`fastapi 0.139.1`、`sglang-router 0.3.2` 等依赖。
- 已执行 `pip install -r requirements.txt`，当前返回码为 0。
- 已按 CI 逻辑在 venv site-packages 中创建 `transfer_queue.py` stub。
- 已执行 `pre-commit install-hooks`，hook 环境已预拉取。
- 已验证 targeted reward 集成测试：

```bash
/home/yangruitao/projects/rinfra-test/.venv-task18/bin/python -m pytest -q -rs \
  tests/engine/rewards/test_reward_worker.py::TestRewardExecutorSingleSample::test_execute_metadata_rm_type_override \
  tests/engine/rewards/test_reward_worker.py::TestBatchedAsyncRM::test_batch_mixed_results
```

结果：`2 passed in 10.21s`。

- 已验证 reward + rollout stats targeted 回归：

```bash
/home/yangruitao/projects/rinfra-test/.venv-task18/bin/python -m pytest -v \
  tests/engine/rewards \
  tests/utils/test_multimodal_rollout_stats.py
```

结果：`59 passed in 49.18s`。

- 已验证 CI lint 口径：

```bash
/home/yangruitao/projects/rinfra-test/.venv-task18/bin/ruff check relax/ tests/
/home/yangruitao/projects/rinfra-test/.venv-task18/bin/ruff format --check relax/ tests/
```

结果：`All checks passed!`，`350 files already formatted`。

- 已验证完整 CI pytest 口径：

```bash
/home/yangruitao/projects/rinfra-test/.venv-task18/bin/python -m pytest tests/ \
  -v --tb=short -x --ignore=tests/autoscale
```

结果：`366 passed, 226 skipped, 33 warnings in 68.81s`。

- 已验证 PR 模板要求：

```bash
/home/yangruitao/projects/rinfra-test/.venv-task18/bin/pre-commit run --all-files
```

结果：全部 hook passed。

- 启用方式：

```bash
source /home/yangruitao/projects/rinfra-test/.venv-task18/bin/activate
```

- 等待提交、推送和创建 PR。创建 PR 前需先向用户确认。

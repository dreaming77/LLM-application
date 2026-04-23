# **基于Qwen大模型的智能家居问答系统**

技术栈：LangGraph｜LangChain｜Milvus｜Qwen｜FastAPI｜Vue3 

基座模型：Qwen2.5-7B-Instruct

嵌入模型：bge-large-zh-v1.5



## 内容概述

- 成功将大语言模型、向量数据库、智能工作流等前沿技术无缝集成，构建了完整的智能家居研究解决方案。
- 通过LangGraph实现的多节点工作流管理，突破了传统单一模型处理的局限性，实现了用户查询的精细化、专业化处理。
- Milvus向量数据库与RAG Fusion技术的结合，为家庭设备知识检索提供了新的效率和质量标准。



## Milvus库

1. 安装 Docker 和 Docker Compose
2. 下载Milvus-standalone的Docker Compose文件（/docker-compose.yml）

```
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

3. 启动milvus服务

```
# 启动 Milvus 服务（在后台运行）
sudo docker-compose up -d

# 查看服务状态
sudo docker-compose ps
```

4. 查看 Milvus 的存储后端，MinIO 存储了 Milvus 的实际数据文件。MinIO 的 Web 界面默认运行在 9000 端口，可以通过浏览器访问 `http://localhost:8000`（用户名: minioadmin, 密码: minioadmin）来查看存储的文件。
5. 数据预处理`milvus/milvus_processor.py`

```plaintext
总体流程：
  main函数启动
      ↓
  初始化MilvusDataProcessor
      ↓ （调用__init__）
  连接Milvus（_connect_milvus） + 加载Embedding模型
      ↓
  创建Milvus集合（create_collection：定义Schema → 删旧集合→建索引）
      ↓
  处理JSONL文件（process_jsonl_file）
      ↓ 循环每一行JSON
          单条数据清洗（_process_single_record）
          ↓ 积累到batch_size
              批量生成Embedding（_get_embedding）
                ↓
              批量插入Milvus（_insert_batch：批量失败则单条插入）
      ↓ 文件处理完成
  加载集合到内存 + 查询集合实体数（get_collection_info）
      ↓
  测试相似搜索（search_similar_questions）
      ↓
  关闭Milvus连接（close）
      ↓
  程序结束
```

> 数据集是jsonl格式，**每行一个独立 JSON 对象**。

1）定义数据字段id 、question、answer、label、related_diseases、score、question_vector（1024维度，用于向量查询）。

2）单条数据简单清洗，验证字段是否存在，不存在则跳过。

## 后端 - LangGraph

1. **状态定义**

> 状态 → 状态管理器 → 状态序列化器 → 状态序验证器。

1）`state.py` 是整个状态管理的基础，它通过 `GraphState` 这个 `TypedDict` 定义了工作流中需要跟踪的所有状态字段。**作用**：规定了状态的 “形状”，所有后续组件（管理器、序列化器、验证器）都基于此结构进行操作，确保状态数据的一致性。

2）`stateManager.py` 负责状态的创建、更新、查询和销毁，是状态的核心操作入口。

3）`stateSerializer.py` 负责将 `GraphState` 实例在 “内存对象” 和 “可存储 / 传输格式（如 JSON）” 之间转换。

4）`stateValidator.py` 负责验证状态的合法性和流程的正确性。

2. **节点定义**

> 意图识别 → 查询重写 → 文档检索 → RAG Fusion → 响应生成 → 响应优化。

1）`medical_intent_detection_node` 负责分析用户查询的意图，提取关键实体

2）`query_rewriting_node` 基于意图和实体，负责生成多个与原始查询相关的重写查询，提升后续文档检索的全面性。

3）`document_retrieval_node` 负责使用重写后的查询生成向量，从 Milvus 向量数据库中检索相关文档（问题 - 答案对）。<u>注意要熟悉去重策略</u> ⚠️

4）`rag_fusion_node` 负责对多个查询的检索结果进行加权融合（使用 Reciprocal Rank Fusion 算法），提升文档相关性排序。

5）`response_generation_node` 负责基于融合后的上下文、用户查询和意图，生成初始回答，并提取引用来源、计算置信度。

6）`response_refinement_node` 负责优化初始回答的格式（如修复空格、标点），根据风险级别（如高风险情况）调整内容，提升安全性和专业性。

7）`node_manager.py` 负责注册所有节点，建立节点名称与函数的映射，通过`execute_node`方法根据当前状态的 “next_step” 调用对应节点，确保流程按预定顺序执行。

3. **工作流定义**

> 用于构建和管研究助手的工作流系统。

1）`graph_definition.py` 专注于**工作流图的定义与配置**。

2）`workflow_manager.py` 该文件实现了**工作流的执行、监控和会话管理**。

4. **主函数 `main.py`**

该文件是整个问答助手 API 服务的入口，主要负责 FastAPI 应用的初始化、配置、路由定义及服务器启动，核心功能包括：

- **应用初始化**：定义了 FastAPI 应用实例，配置了标题、描述、版本等元信息，并通过`lifespan`上下文管理器实现系统启动初始化（如加载模型、工作流）和关闭清理逻辑。

- **CORS 配置**：添加跨域资源共享中间件，允许所有来源（生产环境需限制）的请求访问 API。

- 路由定义

  ：包含多个 API 端点，覆盖核心功能：

  - 基础信息接口（根路径`/`、系统信息`/api/system/info`、默认配置`/api/config/default`）；
  - 健康检查接口（基础健康检查`/health`、详细健康检查`/api/health/detail`、模型健康检查`/api/health/models`等）；
  - 核心业务接口（处理查询`/api/query`、继续对话`/api/conversation/{session_id}/continue`、会话状态管理`/api/session/{session_id}`等）。

- **错误处理**：自定义了 HTTP 异常和通用异常的处理器，统一返回格式。

- **服务器启动**：通过`uvicorn`启动服务，配置主机、端口、工作进程数等参数。

5. **研究控制器`controller.py`**

该文件定义了`MedicalResearchController`类，作为 API 的核心控制器，负责协调模型、工作流、状态管理等组件处理业务逻辑，核心功能包括：

- **初始化与关闭**：`initialize`方法负责初始化模型和工作流，`close`方法负责资源清理。
- **健康检查**：`health_check`方法整合 Milvus 数据库、工作流、模型的健康状态，返回系统整体健康信息；`model_health_check`专门检查嵌入模型和生成模型的可用性。
- **查询处理**：`process_medical_query`方法接收用户查询，验证合法性后通过工作流管理器处理查询并返回结果。
- **会话管理**：支持对话延续（`continue_conversation`，基于会话 ID 维持对话历史）、会话状态查询（`get_session_status`）、会话中断（`interrupt_processing`）等功能。
- **系统信息**：`get_system_info`返回系统版本、工作流信息等基础数据。

该控制器是连接 API 接口与底层业务逻辑（模型、工作流、数据库）的中间层，实现了请求的验证、转发和结果处理。

```
后端启动命令：
    uvicorn main:app --reload --host 0.0.0.0 --port 8888
```



## 前端



主要文件内容如下：

1. 前端入口文件（main.js）
2. 路由配置（router/index.js）
3. 状态管理（stores/index.js 和使用Pinia）
4. 服务层（services/api.js 用于调用后端API）
5. 视图文件（views/Home.vue, views/Query.vue, views/History.vue等）
6. 组件文件（components/NavBar.vue, components/QueryForm.vue, components/ResponseDisplay.vue等）
7. 样式文件（styles/global.css）
8. 静态资源（index.html）
9. 配置文件（vite.config.js, package.json）

```sh
前端启动命令：cd frontend/
    npm install
    npm run dev -- --host 0.0.0.0 --port 3000
```

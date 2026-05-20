"""非流式双工（Duplex Offline / Batch）Schema 定义

本模块在 ``core.schemas.duplex`` 已有的 ``DuplexOfflineInput`` 基础上做了扩展，
专门用于 **非流式批量推理** 场景：

- 评测：在评测集上跑双工模型，得到聚合的文本和音频
- 刷数据：对一批音视频生成模型回复，离线落盘
- 调试 / Regression：复现某段对话，无需保证实时性

与原 ``DuplexOfflineInput`` 的差异：

1. **输入多了 base64 路径**（兼容 client 不能共享文件系统的情况）
2. **加了批处理控制项**：``stop_on_end_of_turn`` / ``max_chunks``
3. **加了输出整段拼接选项**：``return_per_chunk_audio`` / ``return_merged_audio``
4. **加了 client 侧追踪字段**：``request_id`` / ``ticket_id`` / ``queue_wait_ms``

底层调用流程仍然是：

    prepare(system_prompt, ref_audio)
    for chunk_idx in 0..N:
        prefill(audio_chunk, image_frame?)
        result = generate()
        finalize()
        collect(result)
    stop() + cleanup()

整段音频和帧序列在 server 端被切成 ``chunk_ms`` 大小的小段，按序喂给模型。
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from core.schemas.duplex import (
    DuplexConfig,
    DuplexChunkResult,
)


# =============================================================================
# 输入
# =============================================================================

class DuplexBatchRequest(BaseModel):
    """非流式双工批量推理请求

    输入媒体的三种载体（**互斥**，优先级 path > base64 > 无）：

    - **path**: 服务端可读的绝对路径（推荐评测场景，零拷贝）
    - **base64**: 直接传 base64 数据（跨机调用、单元测试用）
    - **不传**: 仅文本对话时可以省略音频

    音频要求：16kHz mono float32（与实时 duplex 一致）。
    图像 ``image_paths`` / ``image_base64_list`` 是 **per-chunk** 的，
    即第 i 张对应第 i 个 ``chunk_ms`` 时间窗。若数量不足，server 会用最后一张
    填充至音频 chunk 数；若超出，超出的帧被丢弃。
    """

    # ---------- system / 模板 ----------
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="系统提示文本（server 端会自动包裹特殊 token 给到 duplex 模型）",
    )

    # ---------- 用户音频（path 与 base64 二选一） ----------
    user_audio_path: Optional[str] = Field(
        default=None,
        description="用户音频文件路径（server 可读，推荐）。支持 wav/mp3/m4a 等 soundfile 支持的格式。",
    )
    user_audio_base64: Optional[str] = Field(
        default=None,
        description="用户音频 base64（仅当不能共享文件系统时使用）。"
                    "格式：16kHz mono float32 raw bytes 或 wav 容器（自动检测）。",
    )

    # ---------- 视觉输入（可选，per-chunk） ----------
    image_paths: Optional[List[str]] = Field(
        default=None,
        description="图像帧文件路径列表，每个对应 1 个 chunk。仅 omni 双工场景需要。",
    )
    image_base64_list: Optional[List[str]] = Field(
        default=None,
        description="图像帧 base64 列表，每个对应 1 个 chunk（与 image_paths 二选一）。",
    )
    max_slice_nums: int = Field(
        default=1,
        ge=1,
        description="HD 图像切片数；1 表示不切片",
    )

    # ---------- TTS 音色 ----------
    ref_audio_path: Optional[str] = Field(
        default=None,
        description="参考音频路径（决定 TTS 音色）。不填则按 system_prompt 语种自动选择默认音色。",
    )
    ref_audio_base64: Optional[str] = Field(
        default=None,
        description="参考音频 base64（与 ref_audio_path 二选一）",
    )
    prompt_wav_path: Optional[str] = Field(
        default=None,
        description="TTS prompt 音频路径，不填走 ref_audio_path",
    )

    # ---------- 双工模型参数 ----------
    config: DuplexConfig = Field(
        default_factory=DuplexConfig,
        description="双工模型参数（force_listen_count / sampling 等），见 DuplexConfig",
    )

    # ---------- 批处理控制 ----------
    stop_on_end_of_turn: bool = Field(
        default=False,
        description="True：当模型返回 end_of_turn=True 时停止（适合短交互评测）；"
                    "False（默认）：跑完所有音频 chunk（适合长视频/长音频）",
    )
    max_chunks: Optional[int] = Field(
        default=None,
        ge=1,
        description="最多跑多少个 chunk（兜底，None=不限）",
    )
    leading_silence_ms: int = Field(
        default=0,
        ge=0,
        description="在用户音频前插入多少毫秒静音（覆盖 force_listen_count 之外的启动期），可帮助稳定双工状态机",
    )

    # ---------- 输出选项 ----------
    return_per_chunk_audio: bool = Field(
        default=True,
        description="True（默认）：每个 chunk 的 audio_data 都返回；False：仅返回汇总 merged_audio_data",
    )
    return_merged_audio: bool = Field(
        default=True,
        description="True（默认）：把所有 speak chunk 的 24kHz 音频拼成一整段返回",
    )
    include_text_timeline: bool = Field(
        default=True,
        description="True（默认）：返回每个 chunk 的文本时间线（chunks 数组）；False：仅返回 full_text",
    )

    # ---------- client 追踪 ----------
    request_id: Optional[str] = Field(
        default=None,
        description="client 自定义 request id（用于日志关联）",
    )


# =============================================================================
# 输出
# =============================================================================

class DuplexBatchResponse(BaseModel):
    """非流式双工批量推理响应

    一次请求一次性返回所有结果，包含：

    - ``full_text``: 模型回复的完整文本（所有 speak chunk 文本拼接）
    - ``chunks``: 时间线（仅当 ``include_text_timeline=True``）
    - ``merged_audio_data``: 拼接的完整音频（仅当 ``return_merged_audio=True``）
    - 各种统计：``total_chunks`` / ``audio_duration_s`` / ``total_duration_ms``
    """

    # ---------- 状态 ----------
    success: bool = Field(..., description="是否成功")
    error: Optional[str] = Field(default=None, description="失败原因")

    # ---------- 文本 ----------
    full_text: str = Field(default="", description="模型回复的完整文本")
    chunks: List[DuplexChunkResult] = Field(
        default_factory=list,
        description="每个 chunk 的详细时间线（include_text_timeline=False 时为空）",
    )

    # ---------- 音频 ----------
    merged_audio_data: Optional[str] = Field(
        default=None,
        description="拼接的完整 24kHz mono float32 base64 音频"
                    "（仅 return_merged_audio=True）",
    )
    merged_audio_sample_rate: Optional[int] = Field(
        default=None, description="merged_audio_data 的采样率（24000）",
    )

    # ---------- 统计 ----------
    total_chunks: int = Field(default=0, description="实际跑了多少个 chunk")
    speak_chunks: int = Field(default=0, description="其中 speak（产生音频/文本）的 chunk 数")
    listen_chunks: int = Field(default=0, description="其中 listen 的 chunk 数")
    audio_duration_s: float = Field(default=0.0, description="输出音频累计时长（秒）")
    total_duration_ms: float = Field(default=0.0, description="端到端总耗时（毫秒）")
    stopped_reason: str = Field(
        default="audio_exhausted",
        description="停止原因：audio_exhausted / end_of_turn / max_chunks / error",
    )

    # ---------- 追踪 ----------
    request_id: Optional[str] = Field(default=None, description="echo 自请求的 request_id")
    ticket_id: Optional[str] = Field(default=None, description="server FIFO 队列 ticket id")
    queue_wait_ms: float = Field(default=0.0, description="入队到分配 worker 的等待时间")
    worker_id: Optional[str] = Field(default=None, description="实际处理此请求的 worker id")

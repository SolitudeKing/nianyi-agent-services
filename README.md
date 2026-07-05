# NianYi Agent Services

`services` 是给个人 Agent 类项目复用的一组服务适配器，当前包含：

- `image`: OpenAI / MiniMax 图片生成适配器
- `tts`: MiniMax 语音合成、音色复刻、音色管理适配器

模块只依赖 Python 标准库，运行时不绑定具体业务项目，也不读取全局配置。API key、endpoint 和 timeout 都由调用方显式传入。

## 作为 Git Submodule 使用

在目标 Agent 项目中引入：

```powershell
git submodule add <services-repo-url> services
git submodule update --init --recursive
python -m pip install -e ./services
```

如果目标项目直接从项目根目录运行，并且 submodule 路径保持为 `services`，也可以直接导入：

```python
from services import ImageRequest, ImageService
```

推荐仍然执行 editable install，这样 IDE、类型提示和测试环境都会更稳定。

更新 submodule：

```powershell
git submodule update --remote services
```

## 使用示例

图片生成：

```python
from services import ImageRequest, ImageService

image_service = ImageService.fromProvider(
    "openai",
    api_key="sk-...",
)

response = image_service.generate(
    ImageRequest(
        model="gpt-image-1.5",
        prompt="a calm workspace for building personal agents",
        aspect_ratio="1:1",
    )
)
```

语音合成：

```python
from services import SpeechRequest, SpeechService

speech_service = SpeechService.fromProvider(
    "minimax",
    api_key="...",
)

response = speech_service.synthesize(
    SpeechRequest(
        model="speech-02-hd",
        text="你好，我是你的个人开发 Agent。",
        voice_id="male-qn-qingse",
    )
)
```

## 维护边界

- 公共能力放在 `image/`、`tts/` 及顶层导出中。
- 具体业务项目的 prompt、配置读取、数据库、缓存、任务编排不要放进这个模块。
- 新增 provider 时，优先实现对应的 `GenerateImage` 或 `GenerateSpeech` 子类，再在 `ImageService` / `SpeechService` 中注册。
- `py.typed` 已保留，调用方可以获得类型提示。

## 本地检查

```powershell
python -m pytest
python -m ruff check .
```

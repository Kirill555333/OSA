# ARCHITECTURE — OSA

Сжатая карта архитектуры. Актуально на 2026-10-20 (после Блока 4 «Модели и интеллект»).

## Слои

```text
Пользователь (текст / голос / GUI)
                │
       ┌────────┴────────┐
       │                 │
   main.py           gui/window.py
   (терминал)        (PySide6 HUD)
       │                 │
       └────── InputHub ─┘
                │
             Agent
                │
   ┌────────────┼─────────────┐
   │            │             │
Context      Model         ToolRegistry
Manager    Interface           │
   │            │             │
   │      ┌─────┴─────┐       │
   │  LlamaCpp    LlamaCpp    │
   │  (reasoning) (vision)    │
   │            │             │
   └────────────┼─────────────┘
                │
       ActionSafetyPipeline
                │
           ActionRouter
                │
              Tools

Models

Модельный слой (src/osa/models/) устроен так:

    ModelInterface (interface.py) — контракт: generate(ModelRequest) -> ModelResponse, generate_stream, health_check.

    ModelRequest содержит messages, temperature, max_tokens, tools, tool_choice, metadata.

    tool_choice — "auto" (по умолчанию), "required", "none". Управляется через env OSA_TOOL_CHOICE или напрямую при создании Agent.

    LlamaCppModel (local.py) — OpenAI-совместимый клиент к llama-server. Парсер fail-closed: tool_calls принимаются только для предложенных инструментов.

URL сервера

Приоритет:

    LlamaCppConfig(base_url=...) — явный аргумент.

    OSA_LLAMA_URL env.

    http://127.0.0.1:8080 — дефолт.

Конфигурация моделей

config/default.yaml, секция models::
yaml

models:
  reasoning:
    base_url: "http://127.0.0.1:8081"
    model_name: "qwen3-8b"
    timeout: 45.0
  vision:
    base_url: "http://127.0.0.1:8080"
    model_name: "qwen2.5-vl-3b"
    timeout: 60.0

    reasoning — модель для agent.chat. Все инструменты, кроме inspect_screen.

    vision — модель для inspect_screen. Должна поддерживать --mmproj.

    router — опционально, зарезервировано под будущий мульти-модельный диспетчер.

Обе роли по умолчанию смотрят на один URL. osa --check пробует обе и печатает статус.
Системный промпт

Лежит в src/osa/prompts/system_ru.txt. Переопределяется через OSA_SYSTEM_PROMPT=/path/to/file.txt. Fallback — короткая встроенная строка.
Две модели на Windows

RX 6600 (8 ГБ VRAM) не тянет 8B + 3B одновременно. Рабочая схема — два llama-server процесса на разных портах, загружаемых по требованию:
bash

# reasoning
llama-server -m models/qwen3-8b.gguf --port 8081 --jinja -c 8192

# vision (отдельный процесс)
llama-server -m models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  --mmproj models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
  --port 8080 --jinja -c 8192

config/default.yaml направляет роли на эти порты. Если reasoning не запущен — osa работает на vision (как сейчас), с предупреждением.
Единый источник версии

src/osa/version.py::__version__ — единственный. pyproject.toml читает через dynamic = ["version"]. main.py импортирует, не дублирует.
Пути

    OSA_HOME (по умолчанию ~/.osa) — данные: память, настройки, логи, скриншоты.

    OSA_WORKSPACE (по умолчанию cwd) — разрешённые файлы для SafeFilesystem/SafeShell.

Невалидные значения env дают понятную ошибку, не тихий fallback.
Безопасность

    ActionRequest → ActionSafetyPipeline: режим → разрешения → безопасность → подтверждение → ActionRouter.

    Recovery не обходит safety.

    Observability не хранит сырые аргументы, секреты маскируются.

    close_application, write_file, patch_file, terminal_execute — CONFIRM.

    Ввод в Terminal / iTerm / VS Code — подтверждение через DesktopToolSafetyPolicy.

Тесты

    pytest -q — ~2100 passed, 9 skipped.

    Пропущены: живой звук (OSA_RUN_LIVE_AUDIO=1 включает), Playwright Chromium, Windows UI Automation, платформенные guard'ы.

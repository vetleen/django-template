# Vetle's Django Template

## Setup

### Prerequisites
- Python 3.8+ (Python 3.14+ recommended)
- Node.js and npm (for Tailwind CSS and Flowbite)
- Redis (required for Django Channels/WebSocket support)

### Installation Steps

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies include:
   - Django 6.0.1
   - Django Channels (for WebSocket support)
   - channels-redis (Redis backend for Channels)
   - daphne (ASGI server)
   - openai (OpenAI API client)
   - tiktoken (token counting)

3. **Install Node dependencies:**
   ```bash
   npm install
   ```
   
   Includes Tailwind CSS and Flowbite UI components.

4. **Set up Redis:**
   - **Windows (WSL)**: 
     ```bash
     # In WSL terminal:
     sudo apt-get update
     sudo apt-get install redis-server
     redis-server
     ```
   - **Windows (Native)**: Download from [Redis for Windows](https://github.com/microsoftarchive/redis/releases) or use WSL (recommended)
   - **Linux**: `sudo apt-get install redis-server` (Ubuntu/Debian) or `brew install redis` (Mac)
   - **Docker**: `docker run -d -p 6379:6379 redis:latest`
   
   Redis runs on port 6379 by default. Ensure Redis is running before starting the Django server.

5. **Apply database migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Build Tailwind CSS (required for styling):**
   ```bash
   npx @tailwindcss/cli -i ./static/src/input.css -o ./static/src/output.css --watch
   ```
   
   Keep this running in a separate terminal while developing. The `--watch` flag automatically rebuilds CSS when you make changes.

7. **Configure environment variables:**
   Create a `.env` file in the project root (see [Environment variables](#environment-variables) section below).

### Running the Server

**For WebSocket support (required for chat application):**
```bash
daphne -b 127.0.0.1 -p 8000 config.asgi:application
```

**Alternative: Django 6.0 runserver (also supports WebSockets):**
```bash
python manage.py runserver 8000
```

Note: The chat application requires WebSocket support, so use one of the above methods. The `daphne` server is recommended for production-like environments.

### Heroku deployment

**Checklist before first deploy:**

1. **Node.js buildpack** (required for Tailwind in release):
   ```bash
   heroku buildpacks:add --index 1 heroku/nodejs
   ```
2. **Config vars:** Set `DJANGO_SECRET_KEY` (required when `DEBUG=False`). Set `DJANGO_CSRF_TRUSTED_ORIGINS` to your app URL (e.g. `https://secure-atoll-88335-7f808e9c7ae1.herokuapp.com`) so form POSTs (login, etc.) work over HTTPS. Optionally set `DJANGO_ALLOWED_HOSTS`; default includes `.herokuapp.com`. Add Postgres and Redis addons if not already attached; Heroku sets `DATABASE_URL` and `REDIS_URL` automatically.
3. **Python version:** Pinned in `.python-version` (Heroku’s Python buildpack uses this; `runtime.txt` is deprecated).

Tailwind CSS is built during the **Node build phase** (`npm run build` → `build:css`), so `static/src/output.css` is in the slug before the web dyno runs. Release runs: migrate → collectstatic. Compression is on-the-fly (`COMPRESS_OFFLINE=False`).

**Viewing app logs:** `heroku logs --tail` mixes all addons (e.g. heroku-redis). To see web dyno and release output:
```bash
heroku logs --tail --source app -a YOUR_APP_NAME
```
Or check release result: `heroku releases -a YOUR_APP_NAME` (failed release = deploy did not go live).

### Running Tests

```bash
# Account tests (auth, signup, password reset, email verification)
python manage.py test accounts

# Chat application tests (comprehensive, includes WebSocket tests)
python manage.py test llm_chat

# LLM service tests (requires OPENAI_API_KEY)
TEST_APIS=True python manage.py test llm_service
```

## Admin & superuser (dev)
- Admin UI: `/admin/`
- A dev superuser is auto-created on runserver when `DJANGO_USER_NAME` and `DJANGO_PASSWORD` are set.

## Environment variables

Create a `.env` file in the project root:

```env
# Django Settings
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
# Production (e.g. Heroku): set to app URL so CSRF works for form POSTs over HTTPS
# DJANGO_CSRF_TRUSTED_ORIGINS=https://your-app.herokuapp.com
DJANGO_PASSWORD_RESET_TIMEOUT=3600

# APIs
OPENAI_API_KEY=sk-...  # Required for LLM functionality
TEST_APIS=False  # Set to True to run LLM service tests

# Redis (for Django Channels/WebSocket support)
REDIS_URL=redis://127.0.0.1:6379/0  # Optional, defaults to localhost:6379

# Superuser (dev-only, auto-created on runserver)
DJANGO_USER_NAME=you@example.com
DJANGO_PASSWORD=your-password

# Email (dev default prints to terminal)
DJANGO_EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
# DEFAULT_FROM_EMAIL=noreply@example.com   # Set in production for verification/password-reset emails

# Email verification (optional)
EMAIL_VERIFICATION_REQUIRED=True   # Set False to skip verification (e.g. dev)
EMAIL_VERIFICATION_TIMEOUT=86400   # Token validity in seconds (default: 24h)
```

`python-dotenv` loads `.env` in `config/settings.py`.

**Note:** The `REDIS_URL` is optional and defaults to `redis://127.0.0.1:6379/0` if not set. For production or custom Redis setups, specify the full Redis URL.

## Tailwind + Flowbite
- Input: `static/src/input.css`
- Output: `static/src/output.css`
- Build/watch:
```
npx @tailwindcss/cli -i ./static/src/input.css -o ./static/src/output.css --watch
```

Flowbite is installed via npm and loaded in `static/src/input.css` and `templates/_base.html`. Dark mode uses the class strategy: `@custom-variant dark` in `input.css` and a `.dark` class on `<html>`; templates use `dark:` variants for dark-mode styling.

## Django Compressor
Compressor is enabled in `config/settings.py` and used in `templates/_base.html` to load the CSS bundle.

## Auth flows
- Login: `/accounts/login/`
- Signup: `/accounts/signup/`
- Delete account: `/accounts/delete/`
- Password change: `/accounts/password_change/`
- Password reset: `/accounts/password_reset/`
- Email verification: `/accounts/verify-email/sent/`, `/accounts/verify-email/<token>/`, `/accounts/verify-email/resend/`, `/accounts/verify-required/` (see [Email verification](#email-verification)).

## Email verification

New signups must verify their email before they can log in (when `EMAIL_VERIFICATION_REQUIRED` is True).

- **Flow:** Sign up → "Check your email" page → user clicks link in email → verified and logged in.
- **Token expiry:** 24 hours (configurable via `EMAIL_VERIFICATION_TIMEOUT`).
- **Resend:** "Resend verification email" on the check-your-email and verify-required pages. Rate limit: 1 minute after signup, then doubles each time (1 → 2 → 4 → 8 minutes); resets after 24 hours.
- **URLs:** `/accounts/verify-email/sent/`, `/accounts/verify-email/<token>/`, `/accounts/verify-email/resend/`, `/accounts/verify-required/`.
- **Templates:** `registration/verify_email_sent.html`, `registration/verify_email_error.html`, `registration/verify_required.html`; email body/subject: `registration/email_verification_*.txt`.
- **Disable:** Set `EMAIL_VERIFICATION_REQUIRED=False` in `.env` to allow login without verification (e.g. for local dev).
- **Production:** Verification links are built with `request.build_absolute_uri()`, so they use the current host and work with HTTPS. Set `DEFAULT_FROM_EMAIL` (e.g. `noreply@yourdomain.com`) for a proper sender; the same `DJANGO_EMAIL_BACKEND` used for password reset is used for verification emails.
- **Admin:** User list shows `email_verified`; `EmailVerificationToken` is registered so you can see pending tokens and expiry.

## User settings & dark mode

**UserSettings** (accounts app): OneToOne to User, created automatically when a user is created (signal). Stores per-user preferences; currently `theme` with choices `light` or `dark` (default `light`). Access via `user.settings.theme`.

**Dark mode** (Flowbite class-based):
- Toggle in the top nav for **logged-in users only**; theme is saved in `UserSettings` and persists across sessions.
- Anonymous users get theme from `localStorage` or system preference (no persistence).
- Implemented via `@custom-variant dark` in `static/src/input.css`, inline script in `<head>` to avoid FOUC, and context processor `accounts.context_processors.theme` that passes `theme` into templates.
- **Theme update:** `POST /accounts/settings/theme/` with `theme=light` or `theme=dark` (login required); returns JSON `{"theme": "…"}`.

## LLM Service

The `llm_service` app provides a centralized service for interacting with OpenAI's Responses API.

### Features
- Structured JSON output with schema validation
- Tool calling with automatic execution
- Streaming support with real-time event handling
- Usage tracking and cost calculation
- Comprehensive logging of all LLM calls

### Basic Usage

**Non-streaming (blocking):**
```python
from llm_service.services import LLMService
from django.contrib.auth import get_user_model

User = get_user_model()
service = LLMService()
user = User.objects.first()

json_schema = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
    },
    "required": ["message"],
    "additionalProperties": False,
}

call_log = service.call_llm(
    user_prompt="Say hello",
    json_schema=json_schema,
    schema_name="greeting",
    user=user,
    retries=2,  # Optional, default: 2
)

# Access parsed JSON
print(call_log.parsed_json["message"])
```

**Streaming (real-time events):**
```python
gen = service.call_llm_stream(
    user_prompt="Say hello",
    json_schema=json_schema,
    schema_name="greeting",
    user=user,
    retries=2,  # Optional, default: 2
)

for event_type, event in gen:
    if event_type == "response.output_text.delta":
        print(event.delta, end="")  # Print tokens as they arrive
    elif event_type == "final":
        call_log = event["call_log"]
        print(f"\nCompleted: {call_log.parsed_json}")
        break
```

### Tool Calling

Both methods support tool calling with automatic execution. Define tools and pass them:

```python
from llm_service.tools.secret_number import GET_SECRET_NUMBER_TOOL

call_log = service.call_llm(
    user_prompt="Get the secret number",
    json_schema=json_schema,
    schema_name="result",
    tools=[GET_SECRET_NUMBER_TOOL],
    user=user,
)
```

**Tool execution timing:**
- **Non-streaming (`call_llm`)**: Tools are executed after the full response is received
- **Streaming (`call_llm_stream`)**: Tools are executed immediately as their arguments are streamed (more efficient for multi-turn tool calls)

### Retry Logic

- **Non-streaming**: Retries on rate limits (exponential backoff), timeouts (incremental delay), and retryable errors
- **Streaming**: Each stream round retries independently, but only if the error occurs before any events are yielded (avoids duplicate events)

### Usage Tracking

All calls are automatically tracked in:
- `LLMCallLog`: Individual call logs with tokens, cost, and response data
- `UserMonthlyUsage`: Monthly aggregates per user

View in admin: `/admin/llm_service/`

### Testing

Run tests with `TEST_APIS=True`:
```bash
TEST_APIS=True python manage.py test llm_service
```

Tests are in:
- `llm_service/tests/test_llm_service.py` - Non-streaming tests
- `llm_service/tests/test_llm_service_stream.py` - Streaming tests

## Chat Application (`llm_chat`)

The `llm_chat` app provides a real-time chat interface with an LLM assistant, featuring WebSocket-based streaming, markdown rendering, and persistent chat history.

### Features

- **Real-time streaming**: WebSocket-based message streaming using Django Channels
- **Markdown rendering**: Full markdown support with syntax highlighting, code blocks, lists, headers, links, etc.
- **Chat history**: Persistent chat threads with automatic title generation
- **Token-aware context**: Smart chat history truncation using token counts
- **Multi-tab support**: Chat updates sync across multiple browser tabs
- **Connection management**: Automatic reconnection with connection status indicators
- **Accessibility**: Proper ARIA handling and keyboard navigation support

### Architecture

**Models:**
- `ChatThread`: Represents a conversation thread with a user, title, and timestamps
- `ChatMessage`: Individual messages with role (user/assistant), content, status, and token counts

**Services:**
- `ChatService`: Orchestrates LLM streaming, message persistence, and thread title generation
- `assemble_system_instruction()`: Builds system prompts with dynamic chat history
- `assemble_chat_history()`: Retrieves and formats chat history up to token limits

**WebSocket Consumer:**
- `ChatConsumer`: Handles WebSocket connections, message streaming, and group broadcasting
- Routes: `/ws/chat/<thread_id>/`
- Events: `response.output_text.delta`, `final`, `thread.title.updated`, `response.error`

**Views:**
- `chat_view`: Main chat interface (GET/POST)
- `chat_messages_json`: API endpoint for AJAX message loading
- URLs: `/chat/` (new chat), `/chat/<uuid>/` (specific thread)

### Usage

**Access the chat:**
- Navigate to `/chat/` for a new chat
- Navigate to `/chat/<thread_id>/` for an existing thread
- Requires authentication (login required)

**Sending messages:**
- Type in the textarea and press Enter to send
- Use Shift+Enter for multi-line input
- Textarea auto-resizes up to ~5 lines

**Markdown support:**
The chat supports full markdown rendering including:
- Headers (`#`, `##`, `###`, etc.)
- **Bold** (`**text**`), *italic* (`*text*`), ~~strikethrough~~ (`~~text~~`)
- Code blocks (```python ... ```) and inline code (`` `code` ``)
- Lists (ordered and unordered)
- Blockquotes (`> quote`)
- Links (`[text](url)`)
- Tables
- Horizontal rules (`---`)
- Emoji shortcodes (`:sparkles:`, `:tada:`, etc.)

### System Instructions

The system automatically assembles prompts with:
- Dynamic chat history (up to 20,000 tokens by default)
- Formatting instructions encouraging markdown usage
- Context-aware conversation continuation

See `llm_chat/system_instructions.py` for customization.

### Testing

Run chat tests:
```bash
python manage.py test llm_chat
```

Test files:
- `llm_chat/tests/test_models.py` - Model tests
- `llm_chat/tests/test_services.py` - Service layer tests
- `llm_chat/tests/test_consumers.py` - WebSocket consumer tests
- `llm_chat/tests/test_views.py` - View tests
- `llm_chat/tests/test_connection.py` - WebSocket connection tests
- `llm_chat/tests/test_integration.py` - End-to-end integration tests
- `llm_chat/tests/test_markdown.py` - Markdown rendering tests

### Frontend Features

**Markdown Rendering:**
- Uses `marked.js` for parsing and `DOMPurify` for sanitization
- CSS styling scoped to `#chat-messages .markdown-content`
- Streaming markdown re-parses when complete blocks are detected
- Server-rendered messages are processed on page load

**WebSocket Management:**
- Automatic connection when thread is active
- Reconnection with exponential backoff (max 5 attempts)
- Connection status: green dot when connected (no label), spinner and "Connecting..." when connecting, text when reconnecting or disconnected
- Heartbeat ping/pong for connection health

**UI Features:**
- Auto-scrolling when messages arrive
- Sidebar navigation with thread history
- Dynamic thread ordering (most recent at top)
- Form disabling during streaming
- Rate limiting (1 second between messages)

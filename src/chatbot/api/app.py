"""Synchronous JSON FastAPI application."""

import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from chatbot.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MemoryResponse,
    TokenRequest,
    TokenResponse,
)
from chatbot.auth.service import InvalidTokenError
from chatbot.config import Settings
from chatbot.graph.service import SAFE_FAILURE_MESSAGE, InvalidProcessRequest
from chatbot.services.container import AppContainer, build_container
from chatbot.services.logging import (
    configure_logging,
    get_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)

logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def create_app(
    *,
    settings: Settings | None = None,
    container: AppContainer | None = None,
) -> FastAPI:
    """Create an app with optional injected services for offline tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime_settings = container.settings if container is not None else settings or Settings()
        configure_logging(runtime_settings.log_level)
        runtime_container = container or build_container(runtime_settings)
        application.state.container = runtime_container
        try:
            yield
        finally:
            if container is None:
                runtime_container.close()

    application = FastAPI(title="Chatbot Multi-Node Template", lifespan=lifespan)

    @application.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        incoming = request.headers.get("X-Request-ID", "")
        correlation_id = incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid4().hex
        request.state.correlation_id = correlation_id
        token = set_correlation_id(correlation_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = correlation_id
            return response
        finally:
            logger.info(
                "request_completed",
                extra={
                    "event": "request_completed",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1_000, 2),
                },
            )
            reset_correlation_id(token)

    @application.get("/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/health/ready", response_model=HealthResponse)
    def ready(request: Request) -> HealthResponse:
        try:
            _container(request).memories.ping()
        except Exception as error:
            logger.warning(
                "readiness_failed",
                extra={"event": "readiness_failed", "error_type": type(error).__name__},
            )
            raise HTTPException(status_code=503, detail="Service is not ready") from None
        return HealthResponse(status="ready")

    @application.post("/auth/token", response_model=TokenResponse)
    def issue_token(payload: TokenRequest, request: Request) -> TokenResponse:
        auth = _container(request).auth
        if not auth.authenticate(payload.username, payload.password.get_secret_value()):
            logger.warning("auth_failed", extra={"event": "auth_failed"})
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = auth.issue_access_token(payload.username)
        return TokenResponse(access_token=token, expires_in=auth.expires_in_seconds)

    @application.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        request: Request,
        subject: str = Depends(_current_subject),
    ) -> ChatResponse:
        try:
            result = _container(request).chatbot.chat(
                subject=subject,
                session_id=payload.session_id,
                message=payload.message,
                memory_consent=payload.memory_consent,
                correlation_id=get_correlation_id(),
                process_action=payload.process_action,
                process_name=payload.process_name,
            )
        except InvalidProcessRequest as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except Exception as error:
            logger.warning(
                "chat_failed",
                extra={"event": "chat_failed", "error_type": type(error).__name__},
            )
            raise HTTPException(status_code=503, detail=SAFE_FAILURE_MESSAGE) from None
        return ChatResponse(
            session_id=payload.session_id,
            response=result.response,
            route=result.route,
            correlation_id=get_correlation_id(),
            processes=[item.model_dump() for item in result.processes],
        )

    @application.get("/memories", response_model=list[MemoryResponse])
    def list_memories(
        request: Request,
        subject: str = Depends(_current_subject),
    ) -> list[MemoryResponse]:
        return [
            MemoryResponse.model_validate(record.model_dump())
            for record in _container(request).memories.list_for_subject(subject)
        ]

    @application.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_memory(
        memory_id: UUID,
        request: Request,
        subject: str = Depends(_current_subject),
    ) -> Response:
        deleted = _container(request).memories.delete(subject, str(memory_id))
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return application


def _container(request: Request) -> AppContainer:
    return request.app.state.container


def _current_subject(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return _container(request).auth.verify_access_token(credentials.credentials)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Not authenticated") from None


app = create_app()

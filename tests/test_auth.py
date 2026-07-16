from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
import pytest

from chatbot.auth.service import AuthService, InvalidTokenError


def make_service(*, now: datetime | None = None) -> AuthService:
    password_hash = bcrypt.hashpw(b"correct-password", bcrypt.gensalt()).decode()
    return AuthService(
        username="admin",
        password_hash=password_hash,
        jwt_secret="a-secure-test-secret-that-is-long-enough-for-all-hmac-tests-1234",
        token_ttl=timedelta(minutes=30),
        clock=lambda: now or datetime.now(UTC),
    )


def test_fixed_credentials_issue_a_subject_scoped_access_token() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service = make_service(now=now)

    assert service.authenticate("admin", "correct-password") is True
    assert service.authenticate("admin", "wrong-password") is False
    assert service.authenticate("someone-else", "correct-password") is False

    token = service.issue_access_token("admin")

    assert service.verify_access_token(token) == "admin"


def test_access_token_rejects_expiry_and_unexpected_algorithm() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service = make_service(now=now)
    expired = jwt.encode(
        {"sub": "admin", "iat": now - timedelta(hours=1), "exp": now - timedelta(minutes=1)},
        service.jwt_secret,
        algorithm="HS256",
    )
    wrong_algorithm = jwt.encode(
        {"sub": "admin", "iat": now, "exp": now + timedelta(minutes=30)},
        service.jwt_secret,
        algorithm="HS384",
    )

    with pytest.raises(InvalidTokenError):
        service.verify_access_token(expired)
    with pytest.raises(InvalidTokenError):
        service.verify_access_token(wrong_algorithm)


def test_auth_service_rejects_weak_configuration() -> None:
    password_hash = bcrypt.hashpw(b"password", bcrypt.gensalt()).decode()

    with pytest.raises(ValueError, match="JWT secret"):
        AuthService(
            username="admin",
            password_hash=password_hash,
            jwt_secret="too-short",
        )

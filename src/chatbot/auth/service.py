"""Fixed-credential authentication and short-lived JWT access tokens."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hmac import compare_digest
from typing import Any

import bcrypt
import jwt


class InvalidTokenError(ValueError):
    """Raised when an access token cannot be trusted."""


class AuthService:
    """Authenticate one configured operator and manage its access tokens."""

    _algorithm = "HS256"

    def __init__(
        self,
        *,
        username: str,
        password_hash: str,
        jwt_secret: str,
        token_ttl: timedelta = timedelta(minutes=30),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not username.strip():
            raise ValueError("Auth username is required")
        if not password_hash.startswith(("$2a$", "$2b$", "$2y$")):
            raise ValueError("Auth password must be a bcrypt hash")
        if len(jwt_secret.encode()) < 32:
            raise ValueError("JWT secret must be at least 32 bytes")
        if token_ttl <= timedelta(0):
            raise ValueError("Token TTL must be positive")

        self._username = username
        self._password_hash = password_hash.encode()
        self._jwt_secret = jwt_secret
        self._token_ttl = token_ttl
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def jwt_secret(self) -> str:
        """Expose the signing secret only for boundary-level token verification tests."""

        return self._jwt_secret

    @property
    def expires_in_seconds(self) -> int:
        return int(self._token_ttl.total_seconds())

    def authenticate(self, username: str, password: str) -> bool:
        """Return whether both configured credentials match."""

        username_matches = compare_digest(username.encode(), self._username.encode())
        try:
            password_matches = bcrypt.checkpw(password.encode(), self._password_hash)
        except (TypeError, ValueError):
            password_matches = False
        return username_matches and password_matches

    def issue_access_token(self, subject: str) -> str:
        """Issue a signed access token for the configured subject."""

        if not compare_digest(subject, self._username):
            raise ValueError("Unknown token subject")
        now = self._utc_now()
        payload = {
            "sub": subject,
            "iat": now,
            "exp": now + self._token_ttl,
        }
        return jwt.encode(payload, self._jwt_secret, algorithm=self._algorithm)

    def verify_access_token(self, token: str) -> str:
        """Verify signature, algorithm, required claims, expiry, and subject."""

        try:
            payload: dict[str, Any] = jwt.decode(
                token,
                self._jwt_secret,
                algorithms=[self._algorithm],
                options={
                    "require": ["sub", "iat", "exp"],
                    "verify_exp": False,
                    "verify_iat": False,
                },
            )
            subject = payload["sub"]
            issued_at = float(payload["iat"])
            expires_at = float(payload["exp"])
            now = self._utc_now().timestamp()
            if not isinstance(subject, str) or not compare_digest(subject, self._username):
                raise InvalidTokenError
            if issued_at > now + 60 or expires_at <= now:
                raise InvalidTokenError
        except (jwt.PyJWTError, KeyError, TypeError, ValueError, InvalidTokenError):
            raise InvalidTokenError("Invalid or expired access token") from None
        return subject

    def _utc_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("Auth clock must return a timezone-aware datetime")
        return now.astimezone(UTC)

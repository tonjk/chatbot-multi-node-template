import logging

from chatbot.services.logging import JsonFormatter, reset_correlation_id, set_correlation_id


def test_json_logs_include_correlation_but_exclude_unapproved_sensitive_fields() -> None:
    record = logging.LogRecord(
        name="chatbot.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="auth_failed",
        args=(),
        exc_info=None,
    )
    record.event = "auth_failed"
    record.password = "sentinel-password"
    record.authorization = "Bearer sentinel-token"
    token = set_correlation_id("request-123")
    try:
        output = JsonFormatter().format(record)
    finally:
        reset_correlation_id(token)

    assert '"correlation_id":"request-123"' in output
    assert "sentinel-password" not in output
    assert "sentinel-token" not in output

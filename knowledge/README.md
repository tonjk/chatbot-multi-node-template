# Shared Chatbot Knowledge

This directory contains application-owned Markdown that can be searched by the chatbot.

## Memory behavior

Short-term conversation history is stored in a LangGraph checkpoint scoped to the authenticated subject and session. Durable facts and preferences are written only when the current chat request includes explicit memory consent. Users can list or delete their saved durable memories through authenticated API endpoints.

## Authentication behavior

The first release uses one configured operator username and bcrypt password hash. Successful login returns an HS256 access token that expires after 30 minutes. This fixed credential is intended for a local template or single trusted operator, not a multi-user production service.

## Available tools

The assistant can use a bounded arithmetic calculator, return the current time for a valid IANA timezone, or search this shared knowledge base. Other tools are unavailable.


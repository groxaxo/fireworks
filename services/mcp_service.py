"""Model Context Protocol (MCP) service implementation."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class MCPMessage:
    """MCP message structure."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MCPContext:
    """MCP context structure."""
    messages: List[MCPMessage]
    system_prompt: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = None

class MCPService:
    """
    Model Context Protocol service for standardized LLM interactions.
    
    MCP provides a standardized way to format and manage context
    for different LLM providers (Fireworks, DeepInfra, etc.).
    """
    
    def __init__(self):
        """Initialize MCP service."""
        self.contexts = {}
    
    def create_context(
        self,
        context_id: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> MCPContext:
        """Create a new MCP context."""
        context = MCPContext(
            messages=[],
            system_prompt=system_prompt,
            **kwargs
        )
        self.contexts[context_id] = context
        return context
    
    def add_message(
        self,
        context_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to a context."""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} not found")
        
        message = MCPMessage(role=role, content=content, metadata=metadata)
        self.contexts[context_id].messages.append(message)
    
    def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Get a context by ID."""
        return self.contexts.get(context_id)
    
    def format_for_fireworks(self, context_id: str) -> Dict[str, Any]:
        """Format context for Fireworks API."""
        context = self.contexts.get(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")
        
        messages = []
        
        # Add system prompt if exists
        if context.system_prompt:
            messages.append({
                "role": "system",
                "content": context.system_prompt
            })
        
        # Add conversation messages
        for msg in context.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return {
            "messages": messages,
            "max_tokens": context.max_tokens,
            "temperature": context.temperature
        }
    
    def format_for_deepinfra(self, context_id: str) -> Dict[str, Any]:
        """Format context for DeepInfra API."""
        # DeepInfra uses OpenAI-compatible format
        return self.format_for_fireworks(context_id)
    
    def update_context_params(
        self,
        context_id: str,
        **kwargs
    ):
        """Update context parameters."""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} not found")
        
        context = self.contexts[context_id]
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
    
    def clear_messages(self, context_id: str):
        """Clear all messages from a context."""
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} not found")
        
        self.contexts[context_id].messages = []
    
    def delete_context(self, context_id: str):
        """Delete a context."""
        if context_id in self.contexts:
            del self.contexts[context_id]
    
    def export_context(self, context_id: str) -> str:
        """Export context as JSON."""
        context = self.contexts.get(context_id)
        if not context:
            raise ValueError(f"Context {context_id} not found")
        
        return json.dumps({
            "system_prompt": context.system_prompt,
            "messages": [asdict(msg) for msg in context.messages],
            "max_tokens": context.max_tokens,
            "temperature": context.temperature,
            "metadata": context.metadata
        }, indent=2)
    
    def import_context(self, context_id: str, json_data: str):
        """Import context from JSON."""
        data = json.loads(json_data)
        
        context = MCPContext(
            messages=[
                MCPMessage(**msg) for msg in data.get("messages", [])
            ],
            system_prompt=data.get("system_prompt"),
            max_tokens=data.get("max_tokens", 2048),
            temperature=data.get("temperature", 0.7),
            metadata=data.get("metadata")
        )
        
        self.contexts[context_id] = context
        return context

"""LLM service wrapper for OpenAI API."""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service wrapper for OpenAI API interactions."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """Initialize the LLM service.
        
        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
            temperature: Temperature for generation (defaults to settings)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """Generate a text response from the LLM.
        
        Args:
            system_prompt: System message for context
            user_prompt: User message/query
            temperature: Override default temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Generate a structured JSON response from the LLM.
        
        Args:
            system_prompt: System message for context
            user_prompt: User message/query
            response_schema: JSON schema for the expected response
            temperature: Override default temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Parsed JSON response as dictionary
        """
        # Add JSON instruction to system prompt
        enhanced_system = f"""{system_prompt}

You must respond with a valid JSON object matching this schema:
{json.dumps(response_schema, indent=2)}

Respond ONLY with the JSON object, no additional text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
        except Exception as e:
            logger.error(f"LLM structured generation failed: {e}")
            raise
    
    def generate_list(
        self,
        system_prompt: str,
        user_prompt: str,
        item_description: str = "item",
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> List[str]:
        """Generate a list of items from the LLM.
        
        Args:
            system_prompt: System message for context
            user_prompt: User message/query
            item_description: Description of list items for schema
            temperature: Override default temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            List of generated items
        """
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"List of {item_description}s"
                }
            },
            "required": ["items"]
        }
        
        result = self.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=schema,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return result.get("items", [])
    
    async def agenerate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """Async version of generate.
        
        Args:
            system_prompt: System message for context
            user_prompt: User message/query
            temperature: Override default temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text response
        """
        from openai import AsyncOpenAI
        
        async_client = AsyncOpenAI(api_key=self.api_key)
        
        try:
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Async LLM generation failed: {e}")
            raise

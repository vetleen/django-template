"""
Tool registry for managing and executing LLM tools.
"""
from typing import Dict, Callable, Any, Optional
import json


class ToolRegistry:
    """Registry for managing and executing LLM tools."""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable) -> None:
        """Register a tool function."""
        self.tools[name] = func
    
    def execute(self, name: str, arguments: str) -> Optional[str]:
        """
        Execute a tool with the given arguments.
        
        Args:
            name: Name of the tool to execute
            arguments: JSON string of arguments
            
        Returns:
            String result of tool execution, or None if execution failed
        """
        if name not in self.tools:
            print(f"Tool '{name}' not found in registry")
            return None
        
        try:
            # Parse arguments if provided
            args = json.loads(arguments) if arguments else {}
            
            # Execute the tool function
            result = self.tools[name](**args)
            
            # Convert result to string
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            else:
                return str(result)
                
        except Exception as e:
            print(f"Tool execution failed for '{name}': {e}")
            return None
    
    def get_available_tools(self) -> list:
        """Get list of available tool names."""
        return list(self.tools.keys())


# Global tool registry instance
tool_registry = ToolRegistry()

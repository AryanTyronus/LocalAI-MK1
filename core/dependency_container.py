"""
DependencyContainer - Wires all dependencies together.

This container creates and wires:
- Config
- ModelManager
- MemoryManager
- PromptBuilder
- ModeController
- DocumentManager
- ToolRegistry
- GenerationPipeline
- AIService

All dependencies are passed via constructor - no circular imports.
"""

from core.config import Config
from core.model_manager import ModelManager
from core.prompt_builder import PromptBuilder
from core.mode_controller import ModeController
from core.tool_registry import ToolRegistry
from core.intent_classifier import IntentClassifier
from core.generation_pipeline import GenerationPipeline
from memory.memory_manager import MemoryManager
from retrieval.document_manager import DocumentManager
from services.ai_service import AIService
from core.logger import logger


class DependencyContainer:
    """
    Dependency injection container.
    
    Creates and wires all services with proper dependency injection.
    No service instantiates other services internally - all are passed via constructor.
    """
    
    def __init__(self):
        logger.info("Initializing DependencyContainer...")
        
        # 1. Create Config (singleton)
        self._config = Config()
        
        # 2. Create ModelManager (singleton via get_instance)
        self._model_manager = ModelManager.get_instance()
        
        # 3. Create MemoryManager (4-layer memory)
        self._memory_manager = MemoryManager()
        
        # 4. Create PromptBuilder
        self._prompt_builder = PromptBuilder(self._config)
        
        # 5. Configure ModeController with Config
        ModeController.configure(self._config)
        self._mode_controller = ModeController
        
        # 6. Create DocumentManager
        self._document_manager = DocumentManager()
        
        # 7. Register default tools
        self._register_default_tools()
        
        # 8. Create GenerationPipeline with all dependencies
        self._generation_pipeline = GenerationPipeline(
            config=self._config,
            memory_manager=self._memory_manager,
            model_manager=self._model_manager,
            prompt_builder=self._prompt_builder,
            mode_controller=self._mode_controller,
            document_manager=self._document_manager,
            tool_registry=ToolRegistry
        )
        
        # 9. Create AIService with GenerationPipeline
        self._ai_service = AIService(
            pipeline=self._generation_pipeline,
            config=self._config
        )
        
        logger.info("DependencyContainer initialized successfully")
    
    def _register_default_tools(self):
        """Register default tools for the application."""
        
        # Example safe tool: echo
        def _echo(params):
            return {'echo': params}
        
        ToolRegistry.register_tool(
            'echo', 
            'Echo back parameters', 
            {'type': 'object'}, 
            _echo
        )
        
        # Example system-level tool (requires confirmation by default)
        def _open_app(params):
            app = params.get('app_name')
            return f"(simulated) opened {app}"
        
        ToolRegistry.register_tool(
            'open_app', 
            'Open an application on the system (simulated)', 
            {'app_name': 'string'}, 
            _open_app
        )
    
    def get_ai_service(self) -> AIService:
        """Get the AIService instance."""
        return self._ai_service
    
    def get_config(self) -> Config:
        """Get the Config instance."""
        return self._config
    
    def get_model_manager(self):
        """Get the ModelManager instance."""
        return self._model_manager
    
    def get_memory_manager(self) -> MemoryManager:
        """Get the MemoryManager instance."""
        return self._memory_manager
    
    def get_generation_pipeline(self) -> GenerationPipeline:
        """Get the GenerationPipeline instance."""
        return self._generation_pipeline
    
    def get_document_manager(self) -> DocumentManager:
        """Get the DocumentManager instance."""
        return self._document_manager


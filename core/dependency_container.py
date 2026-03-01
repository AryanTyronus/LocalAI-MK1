"""
DependencyContainer - Wires all dependencies together.

This container creates and wires:
- Config
- ModelManager
- ModelAdapter (via ModelAdapterFactory)
- TokenBudgetManager (Phase 2)
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
from core.model_adapter import ModelAdapterFactory
from core.token_budget import TokenBudgetManager
from core.prompt_builder import PromptBuilder
from core.mode_controller import ModeController
from core.tool_registry import ToolRegistry
from core.intent_classifier import IntentClassifier
from core.generation_pipeline import GenerationPipeline
from memory.memory_manager import MemoryManager
from retrieval.document_manager import DocumentManager
from services.ai_service import AIService
from core.logger import logger
from tools.python_executor import PythonExecutor
from tools.file_reader import FileReader
from tools.stock_fetcher import StockFetcher
from tools.news_fetcher import NewsFetcher
from tools.weather_fetcher import WeatherFetcher
from tools.indian_market_fetcher import IndianMarketFetcher
from tools.current_affairs_fetcher import CurrentAffairsFetcher


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
        
        # 3. Create ModelAdapter via Factory (Phase 1: Model Abstraction Layer)
        self._model_adapter = ModelAdapterFactory.create_adapter(
            adapter_type="local",
            model_manager=self._model_manager,
            config=self._config
        )
        logger.info(f"Model adapter created: {self._model_adapter.model_name}")
        
        # 4. Create TokenBudgetManager (Phase 2: Token Budget Enforcement)
        self._token_budget_manager = TokenBudgetManager(self._config)
        logger.info(f"TokenBudgetManager initialized: max={self._config.max_context_tokens} tokens")
        
        # 5. Create MemoryManager (4-layer memory)
        self._memory_manager = MemoryManager()
        
        # 6. Create PromptBuilder
        self._prompt_builder = PromptBuilder(self._config)
        
        # 7. Configure ModeController with Config
        ModeController.configure(self._config)
        self._mode_controller = ModeController
        
        # 8. Create DocumentManager
        self._document_manager = DocumentManager()
        
        # 9. Register default tools
        self._register_default_tools()
        
        # 10. Create GenerationPipeline with ModelAdapter and TokenBudgetManager
        self._generation_pipeline = GenerationPipeline(
            config=self._config,
            memory_manager=self._memory_manager,
            model_adapter=self._model_adapter,
            prompt_builder=self._prompt_builder,
            mode_controller=self._mode_controller,
            document_manager=self._document_manager,
            tool_registry=ToolRegistry,
            token_budget_manager=self._token_budget_manager
        )
        
        # 11. Create AIService with GenerationPipeline
        self._ai_service = AIService(
            pipeline=self._generation_pipeline,
            config=self._config
        )
        
        logger.info("DependencyContainer initialized successfully")
    
    def _register_default_tools(self):
        """Register default tools for the application."""

        python_executor = PythonExecutor(timeout_seconds=5, max_code_chars=2000)
        file_reader = FileReader()
        stock_fetcher = StockFetcher(timeout_seconds=5)
        news_fetcher = NewsFetcher(timeout_seconds=6, max_items=5)
        weather_fetcher = WeatherFetcher(timeout_seconds=6)
        indian_market_fetcher = IndianMarketFetcher(timeout_seconds=6)
        current_affairs_fetcher = CurrentAffairsFetcher(timeout_seconds=6)

        ToolRegistry.register_tool(
            'python_executor',
            'Execute bounded Python snippets in a short-lived subprocess',
            {'code': 'string'},
            python_executor.execute,
            category='diagnostics',
            human_name='Python Executor',
            example_usage='run python: print(2 + 2)'
        )

        ToolRegistry.register_tool(
            'file_reader',
            'Read a text file from the current workspace',
            {'filepath': 'string'},
            file_reader.execute,
            category='filesystem',
            human_name='File Reader',
            example_usage='read file structured_memory.json'
        )

        ToolRegistry.register_tool(
            'stock_fetcher',
            'Fetch delayed stock quote for a ticker symbol',
            {'symbol': 'string'},
            stock_fetcher.execute,
            category='markets',
            human_name='Stock Fetcher',
            example_usage='stock price for AAPL'
        )

        ToolRegistry.register_tool(
            'news_fetcher',
            'Fetch latest real-time news headlines by topic',
            {'topic': 'string', 'limit': 'integer'},
            news_fetcher.execute,
            category='news',
            human_name='News Fetcher',
            example_usage='latest news on AI'
        )

        ToolRegistry.register_tool(
            'weather_fetcher',
            'Fetch latest weather updates for a location',
            {'location': 'string'},
            weather_fetcher.execute,
            category='weather',
            human_name='Weather Fetcher',
            example_usage='weather in London'
        )

        ToolRegistry.register_tool(
            'indian_market_fetcher',
            'Fetch Indian stock market overview and NSE/BSE stock details',
            {'symbol': 'string'},
            indian_market_fetcher.execute,
            category='markets',
            human_name='Indian Market Fetcher',
            example_usage='NSE stock TCS'
        )

        ToolRegistry.register_tool(
            'current_affairs_fetcher',
            'Fetch live current-affairs facts',
            {'query': 'string'},
            current_affairs_fetcher.execute,
            category='news',
            human_name='Current Affairs Fetcher',
            example_usage='who is the president of the united states'
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
    
    def get_model_adapter(self):
        """Get the ModelAdapter instance (Phase 1)."""
        return self._model_adapter
    
    def get_token_budget_manager(self):
        """Get the TokenBudgetManager instance (Phase 2)."""
        return self._token_budget_manager
    
    def get_memory_manager(self) -> MemoryManager:
        """Get the MemoryManager instance."""
        return self._memory_manager
    
    def get_generation_pipeline(self) -> GenerationPipeline:
        """Get the GenerationPipeline instance."""
        return self._generation_pipeline
    
    def get_document_manager(self) -> DocumentManager:
        """Get the DocumentManager instance."""
        return self._document_manager

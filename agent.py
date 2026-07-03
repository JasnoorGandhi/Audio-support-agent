"""
LLM Agent with RAG Tool Integration

Customer support agent using LangChain ReAct agent with ChromaDB-backed RAG.
Uses Groq-hosted models via LangChain ChatGroq.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for LLM agents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_initialized = False
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def process_query(self, text: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass


class CustomerSupportAgent(BaseAgent):
    """
    Customer Support Agent using LangChain ReAct + ChromaDB RAG.

    Config keys:
      api_key     - Groq API key (or set GROQ_API_KEY env var)
      model       - Groq model id (default: llama-3.3-70b-versatile)
      temperature - Sampling temperature (default: 0.7)
      chroma_db_path - Path for ChromaDB persistence (default: ./data/chroma_db)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.collection = None
        self.chroma_client = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize LLM, knowledge base, tools, and ReAct agent."""
        logger.info("Initializing CustomerSupportAgent...")
        self._init_llm()
        await self._setup_knowledge_base()
        tools = await self._create_tools()
        await self._create_agent(tools)
        self.is_initialized = True
        logger.info("CustomerSupportAgent initialized successfully.")

    def _init_llm(self):
        """Initialize Groq as the language model backend."""
        import os
        from pathlib import Path
        from dotenv import load_dotenv
        from langchain_groq import ChatGroq

        _root = Path(__file__).resolve().parents[2]
        load_dotenv(_root / ".env", override=True)

        raw = (
            self.config.get("api_key")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_KEY")
        )
        api_key = (str(raw).strip() if raw else "") or None
        model = self.config.get("model", "llama-3.3-70b-versatile")
        temperature = float(self.config.get("temperature", 0.7))

        if not api_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in your .env file "
                "(https://console.groq.com/keys)."
            )

        self.llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=2,
        )
        logger.info("LLM initialized: Groq — %s", model)

    async def _setup_knowledge_base(self) -> None:
        """Set up ChromaDB with customer support documents."""
        try:
            import chromadb
            import os
            import hashlib

            db_path = self.config.get("chroma_db_path", "./data/chroma_db")
            os.makedirs(db_path, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(path=db_path)
            collection_name = "customer_support_kb"

            # Reuse existing collection if already populated
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                if self.collection.count() > 0:
                    logger.info(f"Knowledge base loaded: {self.collection.count()} documents.")
                    return
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Customer support knowledge base"},
                )

            # Ingest documents
            knowledge_documents = self._get_customer_support_documents()
            logger.info(f"Ingesting {len(knowledge_documents)} documents...")

            documents, metadatas, ids = [], [], []
            for i, doc_data in enumerate(knowledge_documents):
                doc_id = f"doc_{i}_{hashlib.md5(doc_data['content'].encode()).hexdigest()[:8]}"
                documents.append(doc_data["content"])
                metadatas.append({
                    "category": doc_data["category"],
                    "title": doc_data["title"],
                    "doc_id": doc_id,
                })
                ids.append(doc_id)

            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Knowledge base ready: {len(documents)} documents ingested.")

        except Exception as e:
            logger.error(f"Knowledge base setup failed: {e}")
            raise

    def _get_customer_support_documents(self) -> List[Dict[str, str]]:
        """Predefined customer support knowledge base (16 documents)."""
        return [
            {"title": "Return Policy Overview", "category": "returns",
             "content": "We offer a 30-day return policy for all products purchased from our store. Items must be in original condition with all tags and packaging intact. Returns are processed within 5-7 business days of receiving the returned item. Refunds are issued to the original payment method."},
            {"title": "Return Process Steps", "category": "returns",
             "content": "To initiate a return: 1) Log into your account and go to Order History, 2) Select the order and click 'Return Items', 3) Choose the items to return and reason, 4) Print the prepaid return label, 5) Pack items securely and attach the label, 6) Drop off at any UPS location or schedule pickup."},
            {"title": "Non-Returnable Items", "category": "returns",
             "content": "The following items cannot be returned: personalized or customized products, perishable goods, digital downloads, gift cards, intimate apparel, and items marked as final sale. Health and safety regulations prevent returns of opened cosmetics and personal care items."},
            {"title": "Shipping Methods and Times", "category": "shipping",
             "content": "We offer multiple shipping options: Standard shipping (5-7 business days, free on orders over $50), Express shipping (2-3 business days, $12.99), Next-day shipping (1 business day, $24.99). All orders placed before 2 PM EST ship the same day."},
            {"title": "International Shipping", "category": "shipping",
             "content": "We ship internationally to over 50 countries. International shipping takes 7-14 business days via DHL Express. Shipping costs vary by destination and are calculated at checkout. Customers are responsible for customs fees and import duties. Some restrictions apply to certain products and countries."},
            {"title": "Order Tracking", "category": "shipping",
             "content": "Once your order ships, you'll receive a tracking number via email. Track your package using the tracking number on our website or the carrier's website. You can also track orders by logging into your account and viewing Order History. Tracking updates may take 24 hours to appear."},
            {"title": "Contact Information", "category": "support",
             "content": "Customer support is available 24/7 via multiple channels: Phone: 1-800-HELP-NOW (1-800-435-7669), Email: support@company.com, Live chat on our website (available 6 AM - 12 AM EST), or submit a support ticket through your account dashboard."},
            {"title": "Response Times", "category": "support",
             "content": "Our support team response times: Live chat - immediate during business hours, Phone support - average wait time under 3 minutes, Email support - response within 4 hours during business days, Support tickets - response within 24 hours. Premium customers receive priority support with faster response times."},
            {"title": "Product Warranty", "category": "warranty",
             "content": "All products come with a manufacturer's warranty. Electronics have 1-year warranty covering defects and malfunctions. Apparel and accessories have 90-day warranty against material defects. Warranty claims require proof of purchase and must be initiated within the warranty period."},
            {"title": "Technical Support", "category": "technical",
             "content": "Free technical support is available for all electronic products. Our certified technicians provide assistance with setup, troubleshooting, and software issues. Technical support is available Monday-Friday 8 AM - 8 PM EST via phone or email. We also offer remote assistance for compatible devices."},
            {"title": "Account Management", "category": "account",
             "content": "Manage your account online: Update personal information and addresses, view order history and tracking, manage payment methods, set communication preferences, download invoices and receipts. Account changes may take up to 24 hours to reflect across all systems."},
            {"title": "Order Modifications", "category": "orders",
             "content": "Orders can be modified or canceled within 1 hour of placement if not yet processed. Contact customer service immediately to make changes. Once an order is processed and shipped, it cannot be modified. You can return unwanted items following our return policy."},
            {"title": "Payment Methods", "category": "payment",
             "content": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, and Buy Now Pay Later options through Klarna and Afterpay. Gift cards and store credit can also be used for purchases. Payment is processed securely using 256-bit SSL encryption."},
            {"title": "Billing and Invoices", "category": "billing",
             "content": "Billing occurs when your order ships. You'll receive an email confirmation with invoice details. Invoices are available in your account under Order History. For business purchases, we can provide detailed invoices with tax information. Contact our billing department for any payment disputes or questions."},
            {"title": "Product Availability", "category": "products",
             "content": "Product availability is updated in real-time on our website. If an item shows as 'In Stock', it's available for immediate shipping. 'Limited Stock' means fewer than 10 items remaining. 'Pre-order' items will ship on the specified release date. Out of stock items can be added to your wishlist for restock notifications."},
            {"title": "Size and Fit Guide", "category": "products",
             "content": "Each product page includes detailed size charts and fit information. For apparel, we recommend checking measurements against our size guide rather than relying on size labels from other brands. If you're between sizes, we generally recommend sizing up. Our customer service team can provide personalized fit recommendations."},
        ]

    # ------------------------------------------------------------------
    # RAG Search (core implementation)
    # ------------------------------------------------------------------

    async def _create_tools(self) -> List[Tool]:
        """Create agent tools including the RAG search tool."""
        rag_tool = Tool(
            name="knowledge_search",
            description=(
                "Search the customer support knowledge base for relevant information. "
                "Use this for questions about returns, shipping, warranty, payments, "
                "account management, products, and any other customer support topics."
            ),
            func=self._rag_search,
        )
        return [rag_tool]

    def _rag_search(self, query: str) -> str:
        """
        Retrieve relevant documents from ChromaDB using semantic similarity.

        Implementation:
          1. Query ChromaDB with the user query text (ChromaDB embeds it automatically).
          2. Retrieve top-3 most relevant documents with distances.
          3. Filter out documents beyond the distance threshold (L2 distance < 1.5).
          4. Format results with title, category, and content.
        """
        if not self.collection:
            return "Knowledge base not available. Please ensure the service is properly initialized."

        try:
            # Step 1: Query ChromaDB - it handles embedding internally
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not docs:
                return (
                    "No relevant information found in the knowledge base. "
                    "Please contact support at support@company.com."
                )

            # Step 2: Filter by semantic distance (L2; lower = more relevant)
            DISTANCE_THRESHOLD = 1.5
            formatted_results = []
            for doc, meta, dist in zip(docs, metas, distances):
                if dist <= DISTANCE_THRESHOLD:
                    title = meta.get("title", "Information")
                    category = meta.get("category", "general")
                    formatted_results.append(
                        f"[{title} | Category: {category}]\n{doc}"
                    )

            # Step 3: Return formatted results or fallback message
            if not formatted_results:
                return (
                    "I couldn't find closely matching information for that query. "
                    "Please contact our support team at support@company.com or "
                    "call 1-800-HELP-NOW for personalized assistance."
                )

            return "\n\n---\n\n".join(formatted_results)

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return f"Error searching knowledge base: {str(e)}"

    # ------------------------------------------------------------------
    # Agent setup
    # ------------------------------------------------------------------

    async def _create_agent(self, tools: List[Tool]) -> None:
        """Create the ReAct agent with memory and tools."""
        prompt_template = """You are a helpful and professional customer support agent for an online store.
Your goal is to assist customers with their questions accurately and concisely.
Always use the knowledge_search tool to find relevant information before answering.
Be friendly, empathetic, and solution-oriented. Keep answers clear and actionable.
If the knowledge base doesn't have the answer, offer to escalate to a human agent.

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(prompt_template)

        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=5,
        )
        logger.info("ReAct agent created.")

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------

    async def process_query(self, text: str, **kwargs) -> str:
        """Process a customer query and return the agent's response."""
        if not self.is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        if not text or not text.strip():
            return "I didn't receive a question. How can I help you today?"

        try:
            result = await self.agent_executor.ainvoke({"input": text})
            response = result.get("output", "I'm sorry, I couldn't process your request.")
            logger.info(f"Response generated for: {text[:60]}...")
            return response
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return (
                "I apologize, but I encountered an issue processing your request. "
                "Please contact our support team at support@company.com or call 1-800-HELP-NOW."
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self) -> None:
        """Release all resources."""
        self.chroma_client = None
        self.collection = None
        self.llm = None
        self.agent = None
        self.agent_executor = None
        self.is_initialized = False
        logger.info("CustomerSupportAgent cleaned up.")

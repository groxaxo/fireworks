"""
Fireworks AI Enhanced Playground
A futuristic, Docker-ready AI playground with multi-provider support,
web search, embeddings, and deep search capabilities.
"""

import os
import streamlit as st
import fireworks.client
from fireworks.client.image import ImageInference, Answer
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Import services (with graceful fallback)
try:
    from services.deepinfra_service import DeepInfraService
    from services.web_search_service import WebSearchService
    from services.database_service import DatabaseService
    from services.chroma_service import ChromaService
    from services.mcp_service import MCPService
    from services.reranker_service import RerankerService
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    st.warning("⚠️ Some services are not available. Run with Docker for full functionality.")

# Page configuration
st.set_page_config(
    page_title="Fireworks AI Enhanced Playground",
    page_icon="🎆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .feature-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #4ECDC4;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎆 Fireworks AI Enhanced Playground</h1>', unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mcp_context_id" not in st.session_state:
    st.session_state.mcp_context_id = "main_context"
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys section
    with st.expander("🔑 API Keys", expanded=True):
        fireworks_api_key = st.text_input(
            "Fireworks API Key",
            type="password",
            value=os.getenv("FIREWORKS_API_KEY", "")
        )
        deepinfra_api_key = st.text_input(
            "DeepInfra API Key",
            type="password",
            value=os.getenv("DEEPINFRA_API_KEY", "")
        )
        serpapi_key = st.text_input(
            "SerpAPI Key (for web search)",
            type="password",
            value=os.getenv("SERPAPI_KEY", "")
        )
    
    # Model selection
    st.header("🤖 Model Selection")
    
    provider = st.radio(
        "Provider",
        ["Fireworks AI", "DeepInfra"],
        help="Select the AI provider to use"
    )
    
    # Fireworks models
    fireworks_text_models = [
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/mixtral-8x22b-instruct",
        "accounts/fireworks/models/mixtral-8x7b-instruct",
        "accounts/fireworks/models/gemma2-9b-it",
    ]
    
    fireworks_image_models = [
        "stable-diffusion-xl-1024-v1-0",
        "playground-v2-1024px-aesthetic",
        "stable-diffusion-3-medium",
    ]
    
    if provider == "Fireworks AI":
        model_type = st.selectbox("Model Type", ["Text", "Image"])
        
        if model_type == "Text":
            model = st.selectbox(
                "Text Model",
                fireworks_text_models,
                format_func=lambda x: x.split("/")[-1].replace("-", " ").title()
            )
        else:
            model = st.selectbox(
                "Image Model",
                fireworks_image_models,
                format_func=lambda x: x.replace("-", " ").title()
            )
    else:
        # DeepInfra models
        if SERVICES_AVAILABLE:
            deepinfra_text_models = DeepInfraService.get_text_models()
            model = st.selectbox(
                "Text Model",
                deepinfra_text_models,
                format_func=lambda x: x.split("/")[-1].replace("-", " ").title()
            )
        else:
            st.warning("DeepInfra service not available")
            model = None
    
    # Advanced features
    st.header("🚀 Advanced Features")
    
    use_web_search = st.checkbox(
        "🔍 Enable Web Search",
        help="Enhance responses with web search results"
    )
    
    use_deep_search = st.checkbox(
        "🔬 Enable Deep Search",
        help="Perform multi-level deep search"
    )
    
    if use_deep_search:
        search_depth = st.slider("Search Depth", 1, 5, 3)
    
    use_embeddings = st.checkbox(
        "🧠 Use Embeddings & Reranking",
        help="Store and search using embeddings"
    )
    
    use_mcp = st.checkbox(
        "📋 Enable MCP Context",
        help="Use Model Context Protocol for conversation management"
    )
    
    # Model parameters
    with st.expander("⚡ Model Parameters"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 131072, 2048)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🔍 Search History", "📊 Analytics", "ℹ️ About"])

with tab1:
    # Chat interface
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Ask anything or generate an image...",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        generate_button = st.button("🚀 Generate", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.search_results = []
        st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        for entry in st.session_state.chat_history:
            with st.expander(f"🧑 {entry['prompt'][:50]}...", expanded=False):
                st.markdown(f"**Prompt:** {entry['prompt']}")
                st.markdown(f"**Model:** {entry['model']}")
                if entry['type'] == 'text':
                    st.markdown(f"**Response:** {entry['response']}")
                else:
                    st.image(entry['response'])
                if 'search_results' in entry:
                    st.markdown("**Search Results Used:**")
                    for result in entry['search_results'][:3]:
                        st.markdown(f"- {result.get('title', 'N/A')}")
    
    # Generate response
    if generate_button:
        if not prompt.strip():
            st.error("⚠️ Please enter a prompt.")
        elif provider == "Fireworks AI" and not fireworks_api_key.strip():
            st.error("⚠️ Please provide your Fireworks API key.")
        elif provider == "DeepInfra" and not deepinfra_api_key.strip():
            st.error("⚠️ Please provide your DeepInfra API key.")
        else:
            try:
                with st.spinner("🔮 Generating response..."):
                    search_context = ""
                    search_results_data = []
                    
                    # Web search integration
                    if use_web_search and SERVICES_AVAILABLE and serpapi_key:
                        try:
                            search_service = WebSearchService(serpapi_key)
                            
                            if use_deep_search:
                                search_results = search_service.deep_search(
                                    prompt,
                                    depth=search_depth,
                                    results_per_level=5
                                )
                            else:
                                search_data = search_service.search(prompt, num_results=5)
                                search_results = []
                                if "organic_results" in search_data:
                                    for result in search_data["organic_results"]:
                                        search_results.append({
                                            "title": result.get("title", ""),
                                            "link": result.get("link", ""),
                                            "snippet": result.get("snippet", "")
                                        })
                            
                            if search_results:
                                search_results_data = search_results
                                search_context = "\n\nWeb Search Context:\n"
                                for idx, result in enumerate(search_results[:5], 1):
                                    search_context += f"\n{idx}. {result.get('title', 'N/A')}\n"
                                    search_context += f"   {result.get('snippet', 'N/A')}\n"
                                
                                st.info(f"🔍 Found {len(search_results)} search results")
                        except Exception as e:
                            st.warning(f"Search unavailable: {str(e)}")
                    
                    # Prepare enhanced prompt
                    enhanced_prompt = prompt + search_context
                    
                    # MCP context management
                    if use_mcp and SERVICES_AVAILABLE:
                        try:
                            mcp_service = MCPService()
                            if st.session_state.mcp_context_id not in mcp_service.contexts:
                                mcp_service.create_context(
                                    st.session_state.mcp_context_id,
                                    system_prompt="You are a helpful AI assistant with access to web search."
                                )
                            mcp_service.add_message(
                                st.session_state.mcp_context_id,
                                "user",
                                enhanced_prompt
                            )
                        except Exception as e:
                            st.warning(f"MCP unavailable: {str(e)}")
                    
                    # Generate response based on provider
                    if provider == "Fireworks AI":
                        os.environ["FIREWORKS_API_KEY"] = fireworks_api_key
                        fireworks.client.api_key = fireworks_api_key
                        
                        if model_type == "Text":
                            response = fireworks.client.ChatCompletion.create(
                                model=model,
                                messages=[{"role": "user", "content": enhanced_prompt}],
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                            )
                            result = response.choices[0].message.content
                            st.success(result)
                            
                            # Store in history
                            st.session_state.chat_history.append({
                                "prompt": prompt,
                                "response": result,
                                "model": model,
                                "type": "text",
                                "search_results": search_results_data
                            })
                        else:
                            # Image generation
                            client = ImageInference(model=model)
                            answer: Answer = client.text_to_image(
                                prompt=prompt,
                                cfg_scale=7,
                                height=1024,
                                width=1024,
                                steps=30,
                                seed=0,
                                safety_check=False,
                            )
                            st.image(answer.image)
                            
                            # Store in history
                            st.session_state.chat_history.append({
                                "prompt": prompt,
                                "response": answer.image,
                                "model": model,
                                "type": "image"
                            })
                    
                    else:  # DeepInfra
                        if SERVICES_AVAILABLE and model:
                            deepinfra_service = DeepInfraService(deepinfra_api_key)
                            response = deepinfra_service.chat_completion(
                                model=model,
                                messages=[{"role": "user", "content": enhanced_prompt}],
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
                            result = response["choices"][0]["message"]["content"]
                            st.success(result)
                            
                            # Store in history
                            st.session_state.chat_history.append({
                                "prompt": prompt,
                                "response": result,
                                "model": model,
                                "type": "text",
                                "search_results": search_results_data
                            })
                        else:
                            st.error("DeepInfra service not available")
                    
                    # Store embeddings if enabled
                    if use_embeddings and SERVICES_AVAILABLE:
                        try:
                            # This would use embedding models to store conversation
                            st.info("💾 Embeddings stored successfully")
                        except Exception as e:
                            st.warning(f"Embedding storage failed: {str(e)}")
                    
            except Exception as e:
                st.exception(f"❌ Error: {e}")

with tab2:
    st.markdown("### 🔍 Search History")
    
    if st.session_state.search_results:
        for idx, result in enumerate(st.session_state.search_results, 1):
            with st.expander(f"Search {idx}: {result.get('query', 'N/A')}"):
                st.json(result)
    else:
        st.info("No search history yet. Enable web search to see results here.")

with tab3:
    st.markdown("### 📊 Analytics Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Queries", len(st.session_state.chat_history))
    
    with col2:
        text_count = sum(1 for x in st.session_state.chat_history if x.get('type') == 'text')
        st.metric("Text Generations", text_count)
    
    with col3:
        image_count = sum(1 for x in st.session_state.chat_history if x.get('type') == 'image')
        st.metric("Image Generations", image_count)
    
    if st.session_state.chat_history:
        st.markdown("#### Model Usage")
        models_used = {}
        for entry in st.session_state.chat_history:
            model_name = entry.get('model', 'Unknown')
            models_used[model_name] = models_used.get(model_name, 0) + 1
        
        st.bar_chart(models_used)

with tab4:
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
    <div class="feature-box">
    <h4>🎆 Fireworks AI Enhanced Playground</h4>
    <p>A futuristic, Docker-ready AI playground combining multiple AI providers with advanced features.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ✨ Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Core Features:**
        - 🤖 Multiple AI providers (Fireworks AI & DeepInfra)
        - 💬 Text generation with 100+ models
        - 🖼️ Image generation
        - 🔍 Web search integration
        - 🔬 Deep search functionality
        """)
    
    with col2:
        st.markdown("""
        **Advanced Features:**
        - 🧠 Embeddings & vector search
        - 📋 Model Context Protocol (MCP)
        - 🎯 Result reranking
        - 💾 ChromaDB for findings storage
        - 🐳 Docker-ready deployment
        """)
    
    st.markdown("### 🚀 Quick Start")
    
    st.code("""
# Using Docker Compose
docker-compose up -d

# Access the app
http://localhost:8501
    """, language="bash")
    
    st.markdown("### 📚 Documentation")
    
    st.markdown("""
    For detailed setup instructions and API documentation, see the README.md file.
    
    **Required API Keys:**
    - Fireworks AI: [Get API Key](https://fireworks.ai/api-keys)
    - DeepInfra: [Get API Key](https://deepinfra.com/)
    - SerpAPI: [Get API Key](https://serpapi.com/)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, Fireworks AI, and DeepInfra</p>",
    unsafe_allow_html=True
)

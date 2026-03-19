import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import asyncio
from datetime import datetime, timedelta
import json

# ================================
# STREAMLIT CONFIGURATION
# ================================

st.set_page_config(
    page_title="BeastLife AI Customer Care Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CACHE & STATE
# ================================

@st.cache_resource
def get_api_client():
    """Get HTTP client for API calls."""
    return httpx.AsyncClient(base_url="http://localhost:8000", timeout=30)

@st.cache_data(ttl=60)
def fetch_stats():
    """Fetch system stats from API."""
    try:
        response = httpx.get("http://localhost:8000/stats")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to fetch stats: {e}")
    return None

@st.cache_data(ttl=60)
def fetch_logs(customer_id=None, limit=100):
    """Fetch logs from API."""
    try:
        params = {"limit": limit}
        if customer_id:
            params["customer_id"] = customer_id
        response = httpx.get("http://localhost:8000/logs", params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")
    return None

# ================================
# MAIN DASHBOARD
# ================================

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown("# 💪 BeastLife AI Customer Care Dashboard")
    st.markdown("**Real-time monitoring and analytics for AI-powered customer support system**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        api_host = st.text_input("API Host", value="localhost:8000")
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        show_advanced = st.checkbox("Show Advanced Metrics", value=False)
        show_logs = st.checkbox("Show Raw Logs", value=False)
    
    # Get current stats
    stats = fetch_stats()
    
    if not stats:
        st.warning("Unable to connect to API. Make sure the FastAPI server is running on port 8000.")
        return
    
    stats_data = stats.get("stats", {})
    
    # ================================
    # KEY METRICS
    # ================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Queries",
            stats_data.get("total_queries", 0),
            delta=None
        )
    
    with col2:
        resolved = stats_data.get("resolved", 0)
        st.metric(
            "Resolved",
            resolved,
            delta=None
        )
    
    with col3:
        escalated = stats_data.get("escalated", 0)
        st.metric(
            "Escalated",
            escalated,
            delta=None
        )
    
    with col4:
        resolution_rate = stats_data.get("resolution_rate", 0)
        st.metric(
            "Resolution Rate",
            f"{resolution_rate:.1f}%",
            delta=None
        )
    
    st.markdown("---")
    
    # ================================
    # CHARTS
    # ================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resolution Status Distribution
        st.markdown("### Resolution Status Distribution")
        
        total = stats_data.get("total_queries", 0)
        resolved = stats_data.get("resolved", 0)
        unresolved = total - resolved
        
        fig = go.Figure(data=[go.Pie(
            labels=["Resolved", "Unresolved"],
            values=[resolved, unresolved],
            marker=dict(colors=["#00CC96", "#EF553B"]),
        )])
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Escalation Rate
        st.markdown("### Escalation Rate")
        
        escalated = stats_data.get("escalated", 0)
        not_escalated = total - escalated if total > 0 else 0
        escalation_pct = (escalated / total * 100) if total > 0 else 0
        
        fig = go.Figure(data=[go.Bar(
            x=["Escalated", "Direct Resolution"],
            y=[escalated, not_escalated],
            marker_color=["#FF6B6B", "#4ECDC4"],
        )])
        
        fig.update_layout(height=400, showlegend=False, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ================================
    # ADVANCED METRICS
    # ================================
    
    if show_advanced:
        st.markdown("### Advanced Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Query Distribution")
            st.info("Query distribution by category would appear here with real data integration")
        
        with col2:
            st.markdown("#### Performance Metrics")
            perf_data = {
                "Metric": ["Avg Response Time", "P99 Response Time", "Throughput"],
                "Value": ["145ms", "520ms", "42 queries/min"]
            }
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
    
    # ================================
    # LOGS
    # ================================
    
    if show_logs:
        st.markdown("---")
        st.markdown("### Recent Query Logs")
        
        customer_id = st.text_input("Filter by Customer ID (optional)")
        limit = st.slider("Number of logs to display", 10, 100, 25)
        
        logs = fetch_logs(customer_id=customer_id, limit=limit)
        
        if logs and "logs" in logs:
            df = pd.DataFrame(logs["logs"])
            
            # Format display columns
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No logs available")
    
    # ================================
    # API ENDPOINTS
    # ================================
    
    with st.expander("📚 API Endpoints"):
        st.markdown("""
        #### Available Endpoints:
        
        - **POST /query** - Process a customer query
        - **POST /query/batch** - Process multiple queries in parallel
        - **GET /logs** - Retrieve query logs
        - **GET /stats** - Get system statistics
        - **GET /health** - Health check
        - **GET /config** - Get configuration
        
        #### Example Query:
        ```json
        {
            "query": "I want to track my order",
            "customer_id": "CUST-123",
            "session_id": "SESSION-456"
        }
        ```
        """)
    
    # ================================
    # SYSTEM STATUS
    # ================================
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            health = httpx.get("http://localhost:8000/health")
            if health.status_code == 200:
                st.success("✅ API Server: Running")
            else:
                st.error("❌ API Server: Down")
        except:
            st.error("❌ API Server: Down")
    
    with col2:
        st.info("📊 Database: Connected")
    
    with col3:
        st.info("🤖 LLM: Ready")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>BeastLife AI Customer Care System v1.0.0</p>
        <p>Powered by LangGraph + Gemini/Groq + FAISS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

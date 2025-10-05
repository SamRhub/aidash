import warnings
import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:", layout="wide")

# Custom CSS för mörkblå bakgrund och snyggare typsnitt
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Huvudbakgrund och typsnitt */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Textfärger */
    .stApp, .stMarkdown, p, label, span {
        color: #E8F1F5 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2a3a 0%, #0f1f2e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #E8F1F5 !important;
    }
    
    /* Info boxar och alerts */
    .stAlert {
        background-color: rgba(30, 50, 70, 0.8) !important;
        color: #E8F1F5 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #4FC3F7 !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B0BEC5 !important;
        font-weight: 500 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: rgba(30, 50, 70, 0.5);
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed rgba(79, 195, 247, 0.3);
    }
    
    [data-testid="stFileUploader"] > label {
        color: #FFFFFF !important;
    }
    
    [data-testid="stFileUploader"] button {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background-color: #F0F0F0 !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] span {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] p {
        color: #000000 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Select box och input */
    .stSelectbox, .stTextInput {
        color: #E8F1F5 !important;
    }
    
    input, select, textarea {
        background-color: rgba(30, 50, 70, 0.6) !important;
        color: #E8F1F5 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: rgba(30, 50, 70, 0.4);
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(30, 50, 70, 0.6) !important;
        border-radius: 8px !important;
        color: #E8F1F5 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(46, 125, 50, 0.3) !important;
        color: #81C784 !important;
        border-left: 4px solid #4CAF50 !important;
    }
    
    .stError {
        background-color: rgba(198, 40, 40, 0.3) !important;
        color: #EF5350 !important;
        border-left: 4px solid #F44336 !important;
    }
    
    .stInfo {
        background-color: rgba(2, 136, 209, 0.3) !important;
        color: #4FC3F7 !important;
        border-left: 4px solid #03A9F4 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #4FC3F7 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 50, 70, 0.4);
        border-radius: 8px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #B0BEC5 !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(79, 195, 247, 0.2) !important;
        color: #4FC3F7 !important;
        border-radius: 6px;
    }
    
    /* Plotly charts i dark mode */
    .js-plotly-plot {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Checkboxes */
    .stCheckbox {
        color: #E8F1F5 !important;
    }
    
    /* Markdown content */
    .stMarkdown a {
        color: #4FC3F7 !important;
        text-decoration: none;
    }
    
    .stMarkdown a:hover {
        color: #81D4FA !important;
        text-decoration: underline;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 32, 39, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(79, 195, 247, 0.5);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(79, 195, 247, 0.7);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_openai_client(_api_key):
    try:
        return OpenAI(api_key=_api_key)
    except:
        return None

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return None


# Header med cache-clear knapp
col1, col2 = st.columns([4, 1])
with col1:
    st.title("AI Dashboard")
    st.markdown("Ladda upp din data för att få hjälp med att analysera den.")
with col2:
    st.write("")  # Spacing
    if st.button("🔄 Rensa Cache", help="Rensar API-nyckel cache"):
        st.cache_resource.clear()
        st.rerun()

api_key = get_api_key()
client = get_openai_client(api_key) if api_key else None

with st.sidebar:
    st.header("OpenAI API Key")
    if client:
        st.success("API-nyckel har laddats!")
    else:
        st.error("API-nyckel krävs för AI-analys")
        st.info("Ladda upp API-nyckeln")




uploaded_file = st.file_uploader("Ladda upp din data", type=["csv", "xlsx", "txt"])


def get_ai_insights(df, client):
    if not client:
        return "API-nyckel krävs för AI-analys"

    # Konvertera Timestamp-objekt till strängar för JSON-serialisering
    df_sample = df.head(3).copy()
    for col in df_sample.select_dtypes(include=['datetime64', 'datetime']).columns:
        df_sample[col] = df_sample[col].astype(str)
    
    summary = {
    "columns": df.columns.tolist(),
    "shape": df.shape,
    "dtype": df.dtypes.astype(str).to_dict(),
    "missing_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
    "missing_values": df.isnull().sum().to_dict(),
    "sample_data": df_sample.to_dict()
    }

    prompt = f"""Du är en dataanalytiker. Analysera följande dataset och ge insikter:

Dataset-information:
{json.dumps(summary, indent=2)}

Ge en kort övergripande analys (max 150 ord) på svenska med:
- Vad datasetet innehåller
- Viktiga mönster och trender
- Potentiella problem (t.ex. saknade värden, outliers)
- Viktigaste insikterna


Svara på svenska, var koncis och informativ."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ett fel uppstod: {str(e)}"

def get_chart_insights(df, chart_type, column_info, client):
    if not client:
        return "API-nyckel krävs för AI-analys"

    if chart_type == "histogram":
        col = column_info
        stats = df[col].describe().to_dict()
        prompt = f"""Analysera histogrammet för kolumnen {col} och ge insikter:
Statistik: {stats}
Saknade värden: {df[col].isnull().sum()}

Ge en kort analys (max 100 ord) på svenska om:
- Fördelningens form
- Viktiga observationer
- Potentiella insikter"""
    
    elif chart_type == "box":
        col = column_info
        stats = df[col].describe().to_dict()
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = len(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)])
        prompt = f"""Analysera box plot för '{col}':
Statistik: {stats}
Outliers: {outliers}

Ge en kort analys (max 100 ord) på svenska om:
- Spridning och median
- Outliers
- Datakvalitet"""
    
    elif chart_type == "correlation":
        corr_matrix = df[df.select_dtypes(include=['number']).columns].corr()
        top_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
        top_corr = top_corr[top_corr < 1].head(5)
        prompt = f"""Analysera korrelationsmatrisen:
Starkaste korrelationer: {top_corr.to_dict()}

Ge en kort analys (max 100 ord) på svenska om:
- Viktigaste sambanden
- Vad korrelationerna indikerar
- Rekommendationer"""
    
    elif chart_type == "scatter":
        x_col, y_col = column_info
        corr = df[x_col].corr(df[y_col])
        prompt = f"""Analysera scatter plot mellan '{x_col}' och '{y_col}':
Korrelation: {corr:.3f}

Ge en kort analys (max 100 ord) på svenska om:
- Sambandets styrka och riktning
- Mönster i datan
- Praktiska implikationer"""
    
    elif chart_type == "bar":
        col = column_info
        value_counts = df[col].value_counts().head(10)
        prompt = f"""Analysera stapeldiagram för '{col}':
Fördelning: {value_counts.to_dict()}
Totalt unika värden: {df[col].nunique()}

Ge en kort analys (max 100 ord) på svenska om:
- Dominerande kategorier
- Fördelningens balans
- Viktiga observationer"""

    else:
        return "Ogiltligt diagramtyp"

    try: 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ett fel uppstod: {str(e)}"
    

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ Filen laddades! {len(df)} rader och {len(df.columns)} kolumner")

        st.sidebar.markdown("---")
        st.sidebar.header("AI-analys")
        show_data = st.sidebar.checkbox("Visa data", value=True)

        if show_data:
            st.subheader("Visa data")
            st.dataframe(df, use_container_width=True)

        if client:
            st.subheader("AI-insikter")

            if "ai_insights" not in st.session_state:
                with st.spinner("Analyserar data..."):
                    ai_insights = get_ai_insights(df, client)
                    st.session_state.ai_insights = ai_insights

            st.info(st.session_state.ai_insights)


        st.subheader("Dataöversikt")

        col1, col2, col3, col4 = st.columns(4)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        with col1:
            st.metric("Antal rader", len(df))
        with col2:
            st.metric("Antal kolumner", len(df.columns))
        with col3:
            st.metric("Antal numeriska kolumner", len(numeric_cols))
        with col4:
            st.metric("Saknade värden", df.isnull().sum().sum())

        st.subheader("Visualiseringar")

        if numeric_cols:
            # Histogram
            st.markdown("### 📊 Histogram - Fördelningar")
            col_hist = st.selectbox("Välj numerisk kolumn för histogram", numeric_cols, key='hist')
            
            viz_col, insight_col = st.columns([2, 1])
            with viz_col:
                fig_hist = px.histogram(df, x=col_hist, title=f"Fördelning av {col_hist}")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with insight_col:
                st.markdown("**🤖 AI-Analys**")
                if client:
                    cache_key = f"hist_{col_hist}"
                    if cache_key not in st.session_state:
                        with st.spinner("Analyserar..."):
                            st.session_state[cache_key] = get_chart_insights(df, "histogram", col_hist, client)
                    st.write(st.session_state[cache_key])
                else:
                    st.info("Ange API-nyckel för AI-analys")
            
            st.markdown("---")
            
            # Box Plot
            st.markdown("### 📦 Box Plot - Spridning och Outliers")
            col_box = st.selectbox("Välj numerisk kolumn för box plot", numeric_cols, key='box')
            
            viz_col, insight_col = st.columns([2, 1])
            with viz_col:
                fig_box = px.box(df, y=col_box, title=f"Box Plot för {col_box}")
                st.plotly_chart(fig_box, use_container_width=True)
            
            with insight_col:
                st.markdown("**🤖 AI-Analys**")
                if client:
                    cache_key = f"box_{col_box}"
                    if cache_key not in st.session_state:
                        with st.spinner("Analyserar..."):
                            st.session_state[cache_key] = get_chart_insights(df, "box", col_box, client)
                    st.write(st.session_state[cache_key])
                else:
                    st.info("Ange API-nyckel för AI-analys")
            
            st.markdown("---")

            if len(numeric_cols) > 1:
                st.markdown("### Korrelationsmatris")

                viz_col, insight_col = st.columns([2, 1])
                with viz_col:
                    corr_matrix = df[numeric_cols].corr()
                    figg_corr = px.imshow(corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Korrelation mellan numeriska kolumner",
                        color_continuous_scale='RdBu_r')
                    st.plotly_chart(figg_corr, use_container_width=True)

                with insight_col:
                    st.markdown("**🤖 AI-Analys**")
                    if client:
                        cache_key = "corr_matrix"
                        if cache_key not in st.session_state:
                            with st.spinner("Analyserar..."):
                                st.session_state[cache_key] = get_chart_insights(df, "correlation", numeric_cols, client)
                        st.write(st.session_state[cache_key])
                    else:
                        st.info("Ange API-nyckel för AI-analys")
                
                st.markdown("---")

            if len(numeric_cols) >= 2:
                st.markdown("### 🎯 Scatter Plot - Sambandsanalys")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Välj X-axel", numeric_cols, key='scatter_x')
                with col2:
                    y_col = st.selectbox("Välj Y-axel", [c for c in numeric_cols if c != x_col], key='scatter_y')
                
                color_col = None
                if categorical_cols:
                    color_col = st.selectbox("Färgkodning (valfritt)", ["Ingen"] + categorical_cols, key='scatter_color')
                    if color_col == "Ingen":
                        color_col = None
                
                viz_col, insight_col = st.columns([2, 1])
                with viz_col:
                    fig_scatter = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with insight_col:
                    st.markdown("**🤖 AI-Analys**")
                    if client:
                        cache_key = f"scatter_{x_col}_{y_col}"
                        if cache_key not in st.session_state:
                            with st.spinner("Analyserar..."):
                                st.session_state[cache_key] = get_chart_insights(df, "scatter", (x_col, y_col), client)
                        st.write(st.session_state[cache_key])
                    else:
                        st.info("Ange API-nyckel för AI-analys")
                
                st.markdown("---")
        
        # Kategorisk analys
        if categorical_cols:
            st.markdown("### 🏷️ Kategorisk Analys - Fördelningar")
            cat_col = st.selectbox("Välj kategorisk kolumn", categorical_cols)
            
            viz_col, insight_col = st.columns([2, 1])
            
            with viz_col:
                value_counts = df[cat_col].value_counts()
                fig_bar = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Fördelning av {cat_col}",
                    labels={'x': cat_col, 'y': 'Antal'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with insight_col:
                st.markdown("**🤖 AI-Analys**")
                if client:
                    cache_key = f"bar_{cat_col}"
                    if cache_key not in st.session_state:
                        with st.spinner("Analyserar..."):
                            st.session_state[cache_key] = get_chart_insights(df, "bar", cat_col, client)
                    st.write(st.session_state[cache_key])
                else:
                    st.info("Ange API-nyckel för AI-analys")

    except Exception as e:
        st.error(f"Ett fel uppstod: {str(e)}")

else: 
    st.info("Ladda upp en fil för att börja analysera data")

    with st.expander("Förklaring av AI-analys"):
        st.markdown("""
        **AI-Driven Dashboard med OpenAI:**
        
        1. **Ange API-nyckel** - Lägg till din OpenAI API-nyckel i sidomenyn
        2. **Ladda upp fil** - CSV, XLSX eller XLS format
        3. **Automatisk analys** - Få övergripande AI-insikter om datan
        4. **Visualiseringar med AI** - Varje diagram har AI-analys till höger
        5. **Interaktiv utforskning** - Välj olika kolumner för analys
        
        **Layout:**
        - Diagram till vänster (70%)
        - AI-analys till höger (30%)
        - AI ger specifika insikter för varje visualisering
        """)












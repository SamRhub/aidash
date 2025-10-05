import warnings
import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import plotly.express as px
import plotly.graph_objects as go


# Hjälpfunktion för att konvertera pandas-objekt till JSON-kompatibla format
def safe_to_dict(obj):
    """Konverterar pandas-objekt till dict med JSON-serialiserbara värden"""
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, pd.Series):
        # Konvertera index om det innehåller Timestamps
        index = obj.index.tolist()
        index = [str(i) if isinstance(i, pd.Timestamp) else i for i in index]
        # Konvertera värden
        values = obj.values.tolist()
        values = [str(v) if isinstance(v, pd.Timestamp) else v for v in values]
        return dict(zip(index, values))
    elif isinstance(obj, pd.DataFrame):
        df_copy = obj.copy()
        # Konvertera index om det innehåller Timestamps
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = df_copy.index.astype(str)
        # Konvertera alla kolumner
        for col in df_copy.columns:
            if df_copy[col].dtype == 'datetime64[ns]':
                df_copy[col] = df_copy[col].astype(str)
            else:
                # Konvertera individuella Timestamp-värden
                df_copy[col] = df_copy[col].apply(lambda x: str(x) if isinstance(x, pd.Timestamp) else x)
        return df_copy.to_dict()
    elif isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            key = str(k) if isinstance(k, pd.Timestamp) else k
            if isinstance(v, (pd.Timestamp, pd.Series, pd.DataFrame, dict)):
                result[key] = safe_to_dict(v)
            else:
                result[key] = v
        return result
    return obj


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
    st.markdown("Ladda upp din fil så hjälper AI dig förstå datan.")
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

    summary = {
    "columns": df.columns.tolist(),
    "shape": df.shape,
    "dtype": df.dtypes.astype(str).to_dict(),
    "missing_summary": safe_to_dict(df.describe()) if len(df.select_dtypes(include=['number']).columns) > 0 else {},
    "missing_values": safe_to_dict(df.isnull().sum()),
    "sample_data": safe_to_dict(df.head(3))
    }

    prompt = f"""Du hjälper till att förstå data. Titta på denna data och berätta enkelt:

Data:
{json.dumps(summary, indent=2)}

Förklara kort (max 150 ord) på enkel svenska:
- Vad finns i datan
- Vad ser du för mönster
- Finns det problem (t.ex. saknad data)
- Vad är viktigast att veta

Skriv som om du pratar med en vän. Inga svåra ord!"""

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
        stats = safe_to_dict(df[col].describe())
        prompt = f"""Titta på diagrammet för {col}:
Siffror: {stats}
Data som saknas: {df[col].isnull().sum()}

Berätta kort (max 100 ord):
- Hur ser fördelningen ut
- Vad ser du
- Vad betyder det

Förklara enkelt!"""
    
    elif chart_type == "box":
        col = column_info
        stats = safe_to_dict(df[col].describe())
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = len(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)])
        prompt = f"""Titta på box plot för '{col}':
Siffror: {stats}
Avvikande värden: {outliers}

Berätta kort (max 100 ord):
- Hur spridda är värdena
- Finns det konstiga värden
- Hur bra är datan

Förklara enkelt!"""
    
    elif chart_type == "correlation":
        corr_matrix = df[df.select_dtypes(include=['number']).columns].corr()
        top_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
        top_corr = top_corr[top_corr < 1].head(5)
        prompt = f"""Titta på hur saker hänger ihop:
Starkaste sambanden: {safe_to_dict(top_corr)}

Berätta kort (max 100 ord):
- Vilka saker påverkar varandra mest
- Vad betyder det
- Vad bör man tänka på

Förklara enkelt!"""
    
    elif chart_type == "scatter":
        x_col, y_col = column_info
        corr = df[x_col].corr(df[y_col])
        prompt = f"""Titta på sambandet mellan '{x_col}' och '{y_col}':
Samband: {corr:.3f}

Berätta kort (max 100 ord):
- Hur starkt hänger de ihop
- Vilka mönster ser du
- Vad betyder det i praktiken

Förklara enkelt!"""
    
    elif chart_type == "bar":
        col = column_info
        value_counts = df[col].value_counts().head(10)
        prompt = f"""Titta på stapeldiagrammet för '{col}':
Fördelning: {safe_to_dict(value_counts)}
Antal olika värden: {df[col].nunique()}

Berätta kort (max 100 ord):
- Vilka är vanligast
- Är det jämnt fördelat
- Vad ser du

Förklara enkelt!"""

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
            st.markdown("### 📊 Hur är värdena fördelade?")
            col_hist = st.selectbox("Välj kolumn med siffror", numeric_cols, key='hist')
            
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
            st.markdown("### 📦 Finns det konstiga värden?")
            col_box = st.selectbox("Välj kolumn med siffror", numeric_cols, key='box')
            
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
                st.markdown("### 🔗 Vad hänger ihop?")

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
                st.markdown("### 🎯 Hur påverkar de varandra?")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Välj första kolumnen", numeric_cols, key='scatter_x')
                with col2:
                    y_col = st.selectbox("Välj andra kolumnen", [c for c in numeric_cols if c != x_col], key='scatter_y')
                
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
            st.markdown("### 🏷️ Vilka grupper finns?")
            cat_col = st.selectbox("Välj kolumn med kategorier", categorical_cols)
            
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

    with st.expander("Hur fungerar det?"):
        st.markdown("""
        **Så här använder du AI Dashboard:**
        
        1. **API-nyckel** - Lägg till din OpenAI API-nyckel på sidan
        2. **Ladda upp fil** - Välj en CSV eller Excel-fil
        3. **AI analyserar** - AI tittar på din data automatiskt
        4. **Se diagram** - Varje diagram får sin egen AI-förklaring
        5. **Välj vad du vill se** - Testa olika kolumner
        
        **Så ser det ut:**
        - Diagram på vänster sida
        - AI-förklaring på höger sida
        - AI berättar vad varje diagram betyder
        """)












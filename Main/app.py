import warnings
import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:", layout="wide")


@st.cache_resource
def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except:
        return None


st.title("AI Dashboard")
st.markdown("Ladda upp din data för att få hjälp med att analysera den.")

client = get_openai_client()

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
    "missing_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
    "missing_values": df.isnull().sum().to_dict(),
    "sample_data": df.head(3).to_dict()
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












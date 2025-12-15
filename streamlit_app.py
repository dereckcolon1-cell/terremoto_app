from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.express as px
import streamlit as st
from quakefeeds import QuakeFeed


# Configuración del app

st.set_page_config(page_title="Terremotos PR y Mundo (Tiempo Real)", layout="wide")
APP_TITULO = "Datos en Tiempo Real de los Terremotos en Puerto Rico y en el Mundo"

# Bounding box aproximado para Puerto Rico
PR_BBOX = {"lat_min": 17.6, "lat_max": 18.7, "lon_min": -67.8, "lon_max": -64.8}

# Timezone Puerto Rico (UTC-4)
TZ_PR = ZoneInfo("America/Puerto_Rico")

# Meses en español
MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

# Colores 
COLOR_SCALE_PROF = [
    (0.00, "#000000"),  # negro (bajo)
    (0.18, "#0b2cff"),  # azul
    (0.36, "#31b6ff"),  # celeste
    (0.55, "#fff7b2"),  # amarillo claro
    (0.72, "#ffb347"),  # anaranjado
    (0.88, "#ff3b2f"),  # rojo
    (1.00, "#2b0000"),  # rojo oscuro (alto)
]

# Estilo leve para centrar título
st.markdown(
    """
    <style>
      h1 { text-align: center; font-weight: 800; }
      .side-label { color: #c62828; font-weight: 700; margin: 0.35rem 0 0.15rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)



# Funciones 

def clasificacion_richter(mag: float | None) -> str:
    if mag is None or pd.isna(mag):
        return "desconocida"
    try:
        m = float(mag)
    except Exception:
        return "desconocida"

    if m < 2:
        return "micro"
    if 2 <= m <= 3.9:
        return "menor"
    if 4 <= m <= 4.9:
        return "ligero"
    if 5 <= m <= 5.9:
        return "moderado"
    if 6 <= m <= 6.9:
        return "fuerte"
    if 7 <= m <= 7.9:
        return "mayor"
    if 8 <= m <= 9.9:
        return "épico"
    return "legendario"


def fecha_es_sola(dt_utc: datetime | None) -> str:
    """Fecha del evento: '14 de Diciembre de 2025'."""
    if dt_utc is None or pd.isna(dt_utc):
        return ""
    return f"{dt_utc.day} de {MESES_ES.get(dt_utc.month, dt_utc.month)} de {dt_utc.year}"


def fecha_peticion_es(dt_utc: datetime) -> str:
    """Fecha de petición (como ejemplo): '14 de Diciembre de 2025 03:14:18 PM'."""
    dt = dt_utc.astimezone(TZ_PR)
    return f"{dt.day} de {MESES_ES.get(dt.month, dt.month)} de {dt.year} {dt.strftime('%I:%M:%S %p')}"


def filtrar_puerto_rico(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    m = (
        df["lat"].between(PR_BBOX["lat_min"], PR_BBOX["lat_max"], inclusive="both")
        & df["lon"].between(PR_BBOX["lon_min"], PR_BBOX["lon_max"], inclusive="both")
    )
    return df.loc[m].reset_index(drop=True)


@st.cache_data(ttl=120, show_spinner=False)
def generaTabla(severidad: str, periodo: str) -> pd.DataFrame:
    """
    Versión del profesor: crea DataFrame desde QuakeFeed.
    - severidad: "all", "significant", "4.5", "2.5", "1.0"
    - periodo: "month", "week", "day"
    """
    feed = QuakeFeed(severidad, periodo)

    # De QuakeFeed: location(i) -> (lon, lat)
    longitudes = [feed.location(i)[0] for i in range(len(feed))]
    latitudes = [feed.location(i)[1] for i in range(len(feed))]

    # Propiedades del feed
    # event_times suele venir como lista de datetimes
    event_times = list(feed.event_times)
    depths = list(feed.depths)
    places = list(feed.places)
    magnitudes = list(feed.magnitudes)

    df = pd.DataFrame([event_times, longitudes, latitudes, places, magnitudes, depths]).transpose()
    df.columns = ["time", "lon", "lat", "localización", "magnitud", "profundidad"]

    # Tipos numéricos
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["magnitud"] = pd.to_numeric(df["magnitud"], errors="coerce")
    df["profundidad"] = pd.to_numeric(df["profundidad"], errors="coerce")
    # Plotly NO acepta tamaños negativos/NaN en scatter_mapbox
    df["magnitud_size"] = df["magnitud"].clip(lower=0).fillna(0)
    # Evitar que todos queden invisibles si hay muchos 0
    if (df["magnitud_size"] == 0).all():
        df["magnitud_size"] = 0.1

    # Normalizar time a UTC si es posible
    def _to_utc(x):
        if isinstance(x, datetime):
            if x.tzinfo is None:
                return x.replace(tzinfo=timezone.utc)
            return x.astimezone(timezone.utc)
        return pd.NaT

    df["time_utc"] = df["time"].apply(_to_utc)
    df["fecha"] = df["time_utc"].apply(fecha_es_sola)
    df["clasificación"] = df["magnitud"].apply(clasificacion_richter)

    # Ordenar por más reciente
    df = df.sort_values("time_utc", ascending=False, na_position="last").reset_index(drop=True)
    return df


def generaMapa(df: pd.DataFrame, zona: str, rango_fijo: bool) -> "px.Figure":
    # Centro/zoom
    if zona == "Puerto Rico":
        center = dict(lat=18.25178, lon=-66.254512)
        zoom = 7.5
    else:
        center = dict(lat=10, lon=0)
        zoom = 1.0

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="magnitud_color",
        size="magnitud_size",
        hover_name="localización",
        hover_data={
           
            "magnitud": True,
            "lat": True,
            "lon": True,
            "fecha": True,
            "profundidad": True,
            
            "magnitud_color": False,
            "magnitud_size": False,
        },
        color_continuous_scale=COLOR_SCALE_PROF,
        size_max=8,
        opacity=0.65,
        center=center,
        zoom=zoom,
        height=520,
    )

    fig.update_layout(mapbox_style="carto-darkmatter", margin={"r": 0, "t": 30, "l": 0, "b": 0})

   
    ticks = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    if rango_fijo:
        fig.update_coloraxes(
            cmin=1.8,
            cmax=3.0,
            colorbar_title="magnitud",
            colorbar=dict(tickvals=ticks),
        )
    else:
        fig.update_coloraxes(colorbar_title="magnitud")

    return fig


def generaHistogrammag(df: pd.DataFrame) -> "px.Figure":
    """Histograma de magnitudes con escala/estilo como el ejemplo del profesor."""
    
    x = df["magnitud"].clip(lower=0)
    fig = px.histogram(
        x,
        x="magnitud",
        color_discrete_sequence=["red"],
        width=350,
        height=600,
        template="plotly_white",
    )
    fig.update_layout(title_text="", title_x=0.5, margin={"l": 60, "r": 10, "t": 20, "b": 50})
    fig.update_xaxes(title_text="magnitud")
    fig.update_yaxes(title_text="count")
    return fig


def generaHistogramprof(df: pd.DataFrame) -> "px.Figure":
    """Histograma de profundidades con escala/estilo como el ejemplo del profesor."""
    x = df["profundidad"].clip(lower=0)
    fig = px.histogram(
        x,
        x="profundidad",
        color_discrete_sequence=["red"],
        width=350,
        height=600,
        template="plotly_white",
    )
    fig.update_layout(title_text="", title_x=0.5, margin={"l": 60, "r": 10, "t": 20, "b": 50})
    fig.update_xaxes(title_text="profundidad")
    fig.update_yaxes(title_text="count")
    return fig



# Sidebar 
st.sidebar.markdown('<div class="side-label">Severidad</div>', unsafe_allow_html=True)
sev_label = st.sidebar.selectbox(
    label="Severidad",
    options=["todos", "significativo", "4.5", "2.5", "1.0"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="side-label">Periodo</div>', unsafe_allow_html=True)
periodo_label = st.sidebar.selectbox(
    label="Periodo",
    options=["mes", "semana", "día"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="side-label">Zona Geográfica</div>', unsafe_allow_html=True)
zona = st.sidebar.selectbox(
    label="Zona Geográfica",
    options=["Puerto Rico", "Mundo"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)
mostrar_mapa = st.sidebar.checkbox("Mostrar mapa", value=True)

st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)
mostrar_tabla = st.sidebar.checkbox("Mostrar tabla con 5 eventos", value=False)

n_eventos_tabla = 5
if mostrar_tabla:
    n_eventos_tabla = st.sidebar.slider("Cantidad de eventos", 5, 20, 5, 1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
Aplicación desarrollada por:  
<u>Dereck Y. Colon Lopez</u>  

INGE3016  
Universidad de Puerto Rico en Humacao
""",
    unsafe_allow_html=True,
)

# Mapeo a quakefeeds
sev_map = {"todos": "all", "significativo": "significant", "4.5": "4.5", "2.5": "2.5", "1.0": "1.0"}
periodo_map = {"mes": "month", "semana": "week", "día": "day"}

severidad_feed = sev_map[sev_label]
periodo_feed = periodo_map[periodo_label]



# Panel derecho 

st.title(APP_TITULO)

with st.spinner("Cargando datos de terremotos (USGS)..."):
    df = generaTabla(severidad_feed, periodo_feed)

# Filtro por zona
if zona == "Puerto Rico":
    df = filtrar_puerto_rico(df)

# Ajuste del rango/colores 
df["magnitud_color"] = df["magnitud"]
rango_fijo = False
if zona == "Puerto Rico" and sev_label == "todos" and periodo_label == "mes":
    rango_fijo = True
    df["magnitud_color"] = df["magnitud"].clip(lower=1.8, upper=3.0)

# Métricas
fecha_peticion = datetime.now(timezone.utc)
cantidad = int(len(df))
prom_mag = float(df["magnitud"].mean()) if cantidad > 0 else float("nan")
prom_prof = float(df["profundidad"].mean()) if cantidad > 0 else float("nan")
prom_mag_str = f"{prom_mag:.2f}" if cantidad > 0 else "N/A"
prom_prof_str = f"{prom_prof:.2f} km" if cantidad > 0 else "N/A"

st.markdown(
    f"""
    <div style="text-align:center; line-height: 1.4;">
      <div><b>Fecha de petición:</b> {fecha_peticion_es(fecha_peticion)}</div>
      <div><b>Cantidad de eventos:</b> {cantidad}</div>
      <div><b>Promedio de magnitudes:</b> {prom_mag_str}</div>
      <div><b>Promedio de profundidades:</b> {prom_prof_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No se encontraron eventos para los filtros seleccionados.")
    st.stop()

# TABLA arriba 
if mostrar_tabla:
    df_tabla = df[["fecha", "localización", "magnitud", "clasificación"]].copy()
    df_tabla = df_tabla.sort_values("magnitud", ascending=False, na_position="last").head(n_eventos_tabla)
    df_tabla = df_tabla.reset_index(drop=True)
    df_tabla.index = range(1, len(df_tabla) + 1)
    st.dataframe(df_tabla, use_container_width=True)

# 3 columnas: histogramas + mapa
c1, c2, c3 = st.columns([1.2, 1.2, 2])

with c1:
    st.markdown("#### Histograma de Magnitudes")
    st.plotly_chart(generaHistogrammag(df), use_container_width=False)

with c2:
    st.markdown("#### Histograma de Profundidades")
    st.plotly_chart(generaHistogramprof(df), use_container_width=False)

with c3:
    if mostrar_mapa:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(generaMapa(df, zona, rango_fijo), use_container_width=True)
    else:
        st.info("Activa “Mostrar mapa” en la barra izquierda para ver el mapa.")


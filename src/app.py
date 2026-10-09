"""Dash App Financiera: velas japonesas con indicadores técnicos (versión dockerizada)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, ctx, dcc, html
from plotly.subplots import make_subplots

RUTA_DATOS = Path(__file__).parent / "data" / "dash-stock-ticker-demo.csv"
TEMPLATE = "plotly_dark"
VERDE, ROJO = "#26a69a", "#ef5350"
COLOR = {"azul": "#4cc9f0", "ambar": "#f5b301", "rosa": "#f72585",
         "violeta": "#b388ff", "verde": "#80ed99", "gris": "#9fb3c8"}

ext_style = "https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/slate/bootstrap.min.css"
app = Dash(__name__, external_stylesheets=[ext_style], title="Terminal técnica | Velas")
server = app.server

# --------------------------------------------------------------------------- datos
df = pd.read_csv(RUTA_DATOS, index_col=0, parse_dates=["Date"])
df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")  # texto ISO: ordena igual y evita avisos de plotly 5.17 con pandas 2.1
TICKERS = sorted(df["Stock"].unique())


# --------------------------------------------------------------------------- indicadores
def sma(cierre, n):
    return cierre.rolling(n).mean()


def bollinger(cierre, n, desviaciones):
    media = cierre.rolling(n).mean()
    desv = cierre.rolling(n).std()
    return media, media + desviaciones * desv, media - desviaciones * desv


def rsi(cierre, n):
    cambio = cierre.diff()
    ganancia = cambio.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    perdida = (-cambio.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + ganancia / perdida)


def macd(cierre, rapida, lenta, senal):
    linea = cierre.ewm(span=rapida, adjust=False).mean() - cierre.ewm(span=lenta, adjust=False).mean()
    linea_senal = linea.ewm(span=senal, adjust=False).mean()
    return linea, linea_senal, linea - linea_senal


def estocastico(maximo, minimo, cierre, k, d):
    bajo = minimo.rolling(k).min()
    alto = maximo.rolling(k).max()
    pct_k = 100 * (cierre - bajo) / (alto - bajo)
    return pct_k, pct_k.rolling(d).mean()


def obv(cierre, volumen):
    return (np.sign(cierre.diff()).fillna(0) * volumen).cumsum()


def linea_ad(maximo, minimo, cierre, volumen):
    rango = (maximo - minimo).replace(0, np.nan)
    flujo = ((cierre - minimo) - (maximo - cierre)) / rango
    return (flujo.fillna(0) * volumen).cumsum()


def adx(maximo, minimo, cierre, n):
    sube = maximo.diff()
    baja = -minimo.diff()
    dm_mas = sube.where((sube > baja) & (sube > 0), 0.0)
    dm_menos = baja.where((baja > sube) & (baja > 0), 0.0)
    rango_real = pd.concat([maximo - minimo,
                            (maximo - cierre.shift()).abs(),
                            (minimo - cierre.shift()).abs()], axis=1).max(axis=1)

    def suavizar(serie):
        return serie.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    atr = suavizar(rango_real)
    di_mas = 100 * suavizar(dm_mas) / atr
    di_menos = 100 * suavizar(dm_menos) / atr
    dx = 100 * (di_mas - di_menos).abs() / (di_mas + di_menos)
    return suavizar(dx), di_mas, di_menos


def aroon(maximo, minimo, n):
    def ultimo(x, f):
        return x.size - 1 - f(x[::-1])

    arriba = 100 * maximo.rolling(n + 1).apply(lambda x: ultimo(x, np.argmax), raw=True) / n
    abajo = 100 * minimo.rolling(n + 1).apply(lambda x: ultimo(x, np.argmin), raw=True) / n
    return arriba - abajo


# --------------------------------------------------------------------------- dibujo de indicadores
# Los indicadores "sobre el precio" se dibujan en el mismo candlestick (fila 1);
# los osciladores usan un panel inferior de la MISMA figura, con el eje de tiempo compartido.
def dibujar_sma(fig, fila, d, p):
    fig.add_trace(go.Scatter(x=d["Date"], y=sma(d["Close"], p["periodo"]), mode="lines",
                             name=f"SMA({p['periodo']})", line=dict(width=1.6, color=COLOR["ambar"])),
                  row=1, col=1)


def dibujar_bb(fig, fila, d, p):
    media, sup, inf = bollinger(d["Close"], p["periodo"], p["desv"])
    linea = dict(width=1, color=COLOR["azul"])
    fig.add_trace(go.Scatter(x=d["Date"], y=sup, mode="lines", line=linea, legendgroup="bb",
                             name=f"Bollinger({p['periodo']}, {p['desv']}σ)", hoverinfo="skip"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=inf, mode="lines", line=linea, legendgroup="bb",
                             showlegend=False, fill="tonexty", fillcolor="rgba(76,201,240,0.08)",
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=media, mode="lines", legendgroup="bb", showlegend=False,
                             line=dict(width=1, color=COLOR["azul"], dash="dot"), hoverinfo="skip"),
                  row=1, col=1)


def dibujar_rsi(fig, fila, d, p):
    fig.add_trace(go.Scatter(x=d["Date"], y=rsi(d["Close"], p["periodo"]), mode="lines",
                             name=f"RSI({p['periodo']})", line=dict(width=1.5, color=COLOR["violeta"])),
                  row=fila, col=1)
    for nivel in (30, 70):
        fig.add_hline(y=nivel, line=dict(color="#888", dash="dot", width=1), row=fila, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=fila, col=1)


def dibujar_macd(fig, fila, d, p):
    linea, senal, hist = macd(d["Close"], p["rapida"], p["lenta"], p["senal"])
    colores = [VERDE if v >= 0 else ROJO for v in hist.fillna(0)]
    fig.add_trace(go.Bar(x=d["Date"], y=hist, name="Histograma MACD", marker_color=colores,
                         opacity=0.55), row=fila, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=linea, mode="lines", name=f"MACD({p['rapida']},{p['lenta']})",
                             line=dict(width=1.5, color=COLOR["azul"])), row=fila, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=senal, mode="lines", name=f"Señal({p['senal']})",
                             line=dict(width=1.5, color=COLOR["ambar"])), row=fila, col=1)
    fig.update_yaxes(title_text="MACD", row=fila, col=1)


def dibujar_stoch(fig, fila, d, p):
    pct_k, pct_d = estocastico(d["High"], d["Low"], d["Close"], p["k"], p["d"])
    fig.add_trace(go.Scatter(x=d["Date"], y=pct_k, mode="lines", name=f"%K({p['k']})",
                             line=dict(width=1.4, color=COLOR["azul"])), row=fila, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=pct_d, mode="lines", name=f"%D({p['d']})",
                             line=dict(width=1.4, color=COLOR["rosa"])), row=fila, col=1)
    for nivel in (20, 80):
        fig.add_hline(y=nivel, line=dict(color="#888", dash="dot", width=1), row=fila, col=1)
    fig.update_yaxes(title_text="Estocástico", range=[0, 100], row=fila, col=1)


def dibujar_obv(fig, fila, d, p):
    fig.add_trace(go.Scatter(x=d["Date"], y=obv(d["Close"], d["Volume"]), mode="lines", name="OBV",
                             line=dict(width=1.5, color=COLOR["verde"])), row=fila, col=1)
    fig.update_yaxes(title_text="OBV", row=fila, col=1)


def dibujar_ad(fig, fila, d, p):
    fig.add_trace(go.Scatter(x=d["Date"], y=linea_ad(d["High"], d["Low"], d["Close"], d["Volume"]),
                             mode="lines", name="Línea A/D", line=dict(width=1.5, color=COLOR["ambar"])),
                  row=fila, col=1)
    fig.update_yaxes(title_text="A/D", row=fila, col=1)


def dibujar_adx(fig, fila, d, p):
    valor, di_mas, di_menos = adx(d["High"], d["Low"], d["Close"], p["periodo"])
    fig.add_trace(go.Scatter(x=d["Date"], y=valor, mode="lines", name=f"ADX({p['periodo']})",
                             line=dict(width=2, color=COLOR["gris"])), row=fila, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=di_mas, mode="lines", name="+DI",
                             line=dict(width=1, color=VERDE)), row=fila, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=di_menos, mode="lines", name="-DI",
                             line=dict(width=1, color=ROJO)), row=fila, col=1)
    fig.add_hline(y=25, line=dict(color="#888", dash="dot", width=1), row=fila, col=1)
    fig.update_yaxes(title_text="ADX", row=fila, col=1)


def dibujar_aroon(fig, fila, d, p):
    fig.add_trace(go.Scatter(x=d["Date"], y=aroon(d["High"], d["Low"], p["periodo"]), mode="lines",
                             name=f"Aroon({p['periodo']})", fill="tozeroy",
                             line=dict(width=1.4, color=COLOR["rosa"]), fillcolor="rgba(247,37,133,0.15)"),
                  row=fila, col=1)
    fig.add_hline(y=0, line=dict(color="#888", dash="dot", width=1), row=fila, col=1)
    fig.update_yaxes(title_text="Aroon", range=[-100, 100], row=fila, col=1)


# clave -> nombre, ¿panel propio?, parámetros (nombre, etiqueta, opciones, valor por defecto), dibujo
INDICADORES = {
    "sma": dict(nombre="Media móvil (SMA)", panel=False, dibujo=dibujar_sma,
                params=[("periodo", "Periodo", [5, 10, 20, 50, 100, 200], 20)]),
    "bb": dict(nombre="Bandas de Bollinger", panel=False, dibujo=dibujar_bb,
               params=[("periodo", "Periodo de la media móvil", [10, 20, 30, 50], 20),
                       ("desv", "Desviaciones estándar", [1, 2, 3, 4, 5], 2)]),
    "rsi": dict(nombre="RSI", panel=True, dibujo=dibujar_rsi,
                params=[("periodo", "Periodo", [7, 14, 21], 14)]),
    "macd": dict(nombre="MACD", panel=True, dibujo=dibujar_macd,
                 params=[("rapida", "EMA rápida", [8, 12, 16], 12),
                         ("lenta", "EMA lenta", [21, 26, 30], 26),
                         ("senal", "Señal", [5, 9, 12], 9)]),
    "stoch": dict(nombre="Oscilador estocástico", panel=True, dibujo=dibujar_stoch,
                  params=[("k", "Periodo %K", [5, 9, 14, 21], 14),
                          ("d", "Suavizado %D", [3, 5], 3)]),
    "obv": dict(nombre="On-balance volume (OBV)", panel=True, dibujo=dibujar_obv, params=[]),
    "ad": dict(nombre="Línea de acumulación/distribución (A/D)", panel=True, dibujo=dibujar_ad, params=[]),
    "adx": dict(nombre="Average directional index (ADX)", panel=True, dibujo=dibujar_adx,
                params=[("periodo", "Periodo", [7, 14, 21, 28], 14)]),
    "aroon": dict(nombre="Oscilador Aroon", panel=True, dibujo=dibujar_aroon,
                  params=[("periodo", "Periodo", [14, 25, 50], 25)]),
}
POR_DEFECTO = ["bb", "rsi", "macd", "stoch"]
MINIMO_CONFIRMACIONES = 4


def construir_figura(d, seleccion, p):
    """Un solo plot: el candlestick con todos los indicadores elegidos."""
    activos = [k for k in INDICADORES if k in seleccion]
    paneles = [k for k in activos if INDICADORES[k]["panel"]]
    pesos = [3.0] + [1.0] * len(paneles)
    fig = make_subplots(rows=1 + len(paneles), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[w / sum(pesos) for w in pesos])
    fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Precio",
                                 increasing_line_color=VERDE, decreasing_line_color=ROJO),
                  row=1, col=1)

    fila = 1
    for clave in activos:
        conf = INDICADORES[clave]
        parametros = {nombre: p[(clave, nombre)] for nombre, *_ in conf["params"]}
        if conf["panel"]:
            fila += 1
        conf["dibujo"](fig, fila if conf["panel"] else 1, d, parametros)

    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440 + 170 * len(paneles),
        margin=dict(l=60, r=20, t=40, b=30),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=False)
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
    return fig


# --------------------------------------------------------------------------- layout
def caja_parametros(clave, conf):
    return html.Div(id={"caja": clave}, className="col-lg-3 col-md-4 mb-3", style={"display": "none"}, children=[
        html.Div(className="caja-param", children=[
            html.H6(conf["nombre"], className="titulo-param"),
            *[html.Div(className="mb-2", children=[
                html.Label(etiqueta, className="etiqueta-control"),
                dcc.Dropdown(id={"ind": clave, "param": nombre},
                             options=[{"label": str(o), "value": o} for o in opciones],
                             value=defecto, clearable=False),
            ]) for nombre, etiqueta, opciones, defecto in conf["params"]],
        ]),
    ])


app.layout = html.Div(className="fondo-app", children=[
    html.Div(className="encabezado", children=[
        html.Img(src=app.get_asset_url("logo_velas.svg"), className="logo"),
        html.Div([
            html.H2("Terminal de Análisis Técnico", className="titulo-app"),
            html.P("Velas japonesas con indicadores para decidir long o short", className="subtitulo-app"),
        ]),
    ]),

    html.Div(className="container-fluid px-4", children=[
        html.Div(className="row panel-controles", children=[
            html.Div(className="col-lg-5 col-md-12 mb-2", children=[
                html.Label("Acciones (tickers)", className="etiqueta-control"),
                dcc.Dropdown(
                    id="stock-ticker-input",
                    options=[{"label": s, "value": s} for s in TICKERS],
                    value=["YHOO", "GOOGL"],
                    multi=True,
                ),
            ]),
            html.Div(className="col-lg-7 col-md-12 mb-2", children=[
                html.Label("Indicadores de análisis técnico (elige al menos 4)", className="etiqueta-control"),
                dcc.Dropdown(
                    id="indicadores",
                    options=[{"label": c["nombre"], "value": k} for k, c in INDICADORES.items()],
                    value=POR_DEFECTO,
                    multi=True,
                ),
            ]),
        ]),

        html.Div(className="row", children=[
            caja_parametros(clave, conf) for clave, conf in INDICADORES.items() if conf["params"]
        ]),

        html.Div(id="graphs"),
        html.P("Datos: dash-stock-ticker-demo.csv · Dash + Plotly · contenedor Docker", className="pie-pagina"),
    ]),
])


# --------------------------------------------------------------------------- callbacks
@app.callback(Output({"caja": ALL}, "style"), Input("indicadores", "value"))
def mostrar_parametros(seleccion):
    seleccion = seleccion or []
    return [{"display": "block"} if salida["id"]["caja"] in seleccion else {"display": "none"}
            for salida in ctx.outputs_list]


@app.callback(
    Output("graphs", "children"),
    [Input("stock-ticker-input", "value"),
     Input("indicadores", "value"),
     Input({"ind": ALL, "param": ALL}, "value")]
)
def update_graph(tickers, seleccion, _valores):
    seleccion = seleccion or []
    p = {(entrada["id"]["ind"], entrada["id"]["param"]): entrada["value"]
         for entrada in ctx.inputs_list[2]}
    graphs = []

    if not tickers:
        graphs.append(html.H3("Selecciona un ticker.", style={"marginTop": 20, "marginBottom": 20}))
        return graphs

    if len(seleccion) < MINIMO_CONFIRMACIONES:
        graphs.append(html.Div(
            f"Hay {len(seleccion)} indicador(es) activo(s): elige al menos {MINIMO_CONFIRMACIONES} "
            "para tener 4 confirmaciones antes de decidir la compra o la venta.",
            className="aviso"))

    for ticker in tickers:
        dff = df[df["Stock"] == ticker]
        graphs.append(html.Div(className="card tarjeta mb-4", children=[
            html.Div(className="card-header", children=[html.Span(ticker, className="titulo-tarjeta")]),
            html.Div(className="card-body p-2", children=[
                dcc.Graph(id=f"grafico-{ticker}", figure=construir_figura(dff, seleccion, p)),
            ]),
        ]))

    return graphs


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))  # Render define PORT

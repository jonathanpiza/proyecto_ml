"""Dashboard: predicción del mes siguiente. Consulta la API (variable de entorno API_URL)."""
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
ING = "ingenuo (repite mes anterior)"
NOMBRES = {ING: "Ingenuo (repite el mes anterior)", "lineal": "Regresión lineal", "random_forest": "Random Forest"}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

st.set_page_config(page_title="Predicción del mes siguiente", layout="wide")


# ------------------------------------------------------------------ utilidades
def fmt(v):
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.1f}" if abs(v) >= 10 else (f"{v:,.2f}" if abs(v) >= 1 else f"{v:,.3f}")


def pct(v):
    return "n/d" if v is None or pd.isna(v) else f"{v:.1%}"


def nombre_modelo(m):
    return NOMBRES.get(m, m)


def etiqueta_mes(m):
    return f"{m:02d} {MESES[m - 1]}"          # "01 ene": así las barras salen en orden de calendario


def pedir(ruta, opcional=False, **params):
    """Consulta la API. Si falla, muestra el motivo y detiene el dashboard (salvo si es opcional)."""
    try:
        r = requests.get(f"{API_URL}{ruta}", params=params, timeout=10)
    except requests.RequestException:
        st.error(f"No se pudo conectar con la API en {API_URL}. ¿Está encendida?")
        st.stop()
    if r.status_code != 200:
        if opcional:
            return None
        try:
            detalle = r.json().get("detail", r.text)
        except ValueError:
            detalle = r.text
        st.error(f"La API respondió con un error: {detalle}")
        st.stop()
    return r.json()


def error_tipico(res):
    """MAPE del modelo que de verdad se usa para predecir (en el total o en un grupo)."""
    if "metricas_prueba" in res:
        return (res["metricas_prueba"].get(res["modelo_usado"]) or {}).get("MAPE")
    return res["MAPE_ingenuo"] if res["modelo_usado"] == ING else res["MAPE_modelo"]


def cargar_serie(grupo=None):
    df = pd.DataFrame(pedir("/serie", **({"grupo": grupo} if grupo else {})))
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.set_index("fecha")["valor"]


# ------------------------------------------------------------------ barra lateral
grupos = pedir("/grupos")["grupos"]

st.sidebar.header("Qué quieres ver")
opcion = st.sidebar.selectbox("Selecciona", ["Total"] + grupos,
                              help="Total suma todo. Los demás son cada grupo por separado.") if grupos else "Total"
ventana = st.sidebar.radio("Periodo del histórico", ["Últimos 12 meses", "Últimos 24 meses", "Todo"], index=2)
mostrar_media = st.sidebar.checkbox("Mostrar promedio móvil (3 meses)", value=True,
                                    help="Suaviza los saltos de cada mes para ver la tendencia.")
with st.sidebar.expander("¿Cómo leer este tablero?"):
    st.markdown(
        "- **Predicción:** el valor que se espera para el mes siguiente.\n"
        "- **Error típico:** cuánto se equivocó el modelo, en promedio, en meses que no conocía. Menor es mejor.\n"
        "- **Ingenuo:** modelo de referencia que repite el valor del mes anterior. Un buen modelo debe superarlo.\n"
        "- **Meses de prueba:** los últimos meses, que el modelo no vio al entrenar."
    )

# ------------------------------------------------------------------ datos principales
params = {} if opcion == "Total" else {"grupo": opcion}
res = pedir("/predict", **params)
serie = cargar_serie(None if opcion == "Total" else opcion)
meta = res if opcion == "Total" else pedir("/predict")        # datos generales (cómo se agregó, por qué columna se agrupó)
plural = {"tipo": "tipos", "producto": "productos", "region": "regiones"}.get(meta.get("grupo_por"), "grupos")
nombre_comp = f"Comparar {plural}"
suma_total = meta.get("agg") in ("sum", "count")             # solo tiene sentido hablar de "participación" si los valores se suman

ultimo = float(serie.iloc[-1])
pred = float(res["prediccion"])
cambio = (pred - ultimo) / ultimo if ultimo else None
usado = res["modelo_usado"]
error = error_tipico(res)
mes = res["mes_predicho"]

# ------------------------------------------------------------------ encabezado
st.title("Predicción del mes siguiente")
st.caption(f"Viendo: **{opcion}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Predicción {mes}", fmt(pred), delta=None if cambio is None else f"{cambio:+.1%}",
          help="El cambio es frente al último mes real.")
c2.metric(f"Último mes real ({serie.index[-1]:%Y-%m})", fmt(ultimo))
c3.metric("Error típico del modelo", pct(error), help="Promedio de cuánto se equivocó en los meses de prueba.")
c4.metric("Modelo usado", nombre_modelo(usado))

if cambio is None:
    texto = f"Para **{mes}** se espera **{fmt(pred)}**."
else:
    sentido = "más" if cambio >= 0 else "menos"
    texto = f"Para **{mes}** se espera **{fmt(pred)}**, un **{abs(cambio):.1%} {sentido}** que el último mes real ({fmt(ultimo)})."
if error is not None:
    texto += f" En los meses de prueba, el modelo se equivocó en promedio un **{pct(error)}**."
if usado == ING:
    texto += " Ningún modelo superó al ingenuo, así que se usa la repetición del mes anterior."
st.info(texto)

# ------------------------------------------------------------------ pestañas
nombres = ["Pronóstico", "Patrones", "Precisión del modelo"] + ([nombre_comp] if grupos else []) + ["Datos"]
pestana = dict(zip(nombres, st.tabs(nombres)))

# ---- 1. Pronóstico
with pestana["Pronóstico"]:
    n = {"Últimos 12 meses": 12, "Últimos 24 meses": 24}.get(ventana)
    recorte = serie.iloc[-n:] if n else serie
    proximo = pd.to_datetime(mes + "-01")
    union = pd.Series({serie.index[-1]: ultimo, proximo: pred})      # une el último mes real con la predicción
    grafica = pd.concat({"Histórico": recorte, "Predicción": union}, axis=1)
    if mostrar_media:
        grafica["Promedio móvil (3 meses)"] = serie.rolling(3).mean().reindex(grafica.index)
    st.subheader("Histórico y predicción")
    st.line_chart(grafica)
    st.caption("El tramo «Predicción» une el último mes real con el valor esperado para el mes siguiente.")

# ---- 2. Patrones
with pestana["Patrones"]:
    a, b = st.columns(2)
    por_mes = serie.groupby(serie.index.month).mean()
    a.subheader("Promedio por mes del año")
    a.caption("Qué meses suelen ser más altos o más bajos (temporada).")
    a.bar_chart(pd.DataFrame({"Promedio": por_mes.values}, index=[etiqueta_mes(m) for m in por_mes.index]))

    variacion = (serie.pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100)
    b.subheader("Variación frente al mes anterior (%)")
    b.caption("Cuánto subió o bajó cada mes respecto al anterior.")
    b.bar_chart(pd.DataFrame({"Variación %": variacion.values}, index=variacion.index))

    st.subheader("Comparación año contra año")
    st.caption("Cada línea es un año: permite ver si este año va por encima o por debajo de los anteriores.")
    tabla = serie.to_frame("valor")
    tabla["anio"], tabla["mes"] = tabla.index.year, tabla.index.month
    anios = tabla.pivot(index="mes", columns="anio", values="valor")
    anios.index = [etiqueta_mes(m) for m in anios.index]
    anios.columns = anios.columns.astype(str)
    st.line_chart(anios)

# ---- 3. Precisión del modelo
with pestana["Precisión del modelo"]:
    prueba = pedir("/prueba", opcional=True, **params)
    if prueba is None:
        st.warning("Faltan los datos de los meses de prueba. Vuelve a ejecutar src/train.py para generarlos.")
    else:
        p = pd.DataFrame(prueba)
        p["fecha"] = pd.to_datetime(p["fecha"])
        p = p.set_index("fecha").rename(columns={"real": "Real", "ingenuo": "Ingenuo", "lineal": "Regresión lineal",
                                                 "random_forest": "Random Forest"})
        st.subheader("Real vs predicho en los meses de prueba")
        st.caption("Son meses que el modelo no vio al entrenar. Cuanto más cerca de la línea «Real», mejor.")
        st.line_chart(p)

    if opcion == "Total":
        m = pd.DataFrame(res["metricas_prueba"]).T.rename(index=NOMBRES)
        a, b = st.columns(2)
        mape = pd.to_numeric(m["MAPE"], errors="coerce") * 100
        a.subheader("Error en porcentaje (MAPE)")
        if mape.notna().any():
            a.bar_chart(mape.to_frame("MAPE %"))
        else:
            a.write("No se puede calcular el porcentaje cuando hay meses con valor 0.")
        b.subheader("Error promedio (MAE)")
        b.bar_chart(pd.to_numeric(m["MAE"], errors="coerce").to_frame("MAE"))
        st.dataframe(pd.DataFrame({"MAE": m["MAE"].map(fmt), "RMSE": m["RMSE"].map(fmt), "MAPE": m["MAPE"].map(pct),
                                   "R2": m["R2"].map(lambda v: "n/d" if pd.isna(v) else f"{v:.2f}")}))
    else:
        if res["MAPE_modelo"] is not None and res["MAPE_ingenuo"] is not None:
            st.subheader("Error en porcentaje (MAPE): modelo vs ingenuo")
            st.bar_chart(pd.DataFrame({"MAPE %": [res["MAPE_ingenuo"] * 100, res["MAPE_modelo"] * 100]},
                                      index=["Ingenuo", nombre_modelo(res["mejor_modelo"])]))
        else:
            st.write("No se puede calcular el porcentaje de error cuando hay meses con valor 0.")

    with st.expander("¿Qué significan estas métricas?"):
        st.markdown(
            "- **MAE:** error promedio, en las mismas unidades de tus datos.\n"
            "- **RMSE:** parecido al MAE, pero castiga más los errores grandes.\n"
            "- **MAPE:** error promedio en porcentaje.\n"
            "- **R2:** con pocos meses de prueba no es confiable; guíate por MAE, RMSE y MAPE."
        )

# ---- 4. Comparar grupos
if grupos:
    with pestana[nombre_comp]:
        resumen = pedir("/resumen", opcional=True)
        wide = pedir("/series", opcional=True)
        if resumen is None or wide is None:
            st.warning(f"Faltan datos para comparar {plural}. Vuelve a ejecutar src/train.py con --grupo.")
        else:
            r = pd.DataFrame(resumen).set_index("grupo")
            series = pd.DataFrame(wide)
            series["fecha"] = pd.to_datetime(series["fecha"])
            series = series.set_index("fecha")

            elegidos = st.multiselect(f"Elige qué {plural} comparar", grupos, default=grupos[:5],
                                      help="Todas las gráficas de esta pestaña usan solo lo que elijas aquí.")
            if not elegidos:
                st.info(f"Elige al menos uno de los {plural} para ver la comparación.")
            else:
                n_comp = {"Últimos 12 meses": 12, "Últimos 24 meses": 24}.get(ventana)
                vent = series.iloc[-n_comp:] if n_comp else series
                sel = vent[elegidos]

                # --- fila 1: evolución
                a, b = st.columns(2)
                a.subheader("Evolución de cada uno")
                a.caption("Valores reales mes a mes.")
                a.line_chart(sel)
                base = sel.apply(lambda col: col / col.iloc[0] * 100 if col.iloc[0] else np.nan)
                b.subheader("Crecimiento relativo (base 100)")
                b.caption("Todos parten de 100 en el primer mes del periodo: así se ve quién crece más aunque tengan tamaños distintos.")
                b.line_chart(base.replace([np.inf, -np.inf], np.nan))

                # --- fila 2: participación y temporada
                a, b = st.columns(2)
                if suma_total:
                    total = series.sum(axis=1).reindex(vent.index)
                    a.subheader("Participación en el total (%)")
                    a.caption(f"Qué parte del total aporta cada uno de los {plural} elegidos en cada mes.")
                    a.area_chart(sel.div(total, axis=0) * 100)
                else:
                    a.subheader("Diferencia entre ellos")
                    a.caption("La participación no aplica porque los valores son promedios y no se suman.")
                por_mes = series[elegidos].groupby(series.index.month).mean()
                relativo = por_mes / por_mes.mean() * 100
                relativo.index = [etiqueta_mes(m) for m in relativo.index]
                b.subheader("Temporada de cada uno")
                b.caption("100 = el promedio de ese grupo. Por encima de 100, un mes más fuerte de lo normal.")
                b.line_chart(relativo)

                # --- fila 3: pronóstico
                a, b = st.columns(2)
                a.subheader("Predicción del mes siguiente")
                a.bar_chart(r.loc[elegidos, ["prediccion"]].rename(columns={"prediccion": "Predicción"}))
                b.subheader("Cambio esperado frente al último mes (%)")
                b.bar_chart((pd.to_numeric(r.loc[elegidos, "cambio"], errors="coerce") * 100).to_frame("Cambio %"))

                # --- fila 4: error
                err = pd.DataFrame({"Mejor modelo": pd.to_numeric(r.loc[elegidos, "MAPE_modelo"], errors="coerce") * 100,
                                    "Ingenuo": pd.to_numeric(r.loc[elegidos, "MAPE_ingenuo"], errors="coerce") * 100})
                st.subheader("Error del modelo vs ingenuo (MAPE %)")
                st.caption("Menor es mejor. Si la barra del modelo no es más baja que la del ingenuo, ese caso es difícil de predecir.")
                if err.notna().any().any():
                    st.bar_chart(err)
                else:
                    st.write("No se puede calcular el porcentaje de error cuando hay meses con valor 0.")

                # --- cuadro comparativo
                filas = {}
                for g in elegidos:
                    s_g = series[g]
                    ult12 = s_g.iloc[-12:].mean()
                    prev12 = s_g.iloc[-24:-12].mean() if len(s_g) >= 24 else np.nan
                    crecimiento = ult12 / prev12 - 1 if prev12 and not np.isnan(prev12) else None
                    rg = r.loc[g]
                    filas[g] = {
                        "Último mes": fmt(s_g.iloc[-1]),
                        "Promedio 12 meses": fmt(ult12),
                        "Crecimiento anual": pct(crecimiento),
                        "Predicción": fmt(rg["prediccion"]),
                        "Cambio esperado": pct(rg["cambio"]),
                        "Modelo usado": nombre_modelo(rg["modelo_usado"]).split(" (")[0],
                        "Error (MAPE)": pct(error_tipico(rg.to_dict())),
                    }
                st.subheader("Cuadro comparativo")
                st.caption("Crecimiento anual: promedio de los últimos 12 meses frente al de los 12 anteriores.")
                st.dataframe(pd.DataFrame(filas).T)

# ---- 5. Datos
with pestana["Datos"]:
    st.subheader(f"Serie mensual: {opcion}")
    datos = serie.rename("valor").to_frame()
    datos.index = datos.index.strftime("%Y-%m")
    st.dataframe(datos)
    st.download_button("Descargar CSV", datos.to_csv().encode("utf-8"), file_name=f"serie_{opcion}.csv", mime="text/csv")

st.caption(f"Datos servidos por la API en {API_URL}")

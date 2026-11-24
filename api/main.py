"""API de predicción del mes siguiente.

Lee los archivos que genera src/train.py en la carpeta salida/ y los entrega por HTTP:
  GET /          estado de la API
  GET /predict   predicción del total  (de un grupo: /predict?grupo=Zapatos)
  GET /serie     histórico mensual del total  (de un grupo: /serie?grupo=Zapatos)
  GET /grupos    lista de grupos disponibles (vacía si entrenaste sin --grupo)
  GET /prueba    real vs predicho en los meses de prueba  (de un grupo: /prueba?grupo=Zapatos)
  GET /resumen   una fila por grupo: último mes, predicción, cambio y error
  GET /series    histórico mensual de todos los grupos juntos (una columna por grupo)
  GET /docs      documentación automática: desde ahí puedes probar cada ruta
"""
import json
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

SALIDA = os.getenv("SALIDA", "salida")
app = FastAPI(title="API de predicción del mes siguiente")


def ruta(nombre):
    return os.path.join(SALIDA, nombre)


def exigir(nombre):
    """Falla con un mensaje claro si todavía no existe un archivo de salida/."""
    if not os.path.exists(ruta(nombre)):
        raise HTTPException(status_code=503, detail=f"Falta {nombre} en {SALIDA}/. Ejecuta primero src/train.py.")


def leer_json(nombre):
    exigir(nombre)
    with open(ruta(nombre), encoding="utf-8") as f:
        return json.load(f)


def datos_grupos():
    if not os.path.exists(ruta("resultado_por_grupo.json")):
        return {"grupos": {}}
    return leer_json("resultado_por_grupo.json")


def validar_grupo(grupo):
    datos = datos_grupos()
    if grupo not in datos["grupos"]:
        raise HTTPException(status_code=404,
                            detail=f"El grupo '{grupo}' no existe. Disponibles: {list(datos['grupos'])}")
    return datos


@app.get("/")
def estado():
    return {"estado": "ok", "rutas": ["/predict", "/serie", "/prueba", "/grupos", "/resumen", "/series", "/docs"]}


@app.get("/predict")
def predict(grupo: Optional[str] = None):
    if grupo is None:
        return leer_json("resultado.json")
    datos = validar_grupo(grupo)
    return {"grupo": grupo, "mes_predicho": datos["mes_predicho"], **datos["grupos"][grupo]}


@app.get("/grupos")
def grupos():
    return {"grupos": list(datos_grupos()["grupos"])}


@app.get("/serie")
def serie(grupo: Optional[str] = None):
    if grupo is None:
        exigir("serie_mensual.csv")
        df = pd.read_csv(ruta("serie_mensual.csv"))
        df.columns = ["fecha", "valor"]
    else:
        validar_grupo(grupo)
        exigir("series_por_grupo.csv")
        s = pd.read_csv(ruta("series_por_grupo.csv"), index_col=0)[grupo]
        df = pd.DataFrame({"fecha": s.index, "valor": s.values})
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/prueba")
def prueba(grupo: Optional[str] = None):
    if grupo is None:
        exigir("prueba.csv")
        df = pd.read_csv(ruta("prueba.csv"))
    else:
        validar_grupo(grupo)
        exigir("prueba_por_grupo.csv")
        df = pd.read_csv(ruta("prueba_por_grupo.csv"))
        df = df[df["grupo"].astype(str) == grupo].drop(columns="grupo")
    df.columns = ["fecha", "real", "ingenuo", "lineal", "random_forest"]
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/resumen")
def resumen():
    datos = datos_grupos()
    if not datos["grupos"]:
        return []
    exigir("series_por_grupo.csv")
    series = pd.read_csv(ruta("series_por_grupo.csv"), index_col=0)
    filas = []
    for g, v in datos["grupos"].items():
        ultimo = float(series[g].iloc[-1])
        filas.append({"grupo": g, "ultimo": ultimo, "prediccion": v["prediccion"],
                      "cambio": (v["prediccion"] - ultimo) / ultimo if ultimo else None,
                      "modelo_usado": v["modelo_usado"],
                      "MAPE_modelo": v["MAPE_modelo"], "MAPE_ingenuo": v["MAPE_ingenuo"]})
    return filas


@app.get("/series")
def series_de_grupos():
    if not datos_grupos()["grupos"]:
        return []
    exigir("series_por_grupo.csv")
    df = pd.read_csv(ruta("series_por_grupo.csv"))
    df.columns = ["fecha"] + list(df.columns[1:])
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")

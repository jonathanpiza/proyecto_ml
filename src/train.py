import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["lag1", "lag2", "lag3", "prom3", "mes"]


def datos_demo():
    """36 meses de datos sintéticos para probar el flujo sin datos reales."""
    rng = np.random.default_rng(42)
    meses = pd.date_range("2023-01-01", periods=36, freq="MS")
    valor = 1000 + np.arange(36) * 15 + 120 * np.sin(2 * np.pi * meses.month / 12) + rng.normal(0, 40, 36)
    return pd.DataFrame({"fecha": meses, "valor": valor.round(2)})


def serie_mensual(df, fecha, valor, agg):
    """Convierte cualquier tabla con fechas en una serie mensual (suma, promedio o conteo)."""
    df = df.copy()
    df[fecha] = pd.to_datetime(df[fecha], errors="coerce")
    df = df.dropna(subset=[fecha]).set_index(fecha)
    if agg == "count":
        s = df.resample("MS").size()
    else:
        s = df[valor].resample("MS").agg(agg)
    return s.fillna(0).astype(float)


def hacer_features(s):
    """Variables: valor de 1, 2 y 3 meses atrás, promedio de esos 3 y número de mes."""
    d = pd.DataFrame({"y": s})
    for k in (1, 2, 3):
        d[f"lag{k}"] = s.shift(k)
    d["prom3"] = d[["lag1", "lag2", "lag3"]].mean(axis=1)
    d["mes"] = d.index.month
    return d.dropna()


def metricas(real, pred):
    return {
        "MAE": float(mean_absolute_error(real, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(real, pred))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="archivo CSV con los datos")
    p.add_argument("--fecha", default="fecha", help="columna de fecha")
    p.add_argument("--valor", default="valor", help="columna numérica (no se usa con --agg count)")
    p.add_argument("--agg", default="sum", choices=["sum", "count", "mean"])
    p.add_argument("--test-meses", type=int, default=3, help="meses reservados para evaluar")
    p.add_argument("--salida", default="modelo", help="carpeta donde guardar modelo, serie y resultado")
    p.add_argument("--demo", action="store_true", help="usar datos sintéticos")
    a = p.parse_args()

    if not a.demo and not a.csv:
        raise SystemExit("Indica --csv archivo.csv o usa --demo")

    df = datos_demo() if a.demo else pd.read_csv(a.csv)
    s = serie_mensual(df, a.fecha, a.valor, a.agg)
    if len(s) < 12:
        raise SystemExit(f"Se necesitan al menos 12 meses de datos (hay {len(s)}).")

    d = hacer_features(s)
    train, test = d.iloc[: -a.test_meses], d.iloc[-a.test_meses:]

    modelos = {
        "lineal": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    resultados = {"ingenuo (repite mes anterior)": metricas(test["y"], test["lag1"])}
    for nombre, m in modelos.items():
        m.fit(train[FEATURES], train["y"])
        resultados[nombre] = metricas(test["y"], m.predict(test[FEATURES]))

    mejor = min(modelos, key=lambda n: resultados[n]["MAE"])
    final = modelos[mejor].fit(d[FEATURES], d["y"])  # reentrena con todos los datos

    proximo = s.index[-1] + pd.offsets.MonthBegin(1)
    x = pd.DataFrame([{
        "lag1": s.iloc[-1], "lag2": s.iloc[-2], "lag3": s.iloc[-3],
        "prom3": s.iloc[-3:].mean(), "mes": proximo.month,
    }])
    prediccion = float(final.predict(x[FEATURES])[0])

    os.makedirs(a.salida, exist_ok=True)
    joblib.dump(final, os.path.join(a.salida, "modelo.joblib"))
    s.rename("valor").to_csv(os.path.join(a.salida, "serie_mensual.csv"))
    resultado = {
        "mejor_modelo": mejor,
        "metricas_prueba": resultados,
        "mes_predicho": proximo.strftime("%Y-%m"),
        "prediccion": round(prediccion, 2),
    }
    with open(os.path.join(a.salida, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Entrena modelos para predecir el valor del mes siguiente (ventas, gastos, PQRS...).

Ejemplos (desde la raíz del proyecto):
  python src/train.py --demo
  python src/train.py --csv data/ventas_demo.csv --fecha fecha --valor monto --agg sum
  python src/train.py --csv data/ventas_demo.csv --fecha fecha --valor monto --agg sum --grupo producto
  python src/train.py --csv data/pqrs_demo.csv --fecha fecha_radicado --agg count --grupo tipo
  python src/train.py --csv data/avocado.csv --fecha Date --valor AveragePrice --agg mean --grupo region

Qué hace:
  1. Limpia los datos (duplicados, fechas inválidas, valores vacíos, negativos y atípicos).
  2. Agrupa por mes y crea las variables: 3 meses atrás, su promedio y el número de mes.
  3. Parte los meses POR FECHA (sin mezclar): lo más viejo entrena y lo último se evalúa.
  4. Compara el modelo ingenuo (repite el mes anterior), la regresión lineal y Random Forest.
  5. Guarda el mejor y predice el mes siguiente. Si ningún modelo supera al ingenuo, usa el ingenuo.
  6. Con --grupo repite todo para cada producto, tipo, región...

Salida (carpeta --salida):
  modelo.joblib, serie_mensual.csv, resultado.json                (total)
  modelos_por_grupo.joblib, resultado_por_grupo.json, series_por_grupo.csv   (solo con --grupo)
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:                      # scikit-learn anterior a 1.4
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(real, est):
        return float(np.sqrt(mean_squared_error(real, est)))

VARIABLES = ["lag1", "lag2", "lag3", "prom3", "mes"]
ING = "ingenuo (repite mes anterior)"


def datos_demo():
    """36 meses de datos sintéticos para probar el flujo sin datos reales."""
    rng = np.random.default_rng(42)
    meses = pd.date_range("2023-01-01", periods=36, freq="MS")
    valor = 1000 + np.arange(36) * 15 + 120 * np.sin(2 * np.pi * meses.month / 12) + rng.normal(0, 40, 36)
    return pd.DataFrame({"fecha": meses, "valor": valor.round(2)})


def limpiar(df, fecha, valor, grupo):
    """Quita duplicados, fechas inválidas y valores malos. Devuelve la tabla limpia y el informe."""
    informe = {"Filas originales": len(df), "Filas duplicadas": int(df.duplicated().sum())}
    df = df.drop_duplicates()

    df[fecha] = pd.to_datetime(df[fecha], errors="coerce")      # si tus fechas son día/mes/año: dayfirst=True
    informe["Fechas inválidas o vacías"] = int(df[fecha].isna().sum())
    df = df.dropna(subset=[fecha])

    if valor:
        df[valor] = pd.to_numeric(df[valor], errors="coerce")
        informe["Valores vacíos"] = int(df[valor].isna().sum())
        df = df.dropna(subset=[valor])
        informe["Valores negativos o en cero"] = int((df[valor] <= 0).sum())
        df = df[df[valor] > 0]
        tope = 20 * df[valor].median()                           # más de 20 veces la mediana: error de digitación
        informe["Valores atípicos (más de 20 veces la mediana)"] = int((df[valor] > tope).sum())
        df = df[df[valor] <= tope]

    if grupo:
        informe["Filas sin grupo (quedan como 'Sin dato')"] = int(df[grupo].isna().sum())
        df[grupo] = df[grupo].fillna("Sin dato")

    informe["Filas finales"] = len(df)
    return df, informe


def serie_mensual(datos, agg, valor, meses):
    """Una serie con un dato por mes. 'datos' debe tener la fecha como índice."""
    if agg == "count":
        s = datos.resample("MS").size()
    else:
        s = datos[valor].resample("MS").agg(agg)
    s = s.reindex(meses)
    s = s.fillna(0) if agg in ("sum", "count") else s.ffill().bfill()   # mes sin datos: 0 al sumar o contar
    return s.astype(float)


def crear_variables(s):
    """Variables: valor de 1, 2 y 3 meses atrás, promedio de esos 3 y número de mes."""
    d = pd.DataFrame({"y": s})
    for k in (1, 2, 3):
        d[f"lag{k}"] = s.shift(k)
    d["prom3"] = d[["lag1", "lag2", "lag3"]].mean(axis=1)
    d["mes"] = d.index.month
    return d.dropna()


def metricas(real, est):
    con_ceros = bool((np.asarray(real) == 0).any())       # con meses en 0 el MAPE no se puede calcular
    return {"MAE": mean_absolute_error(real, est),
            "RMSE": root_mean_squared_error(real, est),
            "MAPE": np.nan if con_ceros else mean_absolute_percentage_error(real, est),
            "R2": r2_score(real, est)}


def ajustar(s, particion):
    """Entrena, evalúa y predice el mes siguiente para UNA serie."""
    d = crear_variables(s)
    x, y = d[VARIABLES], d["y"]
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=False, **particion)

    modelos = {"lineal": LinearRegression(),
               "random_forest": RandomForestRegressor(n_estimators=200, random_state=42)}
    preds = {ING: X_test["lag1"]}
    for nombre, m in modelos.items():
        m.fit(X_train, y_train)
        preds[nombre] = pd.Series(m.predict(X_test), index=X_test.index)

    num = pd.DataFrame({n: metricas(y_test, p) for n, p in preds.items()}).T
    mae_ing = num.loc[ING, "MAE"]
    mejor = min(modelos, key=lambda n: num.loc[n, "MAE"])
    usado = mejor if num.loc[mejor, "MAE"] < mae_ing else ING

    final = modelos[mejor].fit(x, y)                       # reentrena con todos los datos
    proximo = s.index[-1] + pd.offsets.MonthBegin(1)
    x_nuevo = pd.DataFrame([{"lag1": s.iloc[-1], "lag2": s.iloc[-2], "lag3": s.iloc[-3],
                             "prom3": s.iloc[-3:].mean(), "mes": proximo.month}])[VARIABLES]
    pred_modelo = float(final.predict(x_nuevo)[0])
    prediccion = pred_modelo if usado != ING else float(s.iloc[-1])
    return {"num": num, "mejor": mejor, "usado": usado, "final": final, "proximo": proximo,
            "prediccion": prediccion, "n_train": len(X_train), "n_test": len(X_test),
            "inicio_test": X_test.index[0], "fin_test": X_test.index[-1]}


def f_(v):
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:,.1f}" if abs(v) >= 10 else f"{v:,.3f}")


def pct(v):
    return "n/d" if pd.isna(v) else f"{v:.1%}"


def num_o_none(v):
    return None if pd.isna(v) else float(v)


def main():
    p = argparse.ArgumentParser(description="Predice el valor del mes siguiente (total y por grupo).")
    p.add_argument("--csv", help="archivo CSV con los datos")
    p.add_argument("--fecha", default="fecha", help="columna de fecha")
    p.add_argument("--valor", help="columna numérica (no se usa con --agg count)")
    p.add_argument("--agg", default="sum", choices=["sum", "count", "mean"],
                   help="sum: ventas y gastos; count: PQRS; mean: promedios (precios)")
    p.add_argument("--grupo", help="columna para predecir por grupo (producto, tipo, región...)")
    p.add_argument("--train-size", type=float, default=0.7, help="proporción de meses para entrenar (por defecto 0.7)")
    p.add_argument("--test-meses", type=int, help="si lo indicas (ej. 3), usa los últimos N meses para la prueba")
    p.add_argument("--sin-limpieza", action="store_true", help="no limpiar los datos (ya vienen limpios)")
    p.add_argument("--salida", default="salida")
    p.add_argument("--demo", action="store_true", help="usar datos sintéticos")
    a = p.parse_args()

    if a.demo:
        df, a.fecha, a.valor, a.agg, a.grupo = datos_demo(), "fecha", "valor", "sum", None
    elif a.csv:
        df = pd.read_csv(a.csv)
    else:
        raise SystemExit("Indica --csv archivo.csv o usa --demo")

    if a.agg != "count" and not a.valor:
        raise SystemExit("Con --agg sum o mean debes indicar la columna con --valor (con count no hace falta).")
    for col in (a.fecha, a.valor, a.grupo):
        if col and col not in df.columns:
            raise SystemExit(f"No existe la columna '{col}'. Columnas del archivo: {list(df.columns)}")

    valor = None if a.agg == "count" else a.valor
    if a.sin_limpieza:
        df[a.fecha] = pd.to_datetime(df[a.fecha], errors="coerce")
        df = df.dropna(subset=[a.fecha])
        if a.grupo:
            df[a.grupo] = df[a.grupo].fillna("Sin dato")
    else:
        df, informe = limpiar(df, a.fecha, valor, a.grupo)
        print("Informe de limpieza")
        for nombre, cantidad in informe.items():
            print(f"  {nombre:<48}{cantidad:>9,}")

    meses = pd.date_range(df[a.fecha].min().to_period("M").to_timestamp(),
                          df[a.fecha].max().to_period("M").to_timestamp(), freq="MS")
    serie = serie_mensual(df.set_index(a.fecha), a.agg, valor, meses)
    if len(serie) < 12:
        raise SystemExit(f"Se necesitan al menos 12 meses de datos (hay {len(serie)}).")

    particion = {"test_size": a.test_meses} if a.test_meses else {"train_size": a.train_size}
    r = ajustar(serie, particion)
    num = r["num"]

    os.makedirs(a.salida, exist_ok=True)
    joblib.dump(r["final"], os.path.join(a.salida, "modelo.joblib"))
    serie.rename("valor").to_csv(os.path.join(a.salida, "serie_mensual.csv"))
    resultado = {
        "mejor_modelo": r["mejor"],
        "modelo_usado": r["usado"],
        "metricas_prueba": {n: {k: num_o_none(num.loc[n, k]) for k in ["MAE", "RMSE", "MAPE", "R2"]} for n in num.index},
        "mes_predicho": r["proximo"].strftime("%Y-%m"),
        "prediccion": round(r["prediccion"], 4),
    }
    with open(os.path.join(a.salida, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"\nMeses en la serie: {len(serie)} ({serie.index[0]:%Y-%m} a {serie.index[-1]:%Y-%m})")
    print(f"Entrenamiento: {r['n_train']} meses   Prueba: {r['n_test']} meses ({r['inicio_test']:%Y-%m} a {r['fin_test']:%Y-%m})")
    print(f"\n{'Modelo':<34}{'MAE':>14}{'RMSE':>14}{'MAPE':>9}")
    for n in num.index:
        print(f"{n:<34}{f_(num.loc[n, 'MAE']):>14}{f_(num.loc[n, 'RMSE']):>14}{pct(num.loc[n, 'MAPE']):>9}")
    print(f"\nMejor modelo (menor MAE): {r['mejor']}   |   se usa: {r['usado']}")
    print(f"Predicción del total para {resultado['mes_predicho']}: {f_(r['prediccion'])}")

    if a.grupo:
        modelos_g, detalle, omitidos, series_g = {}, {}, [], {}
        for grupo, sub in df.groupby(a.grupo):
            s = serie_mensual(sub.set_index(a.fecha), a.agg, valor, meses)
            if s.std() == 0 or (s == 0).mean() > 0.25:        # serie sin variación o casi vacía
                omitidos.append(grupo)
                continue
            rg = ajustar(s, particion)
            ng = rg["num"]
            modelos_g[grupo] = rg["final"]
            series_g[grupo] = s
            detalle[grupo] = {"mejor_modelo": rg["mejor"], "modelo_usado": rg["usado"],
                              "MAPE_modelo": num_o_none(ng.loc[rg["mejor"], "MAPE"]),
                              "MAPE_ingenuo": num_o_none(ng.loc[ING, "MAPE"]),
                              "prediccion": round(rg["prediccion"], 4)}
        joblib.dump(modelos_g, os.path.join(a.salida, "modelos_por_grupo.joblib"))
        pd.DataFrame(series_g).rename_axis("fecha").to_csv(os.path.join(a.salida, "series_por_grupo.csv"))
        with open(os.path.join(a.salida, "resultado_por_grupo.json"), "w", encoding="utf-8") as f:
            json.dump({"mes_predicho": resultado["mes_predicho"], "grupos": detalle}, f, ensure_ascii=False, indent=2)

        print(f"\nPor {a.grupo}: {len(detalle)} grupos entrenados")
        print(f"{'Grupo':<20}{'Modelo usado':<32}{'MAPE modelo':>12}{'MAPE ingenuo':>14}{'Predicción':>16}")
        for g, v in detalle.items():
            print(f"{str(g):<20}{v['modelo_usado']:<32}{pct(v['MAPE_modelo']):>12}{pct(v['MAPE_ingenuo']):>14}{f_(v['prediccion']):>16}")
        if omitidos:
            print("Grupos omitidos por tener más de 25% de meses sin datos:", omitidos)
    print(f"\nArchivos guardados en {a.salida}/")


if __name__ == "__main__":
    main()
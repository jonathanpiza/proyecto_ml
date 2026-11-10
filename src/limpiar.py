
import argparse
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="archivo CSV a limpiar")
    p.add_argument("--fecha", required=True, help="nombre de la columna de fecha")
    p.add_argument("--valor", help="nombre de la columna de monto (omitir en PQRS)")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    informe = {"Filas originales": len(df)}

    # 1. Duplicados: una fila repetida contaría dos veces en la suma del mes.
    informe["Filas duplicadas"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 2. Fechas: las que no se pueden leer quedan vacías (NaT) y se descartan,
    #    porque sin fecha no se sabe a qué mes pertenecen.
    #    Si tus fechas vienen como día/mes/año, agrega dayfirst=True.
    df[a.fecha] = pd.to_datetime(df[a.fecha], errors="coerce")
    informe["Fechas inválidas o vacías"] = int(df[a.fecha].isna().sum())
    df = df.dropna(subset=[a.fecha])

    # 3. Montos (solo ventas y gastos).
    if a.valor:
        df[a.valor] = pd.to_numeric(df[a.valor], errors="coerce")
        informe["Montos vacíos"] = int(df[a.valor].isna().sum())
        df = df.dropna(subset=[a.valor])

        informe["Montos negativos o en cero"] = int((df[a.valor] <= 0).sum())
        df = df[df[a.valor] > 0]

        # Un monto más de 20 veces mayor que la mediana casi siempre es un error de digitación.
        tope = 20 * df[a.valor].median()
        informe["Montos atípicos (más de 20 veces la mediana)"] = int((df[a.valor] > tope).sum())
        df = df[df[a.valor] <= tope]

    informe["Filas finales"] = len(df)

    print("\nInforme de limpieza")
    print("-" * 56)
    for nombre, cantidad in informe.items():
        print(f"{nombre:<48}{cantidad:>8,}")
    descartadas = informe["Filas originales"] - informe["Filas finales"]
    print("-" * 56)
    print(f"Se descartaron {descartadas:,} filas ({descartadas / informe['Filas originales']:.1%}).")

    base, _ = os.path.splitext(a.csv)
    salida = base.replace("_demo", "") + "_limpio.csv"
    df.to_csv(salida, index=False)
    print(f"Guardado en: {salida}")


if __name__ == "__main__":
    main()
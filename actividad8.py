
# Actividad 8 Regresión NO Lineal



import re
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


# CONFIGURACIÓN

try:
    CARPETA = Path(__file__).resolve().parent
except NameError:  # cuando se ejecuta como celdas / notebook
    CARPETA = Path.cwd()

SALIDA = CARPETA / "resultados"
GRAFICAS = SALIDA / "graficas"
SALIDA.mkdir(exist_ok=True)
GRAFICAS.mkdir(exist_ok=True)

PARES_POR_TABLA = 3       # cuántos pares de variables analizar por cada tabla
MIN_FILAS = 10            # tablas con menos filas se omiten
MAX_FILAS_AJUSTE = 200_000  # muestra máxima para ajustar los modelos
SEMILLA = 42

# (Opcional) Si quieres elegir tú los pares, llénalo así y se usarán en lugar
# de la selección automática:
#   PARES_MANUALES = [
#       ("Ventas_CLV_2022_2026", "Cantidad", "Importe"),
#       ("Existencias_04_09_2026", "Existencia", "Costo"),
#   ]
# El primer elemento es parte del nombre del archivo (o "archivo [hoja]").
PARES_MANUALES = []

# Columnas que parecen identificadores o fechas y no tienen sentido correlacionar
PATRON_EXCLUIR = re.compile(
    r"(?:^|[^a-záéíóúñ0-9])"
    r"(id|codigo|código|clave|folio|sku|cp|telefono|teléfono|rfc|"
    r"fecha|año|ano|anio|mes|dia|día|semana|periodo|year|month|day|date)"
    r"(?:$|[^a-záéíóúñ0-9])",
    re.IGNORECASE,
)


#  Carga de las bases de datos

def limpiar_nombre(texto):
    return re.sub(r"\s+", " ", str(texto)).strip()


def nombres_unicos(columnas):
    vistos, nuevas = {}, []
    for c in columnas:
        c = limpiar_nombre(c) or "columna"
        if c in vistos:
            vistos[c] += 1
            c = f"{c}_{vistos[c]}"
        else:
            vistos[c] = 0
        nuevas.append(c)
    return nuevas


def arreglar_encabezado(df):
    """Si el Excel/CSV trae títulos arriba de la tabla, busca la fila real de encabezados."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        return df
    sin_nombre = sum(str(c).startswith("Unnamed") for c in df.columns)
    if sin_nombre / len(df.columns) < 0.5:
        return df
    for i in range(min(20, len(df))):
        fila = df.iloc[i]
        if fila.notna().mean() >= 0.6:
            nuevo = df.iloc[i + 1:].copy()
            nuevo.columns = [v if pd.notna(v) else f"col_{j}" for j, v in enumerate(fila)]
            return nuevo.reset_index(drop=True)
    return df


def leer_csv(ruta):
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(ruta, sep=None, engine="python", encoding=codificacion)
            if df.shape[1] == 1:  # separador no detectado
                df = pd.read_csv(ruta, sep=";", encoding=codificacion)
            return df
        except UnicodeDecodeError:
            continue
    return pd.read_csv(ruta, sep=None, engine="python", encoding="latin-1", encoding_errors="replace")


def cargar_tablas(carpeta):
    tablas = {}
    for ruta in sorted(carpeta.iterdir()):
        if ruta.name.startswith(("~$", ".")) or not ruta.is_file():
            continue
        ext = ruta.suffix.lower()
        base = limpiar_nombre(ruta.stem)
        try:
            if ext == ".csv":
                tablas[base] = leer_csv(ruta)
            elif ext in (".xlsx", ".xls", ".xlsm"):
                hojas = pd.read_excel(ruta, sheet_name=None)
                for hoja, df in hojas.items():
                    nombre = base if len(hojas) == 1 else f"{base} [{hoja}]"
                    tablas[nombre] = df
        except Exception as e:
            print(f"  ⚠ No se pudo leer {ruta.name}: {e}")
    for nombre in list(tablas):
        df = arreglar_encabezado(tablas[nombre])
        df.columns = nombres_unicos(df.columns)
        tablas[nombre] = df
    return tablas


tablas_originales = cargar_tablas(CARPETA)
print(f"Tablas cargadas: {len(tablas_originales)}")
for nombre, df in tablas_originales.items():
    print(f"  - {nombre}: {df.shape[0]} filas x {df.shape[1]} columnas")


# Preprocesamiento de nulos, outliers y transformaciones


def a_numerico(serie):
    """Convierte textos como '$1,250.50' o '15%' a número cuando la columna es numérica."""
    if pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_datetime64_any_dtype(serie):
        return serie
    texto = serie.dropna().astype(str).str.strip()
    if texto.empty:
        return serie
    s = serie.astype(str).str.strip().str.replace(r"[\$\s%]", "", regex=True)
    coma_decimal = texto.str.fullmatch(r"-?\d+,\d{1,2}").mean() > 0.5
    s = s.str.replace(",", "." if coma_decimal else "", regex=False)
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan, "-": np.nan, "NaT": np.nan})
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= 0.9 * serie.notna().sum():
        return num
    return serie


def variables_analizables(df):
    """Columnas numéricas útiles: no IDs, no fechas, con variación suficiente."""
    columnas = []
    for c in df.select_dtypes(include="number").columns:
        if c.endswith("_log") or PATRON_EXCLUIR.search(c):
            continue
        serie = df[c].dropna()
        if serie.nunique() < 5:
            continue
        es_entero = np.allclose(serie, serie.round())
        if es_entero and serie.nunique() / max(len(serie), 1) > 0.95:
            continue  # parece un identificador
        columnas.append(c)
    return columnas


def preprocesar(df):
    info = {"filas_originales": len(df), "columnas_originales": df.shape[1]}
    df = df.copy()

    # Duplicados
    info["duplicados"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Tipos numéricos
    for c in df.columns:
        df[c] = a_numerico(df[c])

    # Nulos
    info["nulos"] = int(df.isna().sum().sum())
    muchas_nulas = [c for c in df.columns if df[c].isna().mean() > 0.6]
    df = df.drop(columns=muchas_nulas)
    info["columnas_eliminadas"] = len(muchas_nulas)
    for c in df.columns:
        if not df[c].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())       # numéricas -> mediana
        else:
            moda = df[c].mode(dropna=True)            # categóricas -> moda
            df[c] = df[c].fillna(moda.iloc[0] if len(moda) else "Sin dato")

    # Outliers: método IQR, se acotan (winsorización) en lugar de borrar filas.
    # En variables positivas con sesgo fuerte (montos, unidades) el IQR se calcula
    # en escala logarítmica; si no, marcaría como atípica una gran parte de los datos.
    variables = variables_analizables(df)
    total_outliers = 0
    for c in variables:
        sesgada = df[c].min() >= 0 and abs(df[c].skew()) > 1
        valores = np.log1p(df[c]) if sesgada else df[c]
        q1, q3 = valores.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        total_outliers += int(((valores < bajo) | (valores > alto)).sum())
        if sesgada:
            bajo, alto = np.expm1(bajo), np.expm1(alto)
        df[c] = df[c].clip(bajo, alto)
    info["outliers_acotados"] = total_outliers

    # Transformaciones de log(1+x) para variables con sesgo fuerte
    transformadas = []
    for c in variables:
        if df[c].min() >= 0 and abs(df[c].skew()) > 1:
            df[f"{c}_log"] = np.log1p(df[c])
            transformadas.append(c)
    info["transformadas"] = transformadas
    info["variables"] = variables
    info["filas_finales"] = len(df)
    return df, info


tablas, info_prep = {}, {}
for nombre, df in tablas_originales.items():
    limpio, info = preprocesar(df)
    tablas[nombre], info_prep[nombre] = limpio, info
    print(f"{nombre}: {info['duplicados']} duplicados, {info['nulos']} nulos tratados, "
          f"{info['outliers_acotados']} outliers acotados, "
          f"{len(info['transformadas'])} variables con log, "
          f"{len(info['variables'])} variables numéricas útiles")


#Selección de variables y modelos de Regresión No Lineal
#
# Modelo 1 – Polinomial de 2° grado:**  y = a·x² + b·x + c

# Modelo 2 – Exponencial:**  y = A·e^(B·x) + C


def elegir_pares(df, variables, n):
    if len(variables) < 2:
        return []
    corr = df[variables].corr(method="spearman").abs()
    candidatos = []
    for i, a in enumerate(variables):
        for b in variables[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r < 0.999:  # descarta variables casi idénticas
                candidatos.append((r, a, b))
    candidatos.sort(reverse=True)
    return [(a, b) for _, a, b in candidatos[:n]]


def metricas(y, y_pred, k):
    n = len(y)
    sse = np.sum((y - y_pred) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    r2_aj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
    r_mult = np.corrcoef(y, y_pred)[0, 1] if np.std(y_pred) > 0 else np.nan
    rmse = np.sqrt(sse / n)
    return r2, r2_aj, r_mult, rmse


def modelo_polinomial(x, y):
    p = np.polynomial.Polynomial.fit(x, y, 2).convert()
    c, b, a = (list(p.coef) + [0, 0, 0])[:3]
    pred = lambda xx: a * xx ** 2 + b * xx + c
    ecuacion = f"y = {a:.4g}·x² + {b:.4g}·x + {c:.4g}"
    return pred, ecuacion


def modelo_exponencial(x, y):
    # Se ajusta sobre x escalada a [0, 1] para estabilidad numérica
    xmin, rango = x.min(), x.max() - x.min()
    u = (x - xmin) / rango
    f = lambda uu, a, b, c: a * np.exp(b * uu) + c
    mejor = None
    for b0 in (-3, -1, -0.3, 0.3, 1, 3):  # valores iniciales por mínimos cuadrados
        e = np.exp(b0 * u)
        X = np.column_stack([e, np.ones_like(e)])
        (a0, c0), *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = np.sum((y - X @ [a0, c0]) ** 2)
        if mejor is None or sse < mejor[0]:
            mejor = (sse, [a0, b0, c0])
    try:
        params, _ = curve_fit(f, u, y, p0=mejor[1], maxfev=20000)
    except Exception:
        params = mejor[1]
    a, b, c = params
    # Regresar a la escala original de x:  y = A·e^(B·x) + C
    B = b / rango
    A = a * np.exp(-b * xmin / rango)
    pred = lambda xx: a * np.exp(b * (xx - xmin) / rango) + c
    ecuacion = f"y = {A:.4g}·e^({B:.4g}·x) + {c:.4g}"
    return pred, ecuacion


MODELOS = {
    "Polinomial (2° grado)": (modelo_polinomial, 3),
    "Exponencial": (modelo_exponencial, 3),
}


def buscar_tabla(clave):
    for nombre in tablas:
        if clave.lower() in nombre.lower():
            return nombre
    return None


# Definir los pares a analizar
pares = []
if PARES_MANUALES:
    for clave, x, y in PARES_MANUALES:
        nombre = buscar_tabla(clave)
        if nombre and x in tablas[nombre] and y in tablas[nombre]:
            pares.append((nombre, x, y))
        else:
            print(f"  ⚠ Par manual no encontrado: {clave} / {x} / {y}")
else:
    for nombre, df in tablas.items():
        if len(df) < MIN_FILAS:
            continue
        for x, y in elegir_pares(df, info_prep[nombre]["variables"], PARES_POR_TABLA):
            pares.append((nombre, x, y))

print(f"Pares de variables a analizar: {len(pares)}")


# Ajuste de modelos y tabla de coeficientes


rng = np.random.default_rng(SEMILLA)
filas_resultado, ajustes = [], []

for nombre, xcol, ycol in pares:
    datos = tablas[nombre][[xcol, ycol]].dropna()
    if len(datos) > MAX_FILAS_AJUSTE:
        datos = datos.sample(MAX_FILAS_AJUSTE, random_state=SEMILLA)
    x = datos[xcol].to_numpy(float)
    y = datos[ycol].to_numpy(float)
    if len(x) < MIN_FILAS or np.ptp(x) == 0 or np.ptp(y) == 0:
        continue

    pearson = stats.pearsonr(x, y)[0]
    spearman = stats.spearmanr(x, y)[0]
    predicciones = {}
    for modelo, (funcion, k) in MODELOS.items():
        try:
            pred, ecuacion = funcion(x, y)
            r2, r2_aj, r_mult, rmse = metricas(y, pred(x), k)
        except Exception as e:
            pred, ecuacion, r2, r2_aj, r_mult, rmse = None, f"No convergió ({e})", np.nan, np.nan, np.nan, np.nan
        predicciones[modelo] = pred
        filas_resultado.append({
            "Tabla": nombre, "Variable X": xcol, "Variable Y": ycol, "n": len(x),
            "Pearson r": pearson, "Spearman ρ": spearman, "Modelo": modelo,
            "Ecuación": ecuacion, "R²": r2, "R² ajustado": r2_aj,
            "r (y vs ŷ)": r_mult, "RMSE": rmse,
        })
    ajustes.append((nombre, xcol, ycol, x, y, predicciones))

resultados = pd.DataFrame(filas_resultado)
if not resultados.empty:
    mejor = (resultados.dropna(subset=["R² ajustado"])
             .sort_values("R² ajustado", ascending=False)
             .drop_duplicates(["Tabla", "Variable X", "Variable Y"]))
    claves_mejor = set(zip(mejor["Tabla"], mejor["Variable X"], mejor["Variable Y"], mejor["Modelo"]))
    resultados["Mejor modelo"] = [
        "✔" if (t, x, y, m) in claves_mejor else ""
        for t, x, y, m in zip(resultados["Tabla"], resultados["Variable X"],
                              resultados["Variable Y"], resultados["Modelo"])
    ]

    resultados.to_csv(SALIDA / "tabla_coeficientes.csv", index=False, encoding="utf-8-sig")
    try:
        resultados.to_excel(SALIDA / "tabla_coeficientes.xlsx", index=False)
    except Exception:
        pass
    pd.set_option("display.width", 200)
    print(resultados[["Tabla", "Variable X", "Variable Y", "Modelo", "R²", "R² ajustado", "r (y vs ŷ)"]]
          .round(4).to_string(index=False))
else:
    print("No se encontraron pares de variables numéricas para analizar.")

# Gráficas


rutas_graficas = []
for i, (nombre, xcol, ycol, x, y, predicciones) in enumerate(ajustes, start=1):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = rng.choice(len(x), size=min(3000, len(x)), replace=False)
    ax.scatter(x[idx], y[idx], s=8, alpha=0.35, color="gray", label="Datos")
    xs = np.linspace(x.min(), x.max(), 300)
    for (modelo, pred), color in zip(predicciones.items(), ("tab:blue", "tab:red")):
        if pred is None:
            continue
        r2 = resultados.query("Tabla == @nombre and `Variable X` == @xcol and "
                              "`Variable Y` == @ycol and Modelo == @modelo")["R²"].iloc[0]
        ax.plot(xs, pred(xs), color=color, lw=2, label=f"{modelo} (R²={r2:.3f})")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(textwrap.shorten(f"{nombre}: {ycol} vs {xcol}", 80), fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = GRAFICAS / f"grafica_{i:02d}.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    rutas_graficas.append(ruta)
print(f"Gráficas guardadas: {len(rutas_graficas)} en {GRAFICAS}")

# Reporte comparativo en PDF


def corto(texto, n=28):
    return textwrap.shorten(str(texto), n, placeholder="…")


def nivel(r2):
    if pd.isna(r2):
        return "sin ajuste"
    return "fuerte" if r2 >= 0.7 else "moderada" if r2 >= 0.3 else "débil"


def pagina_texto(pdf, titulo, parrafos):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.94, titulo, fontsize=16, weight="bold")
    yy = 0.89
    for p in parrafos:
        for linea in textwrap.wrap(p, 95) or [""]:
            fig.text(0.08, yy, linea, fontsize=10)
            yy -= 0.022
        yy -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def pagina_tabla(pdf, titulo, df, anchos=None, fuente=7):
    for inicio in range(0, max(len(df), 1), 22):
        bloque = df.iloc[inicio:inicio + 22]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(titulo + (" (cont.)" if inicio else ""), fontsize=13, weight="bold", loc="left")
        if not bloque.empty:
            t = ax.table(cellText=bloque.values, colLabels=bloque.columns,
                         loc="upper center", cellLoc="center", colWidths=anchos)
            t.auto_set_font_size(False)
            t.set_fontsize(fuente)
            t.scale(1, 1.5)
            for (fila, _), celda in t.get_celld().items():
                if fila == 0:
                    celda.set_facecolor("#dde6f2")
                    celda.set_text_props(weight="bold")
        pdf.savefig(fig)
        plt.close(fig)


ruta_pdf = SALIDA / "Reporte_Actividad8_Regresion_No_Lineal.pdf"
with PdfPages(ruta_pdf) as pdf:
    # Portada y metodología
    pagina_texto(pdf, "Actividad 8 – Regresión NO Lineal", [
        "Reporte comparativo del análisis de correlación no lineal entre variables numéricas "
        "de las bases de datos de la empresa.",
        f"Bases de datos analizadas ({len(tablas)}): " + "; ".join(tablas.keys()) + ".",
        "Preprocesamiento: se eliminaron filas duplicadas; las columnas con más de 60% de valores "
        "nulos se descartaron; los nulos restantes se imputaron con la mediana (variables numéricas) "
        "o la moda (variables categóricas). Los valores atípicos se detectaron con el método del "
        "rango intercuartílico (IQR, 1.5·IQR) y se acotaron a los límites en lugar de eliminar "
        "registros; en variables positivas con sesgo fuerte el IQR se calculó en escala "
        "logarítmica para no marcar como atípica una parte excesiva de los datos. A las variables con sesgo fuerte (|asimetría| > 1) se les generó una versión "
        "transformada con log(1+x).",
        "Selección de variables: se descartaron identificadores, claves y fechas; para cada tabla "
        f"se eligieron los {PARES_POR_TABLA} pares de variables numéricas con mayor correlación "
        "de Spearman (que captura relaciones monótonas, incluso no lineales).",
        "Modelos de Regresión No Lineal: (1) Polinomial de 2° grado, y = a·x² + b·x + c; "
        "(2) Exponencial, y = A·e^(B·x) + C, ajustado por mínimos cuadrados no lineales "
        "(scipy.optimize.curve_fit).",
        "Métricas: coeficiente de determinación R², R² ajustado, coeficiente de correlación del "
        "modelo r (correlación entre y observada y y estimada), RMSE, y los coeficientes de "
        "correlación de Pearson (lineal) y Spearman (por rangos) de los datos.",
        "Interpretación de R²: ≥ 0.70 relación fuerte; 0.30 a 0.69 moderada; < 0.30 débil.",
    ])

    # Preprocesamiento
    prep = pd.DataFrame([{
        "Tabla": corto(n, 40),
        "Filas orig.": i["filas_originales"], "Filas finales": i["filas_finales"],
        "Duplicados": i["duplicados"], "Nulos tratados": i["nulos"],
        "Cols. eliminadas": i["columnas_eliminadas"], "Outliers acotados": i["outliers_acotados"],
        "Vars. con log": len(i["transformadas"]), "Vars. numéricas": len(i["variables"]),
    } for n, i in info_prep.items()])
    pagina_tabla(pdf, "1. Resumen del preprocesamiento", prep, fuente=8)

    if not resultados.empty:
        # Tabla de coeficientes
        tabla = resultados.assign(**{
            "Tabla": resultados["Tabla"].map(lambda s: corto(s, 26)),
            "Variable X": resultados["Variable X"].map(corto),
            "Variable Y": resultados["Variable Y"].map(corto),
            "Ecuación": resultados["Ecuación"].map(lambda s: corto(s, 42)),
        })[["Tabla", "Variable X", "Variable Y", "Modelo", "Ecuación", "Pearson r",
            "Spearman ρ", "R²", "R² ajustado", "r (y vs ŷ)", "Mejor modelo"]]
        for c in ["Pearson r", "Spearman ρ", "R²", "R² ajustado", "r (y vs ŷ)"]:
            tabla[c] = tabla[c].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}")
        tabla["Mejor modelo"] = tabla["Mejor modelo"].replace("✔", "Sí")
        pagina_tabla(pdf, "2. Coeficientes de determinación y correlación", tabla,
                     anchos=[.12, .1, .1, .1, .2, .06, .06, .06, .06, .06, .06], fuente=6)

        # Comparativo
        resumen = resultados.groupby("Modelo").agg(
            R2_promedio=("R²", "mean"), R2_max=("R²", "max"),
            Veces_mejor=("Mejor modelo", lambda s: (s == "✔").sum()))
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
        resumen["R2_promedio"].plot.bar(ax=axs[0], color=["tab:blue", "tab:red"])
        axs[0].set_title("R² promedio por modelo")
        axs[0].set_ylim(0, 1)
        resumen["Veces_mejor"].plot.bar(ax=axs[1], color=["tab:blue", "tab:red"])
        axs[1].set_title("Número de pares donde fue el mejor modelo")
        for a in axs:
            a.tick_params(axis="x", rotation=0)
            a.set_xlabel("")
        fig.suptitle("3. Comparación de modelos", weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Conclusiones automáticas
        ganador = resumen["Veces_mejor"].idxmax()
        top = resultados.loc[resultados["R²"].idxmax()]
        parrafos = [
            f"Se analizaron {len(ajustes)} pares de variables con 2 modelos no lineales "
            f"({len(resultados)} ajustes en total).",
        ]
        for modelo, fila in resumen.iterrows():
            parrafos.append(
                f"{modelo}: R² promedio = {fila['R2_promedio']:.4f}, R² máximo = {fila['R2_max']:.4f}; "
                f"fue el mejor modelo en {int(fila['Veces_mejor'])} de {len(ajustes)} pares.")
        parrafos.append(
            f"En general, el modelo {ganador} describió mejor las relaciones analizadas, al obtener "
            f"el R² ajustado más alto en la mayoría de los casos.")
        parrafos.append(
            f"La relación más fuerte fue {top['Variable Y']} vs {top['Variable X']} "
            f"(tabla {top['Tabla']}) con el modelo {top['Modelo']}: R² = {top['R²']:.4f}, "
            f"r = {top['r (y vs ŷ)']:.4f} (relación {nivel(top['R²'])}).")
        for (t, xx, yy), g in resultados.groupby(["Tabla", "Variable X", "Variable Y"], sort=False):
            m = g.loc[g["R² ajustado"].idxmax()] if g["R² ajustado"].notna().any() else g.iloc[0]
            parrafos.append(
                f"• {corto(yy, 40)} vs {corto(xx, 40)} ({corto(t, 40)}): mejor ajuste {m['Modelo']} "
                f"con R² = {m['R²']:.4f} (relación {nivel(m['R²'])}); Pearson = {m['Pearson r']:.3f}, "
                f"Spearman = {m['Spearman ρ']:.3f}.")
        parrafos.append(
            "Cuando Spearman es claramente mayor que Pearson, la relación es monótona pero no lineal, "
            "lo que justifica el uso de modelos no lineales. Un R² bajo en ambos modelos indica que "
            "la variable X explica poca variación de Y y que intervienen otros factores.")
        pagina_texto(pdf, "4. Análisis comparativo y conclusiones", parrafos)

        # Gráficas (2 por página)
        for inicio in range(0, len(rutas_graficas), 2):
            fig, axs = plt.subplots(2, 1, figsize=(8.5, 11))
            for ax, ruta in zip(axs, rutas_graficas[inicio:inicio + 2] + [None]):
                ax.axis("off")
                if ruta is not None:
                    ax.imshow(plt.imread(ruta))
            fig.suptitle("5. Gráficas de los modelos ajustados", weight="bold")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

print(f"\n Reporte generado: {ruta_pdf}")
print(f" Tabla de coeficientes: {SALIDA / 'tabla_coeficientes.csv'}")

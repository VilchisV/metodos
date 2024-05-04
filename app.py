# libraries
import streamlit as st
import numpy as np
import pandas as pd
import math
import sympy
from sympy import simplify, symbols, diff, solve, symbols, simplify, Poly, Rational
from fractions import Fraction
import re
import time
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
x = symbols("x")












def ddn(m, z):
    # Crear matriz para diferencias divididas
    a = []
    for g in range(len(m) + 1):
        aux = [0] * len(m)
        a.append(aux)

    for s in range(len(m)):
        a[0][s] = m[s]
        a[1][s] = z[s]

    b = 1
    c = 1
    d = 1
    w = 0
    for i in range(len(a[0])):
        for j in range(len(a[0]) - b):
            a[c + 1][j] = (a[c][j + 1] - a[c][j]) / (a[0][j + d] - a[0][j])
        b += 1
        c += 1
        d += 1

    # Transponer y redondear la matriz para visualización
    matrix = np.transpose(a)
    matrix_r = np.round(matrix, decimals=4)
    return matrix_r



# Configuración básica
st.set_page_config(
    page_title="Mi Aplicación",
    page_icon="📊",
    layout="wide"
)

# Estilo para cambiar el color de la barra lateral
sidebar_style = """
<style>
.css-1v3fvcr {
    background-color: #8B4513; /* Marrón oscuro */
    color: #F5DEB3; /* Trigo */
}
</style>
"""

# Aplicar el estilo a la barra lateral
st.markdown(sidebar_style, unsafe_allow_html=True)

# Placeholder para la imagen
image_placeholder = st.empty()

# Cargar y mostrar la imagen
image_path = "/home/vilchis-vic/Descargas/XD.jpeg"
image = Image.open(image_path)

with image_placeholder.container():
    st.image(image, width=300, caption="Why, God, Why??")

# Simular espera
time.sleep(2)

# Limpiar el placeholder
image_placeholder.empty()

# Barra lateral con opciones
with st.sidebar:
    st.header("Métodos Numéricos II")

    num_method = st.radio(
        "Elija uno de los siguientes métodos numéricos",
        [
            "False Position",
            "Newton-Raphson",
            "Secant",
            "Punto Fijo",
            "Lagrange",
            "Diferencias Divididas",
            "Mínimos Cuadrados",
            "Método del Trapecio",
            "Método de Simpson 1/3",
            "Método de Simpson 3/3"
        ]
    )
    


# False Position
if num_method == "False Position":
    st.markdown("<h1 style='text-align: center;'>False Position</h1>", unsafe_allow_html=True)

    st.info("A root is considered found when the absolute value of $f(c)$, where $c$ is the midpoint of the interval, is smaller than a predefined tolerance value and this occurs before the maximum number of iterations is reached", icon="ℹ️")

    equation = st.text_input("Input the equation")
    if equation:
        function = simplify(equation)
        st.write(function)

        character = [char for char in equation if char.isalpha()]
        if len(character) > 0:
            symbol = symbols(character[0])

    with st.form(key="my_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            a = st.number_input("Input the first interval")
        with col2:
            b = st.number_input("Input the second interval")
        with col3:
            e = st.number_input("Input the tolerance", format="%.6f", value=0.000001)
        with col4:
            N = st.number_input("Input the maximum number of iterations", value=100)

        submit_button = st.form_submit_button(label="Calculate")

    if submit_button:
        fa = function.subs(symbol, a)
        fb = function.subs(symbol, b)

        if fa * fb < 0:
            n = 1
            list_a, list_b, list_c, list_fa, list_fb, list_fc, list_abs_fc = ([] for _ in range(7))
            while True:
                c = b - ((fb * (b - a)) / (fb - fa))
                fc = function.subs(symbol, c)

                lists = [list_a, list_b, list_c, list_fa, list_fb, list_fc, list_abs_fc]
                variables = [a, b, c, fa, fb, fc, math.fabs(fc)]
                variables = [float(i) for i in variables]
                
                for i, j in zip(lists, variables):
                    i.append(j)

                if math.fabs(fc) <= e:
                    result = "success"
                    break

                if n > N:
                    result = "failed"
                    break

                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
                n += 1

            df = pd.DataFrame({
                "a": list_a,
                "b": list_b,
                "c": list_c,
                "f(a)": list_fa,
                "f(b)": list_fb,
                "f(c)": list_fc,
                "| f(c) |": list_abs_fc
            })
            df.index += 1
            st.dataframe(df.head(N).style.format({"E": "{:.6f}"}), use_container_width=True)

            if result == "success":
                st.success(f"This equation has an approximate root of {np.round(float(c), 6)}")
            else:
                st.error("The maximum number of iterations has been exceeded")
        else:
            st.error("This equation doesn't have any solutions")

# Newton-Raphson
if num_method == "Newton-Raphson":
    st.markdown("<h1 style='text-align: center;'>Newton-Raphson</h1>", unsafe_allow_html=True)

    st.info("A root is considered found when the absolute value of the difference between $x_{n}$ and $x_{n-1}$ is smaller than a predefined tolerance value and this occurs before the maximum number of iterations is reached", icon="ℹ️")

    equation = st.text_input("Input the equation")
    if equation:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Initial Equation")
            function = simplify(equation)
            st.write(function)

        character = [char for char in equation if char.isalpha()]
        if len(character) > 0:
            symbol = symbols(character[0])

        with col2:
            st.write("Derived Equation")
            derived_function = diff(function, symbol)
            st.write(derived_function)

    with st.form(key="my_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            x0 = st.number_input("Input the initial guess")
        with col2:
            e = st.number_input("Input the tolerance", format="%.6f", value=0.000001)
        with col3:
            N = st.number_input("Input the maximum number of iterations", value=100)

        submit_button = st.form_submit_button(label="Calculate")

    if submit_button:
        n = 0
        list_xn, list_fxn, list_f_der_xn, list_diff = ([] for _ in range(4))
        while True:
            f = function.subs(symbol, x0)
            f_derivative = derived_function.subs(symbol, x0)
            if f_derivative == 0:
                result = "zero"
                break

            if n == 0:
                lists = [list_xn, list_fxn, list_f_der_xn, list_diff]
                variables = [x0, f, f_derivative, np.nan]
                variables = [float(i) for i in variables]

                for i, j in zip(lists, variables):
                    i.append(j)

            x1 = x0 - (f / f_derivative)
            fx1 = function.subs(symbol, x1)
            f_derivative_x1 = derived_function.subs(symbol, x1)
            diff = math.fabs(x1 - x0)

            lists = [list_xn, list_fxn, list_f_der_xn, list_diff]
            variables = [x1, fx1, f_derivative_x1, diff]
            variables = [float(i) for i in variables]
            
            for i, j in zip(lists, variables):
                i.append(j)
            
            x0 = x1
            n += 1

            if diff <= e:
                result = "success"
                break

            if n > N:
                result = "failed"
                break

        if result != "zero":
            df = pd.DataFrame({
                "xn": list_xn,
                "f(xn)": list_fxn,
                "f'(xn)": list_f_der_xn,
                "| xn - xn-1 |": list_diff
            })
            st.dataframe(df.head(N).style.format({"E": "{:.6f}"}), use_container_width=True)

        if result == "success":
            st.success(f"This equation has an approximate root of {np.round(float(x1), 6)}")
        elif result ==  "failed":
            st.error("The maximum number of iterations has been exceeded")
        else:
            st.error("Division by zero is not allowed")

# Secant
if num_method == "Secant":
    st.markdown("<h1 style='text-align: center;'>Secant</h1>", unsafe_allow_html=True)

    st.info("A root is considered found when the absolute value of the difference between $x_{n}$ and $x_{n-1}$ is smaller than a predefined tolerance value and this occurs before the maximum number of iterations is reached", icon="ℹ️")

    equation = st.text_input("Input the equation")
    if equation:
        function = simplify(equation)
        st.write(function)

        character = [char for char in equation if char.isalpha()]
        if len(character) > 0:
            symbol = symbols(character[0])

    with st.form(key="my_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x0 = st.number_input("Input the first guess")
        with col2:
            x1 = st.number_input("Input the second guess")
        with col3:
            e = st.number_input("Input the tolerance", format="%.6f", value=0.000001)
        with col4:
            N = st.number_input("Input the maximum number of iterations", value=100)

        submit_button = st.form_submit_button(label="Calculate")

    if submit_button:
        n = 0
        list_x, list_fx, list_diff = ([] for _ in range(3))
        while True:
            fx0 = function.subs(symbol, x0)
            fx1 = function.subs(symbol, x1)
            x = x1 - ((fx1 * (x1 - x0)) / (fx1 - fx0))
            fx = function.subs(symbol, x)

            if math.fabs(fx1 - fx0) == 0:
                result = "zero"
                break

            if n == 0:
                lists = [list_x, list_fx, list_diff]
                variables = [x0, fx0, np.nan]
                variables = [float(i) for i in variables]

                for i, j in zip(lists, variables):
                    i.append(j)

            diff = math.fabs(x - x0)

            lists = [list_x, list_fx, list_diff]
            variables = [x, fx, diff]
            variables = [float(i) for i in variables]
            
            for i, j in zip(lists, variables):
                i.append(j)
            
            x0 = x1
            x1 = x
            n += 1

            if diff <= e:
                result = "success"
                break

            if n > N:
                result = "failed"
                break

        if result != "zero":
            df = pd.DataFrame({
                "xn": list_x,
                "f(xn)": list_fx,
                "| xn - xn-1 |": list_diff
            })
            st.dataframe(df.head(N).style.format({"E": "{:.6f}"}), use_container_width=True)

        if result == "success":
            st.success(f"This equation has an approximate root of {np.round(float(x1), 6)}")
        elif result ==  "failed":
            st.error("The maximum number of iterations has been exceeded")
        else:
            st.error("Division by zero is not allowed")


if num_method == "Diferencias Divididas":
    st.markdown("<h1 style='text-align: center;'>Diferencias Divididas</h1>", unsafe_allow_html=True)
    # Texto informativo
    st.info("El método de Newton de las diferencias divididas nos permite calcular los coeficientes $c_j$ de la combinación lineal mediante la construcción de las llamadas diferencias divididas, que vienen definidas de forma recurrente.")
    
    # Presentar fórmulas utilizando st.latex
    st.latex("f[x_i] = f_i")
    
    st.latex("f[x_i, x_{i+1}, \ldots, x_{i+j}] = \\frac{f[x_{i+1}, \ldots, x_{i+j}] - f[x_i, x_{i+1}, \ldots, x_{i+j-1}]}{x_{i+j} - x_i}")
    
    st.info("Tenemos los siguientes casos particulares:")
    
    st.latex("f[x_0, x_1] = \\frac{f[x_1] - f[x_0]}{x_1 - x_0}")
    
    st.latex("f[x_0, x_1, x_2] = \\frac{f[x_1, x_2] - f[x_0, x_1]}{x_2 - x_0}")


    with st.form(key="divided_diff_form"):
        num_puntos = st.number_input("Ingrese el número de puntos", min_value=2, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            x_i = st.text_area("Ingrese los valores de $x_i$ separados por comas (,)", value="0, 1, 2")
        with col2:
            f_i = st.text_area("Ingrese los valores de $f_i$ separados por comas (,)", value="1, 2, 3")
        
        submit_button = st.form_submit_button(label="Calcular")

    if submit_button:
        try:
            # Obtener valores de los campos de texto
            m = [float(x) for x in x_i.split(",")]
            z = [float(x) for x in f_i.split(",")]

            if len(m) != num_puntos or len(z) != num_puntos:
                st.error(f"Error: Las listas deben tener exactamente {num_puntos} elementos.")
            else:
                st.info(f"Tabla de valores con {num_puntos} puntos:")
                tabla = pd.DataFrame({"xi": m, "fi": z})
                st.dataframe(tabla)
                
                # Calcular las diferencias divididas
                dd_matrix = ddn(m, z)
                st.write("Tabla de diferencias:")
                st.dataframe(dd_matrix)
                
                # Construcción del polinomio de diferencias divididas
                a = []
                for g in range(len(m) + 1):
                    aux = []
                    for e in range(len(m)):
                        aux.append(0)
                    a.append(aux)
            
                for s in range(len(m)):
                    a[0][s] = m[s]
                    a[1][s] = z[s]

                b = 1
                c = 1
                d = 1
                w = 0  # Inicializa la variable w
                for i in range(len(a[0])):
                    for j in range(len(a[0]) - b):
                        a[c + 1][j] = (a[c][j + 1] - a[c][j]) / (a[0][j + d] - a[0][j])
                    b += 1
                    c += 1
                    d += 1
                print("\n")
                matrix = np.array(a)
                matrix_t = np.transpose(matrix)
                matrix_r=np.round(matrix_t, decimals=4)
                matrix_df = pd.DataFrame(matrix_r)
                print("Tabla De Diferencias:")
                print(matrix_df)
                # Se obtiene todo el polinomio
                p = 0  # Define polinomio inicialmente
                for t in range(len(a[0])):
                   terminos = 1
                   for r in range(w):
                       terminos *= (x - a[0][r])
                   w += 1  # Actualiza w
                   p += a[t + 1][0] * terminos
                pol = simplify(p)                
                    # Obtener los coeficientes del polinomio
                coefficients = pol.as_poly().all_coeffs()
                   # Convertir los coeficientes a fracciones
                coefficients_as_fractions = [Rational(coef).limit_denominator() for coef in coefficients]
                   # Reconstruir el polinomio a partir de los coeficientes fraccionarios
                polynomial_terms = [f"{coef}*x^{i}" for i, coef in enumerate(coefficients_as_fractions[::-1])]
                   # Imprimir el polinomio con los términos separados
                polynomial_expression = " + ".join(polynomial_terms)   
                sexo = simplify(polynomial_expression)
                st.markdown(f"**Polinomio de Diferencias Divididas:** {pol}")
                st.write(f"{sexo}") 
                sexo = str(sexo)
                # Tu ecuación dinámica (como cadena de texto)
                # Convertimos "**" a "^" para potencias
                latex_expression = sexo.replace("**", "^")
                
                # Eliminamos el símbolo "*" ya que en LaTex, la multiplicación es implícita
                latex_expression = latex_expression.replace("*", "")
                
                # Usamos una expresión regular para transformar divisiones en formato LaTex
                # Esto convierte x/y a \frac{x}{y}
                latex_expression = re.sub(r"(\d+|\w)/(\d+)", r"\\frac{\1}{\2}", latex_expression)
                
                # Renderizamos en LaTex con Streamlit
                st.latex(latex_expression)

                
                                
                

        except Exception as e:
            st.error(f"Ocurrió un error al procesar los datos: {e}")
            
# Interpolación de Lagrange
if num_method == "Lagrange":
    st.markdown("<h1 style='text-align: center;'>Interpolación de Lagrange</h1>", unsafe_allow_html=True)
    st.info("Este método es el más explícito para probar existencia de solución ya que la construye.Sin embargo su utilidad se reduce a eso: a dar una respuesta formal y razonada, pues no es eficiente en términos de cálculo (requiere muchas operaciones y tiene limitaciones técnicas)", icon="ℹ️")
    st.info(" La fórmula de interpolación de Lagrange es:")
    
    # Fórmula de interpolación de Lagrange
    st.latex("P(x) = \\sum_{k=0}^{n} f_k \\cdot l_k(x)")
    
    # Definición de l_k(x)
    st.latex("l_k(x) = \\prod_{j=0, \\; j \\neq k}^{n} \\frac{x - x_j}{x_k - x_j}, \\; \\text{para } k = 0, \\ldots, n.")

    # Entrada de datos para los valores de 'x_i' y 'f_i'
    entrada_x = st.text_input("Ingrese los elementos de la lista 'x_i' separados por comas(,):")
    xi = [float(x) for x in entrada_x.split(",")] if entrada_x else []

    entrada_y = st.text_input("Ingrese los elementos de la lista 'f_i' separados por comas(,):")
    fi = [float(x) for x in entrada_y.split(",")] if entrada_y else []

    # Verificar que las listas tengan la misma longitud y al menos un elemento
    if len(xi) == len(fi) and len(xi) > 0:
        # Crear el polinomio de Lagrange
        n = len(xi)
        x = sympy.Symbol("x")
        polinomio = 0
        divisorL = np.zeros(n, dtype=float)

        # Calcular el polinomio de Lagrange
        for i in range(0, n, 1):
            numerador = 1
            denominador = 1
            for j in range(0, n, 1):
                if j != i:
                    numerador *= x - xi[j]
                    denominador *= xi[i] - xi[j]
            terminoLi = numerador / denominador
            polinomio += terminoLi * fi[i]
            divisorL[i] = denominador

        # Simplificar el polinomio
        polisimple = polinomio.expand()

        # Para evaluación numérica
        px = sympy.lambdify(x, polisimple)

        # Crear datos para el gráfico
        muestras = 101
        a = np.min(xi)
        b = np.max(xi)
        pxi = np.linspace(a, b, muestras)
        pfi = px(pxi)

        # Mostrar resultados
        st.write("Divisores en L(i):", divisorL)
        st.write("Polinomio de Lagrange (expresión):", polinomio)
        st.write("Polinomio de Lagrange (simplificado):", polisimple)

        # Crear y mostrar el gráfico
        fig, ax = plt.subplots()
        ax.plot(xi, fi, 'o', label='Puntos')
        ax.plot(pxi, pfi, label='Polinomio')
        ax.legend()
        ax.set_xlabel("xi")
        ax.set_ylabel("fi")
        ax.set_title("Interpolación de Lagrange")

        buf = BytesIO()  # Crear un buffer para el gráfico
        plt.savefig(buf, format='png')  # Guardar el gráfico en memoria
        buf.seek(0)  # Ir al inicio del buffer
        st.image(buf, use_column_width=True)  # Mostrar la imagen en Streamlit

    else:
        st.warning("Por favor, ingrese listas válidas para 'x_i' y 'f_i', asegurándose de que tengan la misma longitud.")

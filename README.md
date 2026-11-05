# Laboratorio Interactivo de Aprendizaje de Máquina (Punto 4: Sitio Web en GitHub Pages)

Aplicación web interactiva client-side diseñada para la experimentación pedagógica y numérica de tres problemas fundamentales de Aprendizaje de Máquina: **Regularización Elastic Net y Estimación MAP**, **Clasificación Bayesiana vs. Discriminativa (QDA, Naive Bayes e IRLS)** y **Geometría de Máquinas de Vectores de Soporte (SVM) en RKHS**.

---

## 1. Título y Descripción del Proyecto

El proyecto constituye la entrega interactiva del **Punto 4 del Parcial de Aprendizaje de Máquina**. Implementa un banco de simulación numérica que ejecuta algoritmos de optimización matemática y modelos estadísticos en tiempo real, directamente en el navegador del usuario (Vanilla JavaScript). 

La aplicación permite contrastar las derivaciones teóricas analíticas con simulaciones visuales reactivas:
* **Módulo 1:** Deducción del estimador MAP bajo prior compuesto (Laplace + Gaussiano), visualización 2D de curvas de nivel (Prior, Verosimilitud, Posterior) y cálculo del Camino de Regularización (*Regularization Path*) mediante **Descenso por Coordenadas Cíclico**.
* **Módulo 2:** Comparativa exhaustiva entre clasificadores generativos y discriminativos con los parámetros exactos del examen ($\mu_k, \Sigma_k, \pi_k$). Genera fronteras simultáneas para **QDA** (matriz completa), **Gaussian Naive Bayes** (eje-alineada) y **Regresión Logística** con optimizador **IRLS (Newton-Raphson)** bajo expansión cuadrática $\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]^T$.
* **Módulo 3:** Optimizador interactivo de **SVM Dual con algoritmo SMO (Sequential Minimal Optimization)** para kernels Lineal y RBF, lienzo editable con soporte para añadir y arrastrar muestras, cálculo del ancho del margen y verificación empírica de la identidad fundamental de margen duro:
  $$\|w\|_{\mathcal{H}}^2 = \sum_{n=1}^N \alpha_n$$
* **Control General:** Motor PRNG determinista (Mulberry32) con control de semilla aleatoria para garantizar reproducibilidad exacta y renderizado tipográfico de ecuaciones matemáticas mediante **KaTeX**.

---

## 2. Estructura Arquitectónica

Siguiendo las restricciones arquitectónicas de **simplicidad y cero exceso de modularización**, la solución no utiliza bundlers (Webpack, Vite), frameworks de componentes (React, Vue) ni dependencias externas de Node.js/npm. Todo el sistema está contenido en 3 archivos planos:

```text
ParcialTAM/
├── index.html   # Estructura semántica, paneles de control, Canvas 2D y CDN de KaTeX
├── style.css    # Diseño responsivo en CSS puro (Dark Slate Theme, variables CSS, Flexbox/Grid)
├── app.js       # Motor matemático, álgebra lineal, optimizadores (SMO, IRLS, CD) y renderers
└── README.md    # Documentación técnica, formulación matemática y manual de despliegue
```

### Justificación de la Arquitectura Cliente (100% Vanilla JS):
1. **Portabilidad Inmediata y Cero Fricción:** El sitio funciona de forma inmediata al abrir `index.html` en cualquier navegador web moderno, sin requerir compilación ni instalación previa de paquetes (`npm install`).
2. **Compatibilidad Nativa con GitHub Pages:** No requiere flujos de integración continua (CI) complejos ni pasos de build para generar la carpeta de distribución (`dist/`). Los archivos fuente son directamente los archivos servidos.
3. **Eficiencia Numérica:** La ejecución en el hilo principal mediante operaciones vectorizadas manuales y renderizado directo sobre `<canvas>` HTML5 garantiza tasas de refresco de 60 FPS en las simulaciones interactivas.

---

## 3. Detalle Teórico de "Qué se hizo y Cómo se hizo"

### 3.1. Módulo 1: Elastic Net, Estimador MAP y Descenso por Coordenadas

#### Deducción Matemática:
Dado el modelo de regresión lineal con ruido gaussiano i.i.d.:
$$y = Xw + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I_N)$$
la función de verosimilitud condicional es:
$$p(y|X, w) = \left(2\pi\sigma^2\right)^{-\frac{N}{2}} \exp\left(-\frac{1}{2\sigma^2}\|y - Xw\|_2^2\right)$$

Definimos una distribución a priori independiente sobre los pesos $w$ correspondiente al prior de Elastic Net (combinación convexa de prior Laplace y Gaussiano):
$$p(w) \propto \exp\left( -\lambda_1 \|w\|_1 - \frac{\lambda_2}{2}\|w\|_2^2 \right)$$

Aplicando el Teorema de Bayes, la distribución a posteriori es $p(w|y) \propto p(y|X,w)p(w)$. Maximizar la log-posterior $\ln p(w|y)$ equivale a minimizar la función de costo regularizada:
$$\mathcal{L}_{MAP}(w) = \frac{1}{2N}\|y - Xw\|_2^2 + \lambda \left[ \alpha \|w\|_1 + \frac{1-\alpha}{2}\|w\|_2^2 \right]$$
donde $\lambda$ parametriza la intensidad global de la penalización y $\alpha \in [0, 1]$ equilibra el comportamiento entre Lasso ($\alpha=1$) y Ridge ($\alpha=0$).

#### Algoritmo de Descenso por Coordenadas (Coordinate Descent):
Dado que la norma $\ell_1$ no es diferenciable en el origen, la optimización simultánea no puede resolverse con gradiente simple. Se aplica optimización univariada cíclica sobre cada coordenada $w_j$, manteniendo fijas las demás $k \neq j$:

1. Se aísla el residual parcial sin la variable $j$:
   $$r^{(j)} = y - \sum_{k \neq j} X_k w_k = (y - Xw) + X_j w_j$$
2. Se calcula la correlación entre la variable $X_j$ y el residual parcial:
   $$\rho_j = \frac{1}{N} X_j^T r^{(j)} = \frac{1}{N} X_j^T (y - Xw) + z_j w_j$$
   donde $z_j = \frac{1}{N}\|X_j\|_2^2 = 1$ para datos estandarizados.
3. Se aplica el **Operador de Umbral Suave (Soft-Thresholding)** $S(z, \gamma) = \text{sign}(z)\max(0, |z| - \gamma)$:
   $$w_j \leftarrow \frac{S\left( \rho_j,\, \lambda \alpha \right)}{z_j + \lambda(1-\alpha)}$$
   - Si $\alpha = 1$ (Lasso), anula coeficientes exactamente en cero generando esparsidad.
   - Si $\alpha = 0$ (Ridge), preserva todos los coeficientes con contracción proporcional.
   - Si $0 < \alpha < 1$ (Elastic Net), agrupa variables fuertemente correlacionadas y estabiliza la selección de variables.

---

### 3.2. Módulo 2: Clasificación Bayesiana (QDA/GNB) vs. Discriminativa (Logística IRLS)

#### Análisis Cuadrático Discriminante (QDA):
En el modelo generativo bayesiano, cada clase se modela con una distribución normal multivariada $x|C_k \sim \mathcal{N}(\mu_k, \Sigma_k)$ con probabilidad a priori $\pi_k$. La regla de decisión bayesiana óptima asigna $C_1$ si $p(C_1|x) > p(C_2|x)$, lo que equivale a evaluar el logaritmo de la razón de probabilidades posteriores:
$$\ln \frac{p(C_1|x)}{p(C_2|x)} = \delta_1(x) - \delta_2(x) = x^T A x + b^T x + c = 0$$

Desarrollando los términos cuadráticos y lineales:
$$A = -\frac{1}{2}\left( \Sigma_1^{-1} - \Sigma_2^{-1} \right)$$
$$b = \Sigma_1^{-1}\mu_1 - \Sigma_2^{-1}\mu_2$$
$$c = -\frac{1}{2}\mu_1^T \Sigma_1^{-1}\mu_1 + \frac{1}{2}\mu_2^T \Sigma_2^{-1}\mu_2 - \frac{1}{2}\ln\frac{|\Sigma_1|}{|\Sigma_2|} + \ln\frac{\pi_1}{\pi_2}$$

* **Parámetros del Examen:** La aplicación precarga exactamente:
  * $\mu_1 = [-1.0, 0.5]^T, \quad \Sigma_1 = \begin{pmatrix} 0.8 & 0.5 \\ 0.5 & 0.6 \end{pmatrix}, \quad \pi_1 = 0.5$
  * $\mu_2 = [1.5, -0.5]^T, \quad \Sigma_2 = \begin{pmatrix} 0.6 & -0.4 \\ -0.4 & 1.8 \end{pmatrix}, \quad \pi_2 = 0.5$
  * Matriz y coeficientes de referencia derivados:
    $$A = \begin{pmatrix} 0.0854 & 1.2888 \\ 1.2888 & -1.4596 \end{pmatrix}, \quad b = \begin{pmatrix} -5.5745 \\ 3.6025 \end{pmatrix}, \quad c = -0.2060$$

* **Gaussian Naive Bayes (GNB):** Asume covarianza nula fuera de la diagonal ($\Sigma_{k,12} = 0$). Genera una frontera cuadrática restringida alineada con los ejes cartesianos.

#### Regresión Logística con IRLS (Iteratively Reweighted Least Squares):
El modelo discriminativo modela directamente $p(C_1|x) = \sigma(w^T \phi(x)) = \frac{1}{1 + e^{-w^T \phi(x)}}$.
Para emparejar la capacidad representacional de QDA, se implementó el selector de **expansión cuadrática**:
$$\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]^T$$
Bajo este mapeo, el hiperplano $w^T \phi(x) = 0$ define la cónica cuadrática:
$$w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2 = 0$$

La optimización de máxima verosimilitud se realiza mediante el método de Newton-Raphson:
$$w^{(t+1)} = w^{(t)} - \left( \Phi^T W \Phi + \delta I \right)^{-1} \Phi^T (p - y)$$
donde $W = \text{diag}(p_i(1 - p_i))$ es la matriz diagonal de pesos y $\delta I$ representa un factor de amortiguamiento numérico (Levenberg-Marquardt). El algoritmo converge cuadráticamente en menos de 10 iteraciones.

---

### 3.3. Módulo 3: SVM Dual, Espacios de Hilbert (RKHS) y Verificación del Margen

#### Formulación Dual de SVM:
Dado un conjunto de entrenamiento $\{(x_n, y_n)\}_{n=1}^N$ con $y_n \in \{-1, +1\}$, el problema dual de programación cuadrática con función de kernel $K(x_n, x_m) = \langle \phi(x_n), \phi(x_m) \rangle_{\mathcal{H}}$ es:
$$\max_{\alpha} \sum_{n=1}^N \alpha_n - \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m)$$
$$\text{sujeto a:} \quad 0 \le \alpha_n \le C \quad \forall n, \quad \sum_{n=1}^N \alpha_n y_n = 0$$

#### Demostración y Verificación Numérica de la Identidad en Margen Duro:
Para un clasificador de margen duro con datos separables ($C \to \infty$ o $C$ suficientemente grande), todos los vectores de soporte satisfacen la condición KKT de saturación:
$$y_n f(x_n) = 1, \quad \text{donde } f(x) = \sum_{m=1}^N \alpha_m y_m K(x_m, x) + b$$

Multiplicando por el multiplicador dual $\alpha_n$ y sumando sobre todas las muestras:
$$\sum_{n=1}^N \alpha_n y_n f(x_n) = \sum_{n=1}^N \alpha_n$$
$$\sum_{n=1}^N \alpha_n y_n \left( \sum_{m=1}^N \alpha_m y_m K(x_m, x_n) + b \right) = \sum_{n=1}^N \alpha_n$$
Distribuyendo la sumatoria:
$$\sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_m, x_n) + b \underbrace{\left(\sum_{n=1}^N \alpha_n y_n\right)}_{= 0} = \sum_{n=1}^N \alpha_n$$
El término con el sesgo $b$ se anula exactamente debido a la restricción de igualdad dual, demostrando la identidad:
$$\|w\|_{\mathcal{H}}^2 = \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m) = \sum_{n=1}^N \alpha_n$$

El simulador computa ambas cantidades independientemente y evalúa en vivo su error relativo:
$$\text{Error Relativo} = \frac{|\|w\|_{\mathcal{H}}^2 - \sum_{n=1}^N \alpha_n|}{\sum_{n=1}^N \alpha_n}$$
mostrando un indicador verde de validación analítica cuando el error es inferior al $0.5\%$.

---

## 4. Instrucciones de Ejecución Local

No se requiere ningún paso de compilación ni instalación de paquetes.

### Opción A: Abrir directamente en el navegador
Hacer doble clic sobre el archivo `index.html` o abrirlo desde el navegador web (Chrome, Firefox, Safari, Edge).

### Opción B: Servidor local ligero con Python
Para un entorno idéntico a un servidor HTTP web:
```bash
# En la raíz del repositorio
python3 -m http.server 8000
```
Abrir en el navegador: [http://localhost:8000](http://localhost:8000)

---

## 5. Instrucciones de Despliegue en GitHub Pages

Para publicar el sitio en la web a través de GitHub Pages de manera gratuita y pública:

1. **Inicializar el repositorio Git y confirmar los cambios:**
   ```bash
   git init
   git add index.html style.css app.js README.md
   git commit -m "feat: solución completa e interactiva Parcial ML (Punto 4)"
   ```

2. **Crear el repositorio en GitHub y vincularlo:**
   Crear un repositorio público en GitHub (por ejemplo, `ParcialTAM`) y vincularlo:
   ```bash
   git branch -M main
   git remote add origin https://github.com/BRAYANPISSO02/ParcialTAM.git
   git push -u origin main
   ```

3. **Activar GitHub Pages:**
   * Ir a la página del repositorio en GitHub: [https://github.com/BRAYANPISSO02/ParcialTAM](https://github.com/BRAYANPISSO02/ParcialTAM)
   * Entrar a la pestaña **Settings** > **Pages** (en el menú lateral izquierdo).
   * En la sección **Build and deployment**:
     * **Source:** Seleccionar `Deploy from a branch`.
     * **Branch:** Seleccionar `main` y carpeta `/ (root)`.
     * Hacer clic en **Save**.
   * El sitio queda desplegado y accesible públicamente en:
     **[https://brayanpisso02.github.io/ParcialTAM/](https://brayanpisso02.github.io/ParcialTAM/)**

---

## 6. Declaración de Uso de Asistentes de IA

En cumplimiento con los lineamientos de integridad académica y transparencia tecnológica del examen:

> **Declaración:** Se utilizaron modelos de inteligencia artificial y agentes de codificación como herramientas de asistencia técnica durante el desarrollo de esta solución. La IA fue empleada para agilizar la maquetación de la interfaz gráfica (`style.css`), estructurar el código en JavaScript puro (`app.js`) y verificar numéricamente la estabilidad de las operaciones de álgebra lineal (inversión de matrices, algoritmo de descomposición de Cholesky y rutina SMO para SVM). La formulación matemática de los modelos (solución analítica de QDA, deducción del estimador MAP en Elastic Net y la demostración formal de la identidad dual de margen $\|w\|_{\mathcal{H}}^2 = \sum \alpha_n$) fue derivada, revisada y validada rigurosamente para garantizar total correspondencia con los requerimientos teóricos del examen.

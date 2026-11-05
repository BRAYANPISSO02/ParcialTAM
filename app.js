/**
 * ============================================================================
 * Laboratorio Interactivo de Aprendizaje de Máquina (app.js)
 * Implementación 100% Client-Side en Vanilla JS (Zero Dependencies)
 * 
 * Módulos implementados:
 * 1. PRNG determinista con soporte para números aleatorios normales (Box-Muller).
 * 2. Módulo 1: Elastic Net y MAP con Descenso por Coordenadas (Coordinate Descent).
 * 3. Módulo 2: Clasificación Bayesiana (QDA y GNB) vs Discriminativa (IRLS LogReg).
 * 4. Módulo 3: SVM Dual con solver SMO simplificado e Identidades de Margen en RKHS.
 * 5. Mapeo UI, sincronización de sliders y renderizado gráfico Canvas 2D.
 * ============================================================================
 */

'use strict';

// ============================================================================
// 1. GENERADOR DE NÚMEROS PSEUDOALEATORIOS (PRNG) Y ÁLGEBRA LINEAL BÁSICA
// ============================================================================

class PRNG {
  constructor(seed = 42) {
    this.seed = seed;
  }

  setSeed(seed) {
    this.seed = seed;
  }

  // Algoritmo Mulberry32
  random() {
    let t = (this.seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Transformada de Box-Muller para distribución normal estándar N(0, 1)
  normal(mean = 0, std = 1) {
    let u1 = this.random();
    let u2 = this.random();
    while (u1 <= 1e-12) u1 = this.random(); // Evitar log(0)
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * std;
  }
}

const rng = new PRNG(42);

// Operaciones matriciales y de vectores
const Matrix = {
  // Inversión de matriz cuadrada 2x2
  inv2x2(M) {
    const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (Math.abs(det) < 1e-12) return null;
    const invDet = 1 / det;
    return [
      [ M[1][1] * invDet, -M[0][1] * invDet],
      [-M[1][0] * invDet,  M[0][0] * invDet]
    ];
  },

  det2x2(M) {
    return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  },

  // Descomposición de Cholesky para matriz 2x2 simétrica positiva definida
  cholesky2x2(M) {
    const l11 = Math.sqrt(Math.max(1e-10, M[0][0]));
    const l21 = M[1][0] / l11;
    const l22 = Math.sqrt(Math.max(1e-10, M[1][1] - l21 * l21));
    return [
      [l11, 0],
      [l21, l22]
    ];
  },

  // Inversión general de matriz NxN por eliminación Gauss-Jordan con pivoteo parcial
  invert(A) {
    const n = A.length;
    const M = A.map(row => [...row]);
    const I = Array.from({ length: n }, (_, i) => 
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
    );

    for (let i = 0; i < n; i++) {
      // Búsqueda de pivote máximo
      let maxEl = Math.abs(M[i][i]);
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(M[k][i]) > maxEl) {
          maxEl = Math.abs(M[k][i]);
          maxRow = k;
        }
      }

      if (maxEl < 1e-12) return null; // Matriz singular

      // Intercambio de filas
      [M[i], M[maxRow]] = [M[maxRow], M[i]];
      [I[i], I[maxRow]] = [I[maxRow], I[i]];

      // Normalizar fila pivote
      const pivot = M[i][i];
      for (let j = 0; j < n; j++) {
        M[i][j] /= pivot;
        I[i][j] /= pivot;
      }

      // Eliminar otras filas
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = M[k][i];
          for (let j = 0; j < n; j++) {
            M[k][j] -= factor * M[i][j];
            I[k][j] -= factor * I[i][j];
          }
        }
      }
    }
    return I;
  },

  // Multiplicación matriz-vector: M (n x m) * v (m x 1)
  matVecMul(M, v) {
    return M.map(row => row.reduce((sum, val, idx) => sum + val * v[idx], 0));
  },

  // Producto punto entre dos vectores
  dot(u, v) {
    return u.reduce((sum, val, idx) => sum + val * v[idx], 0);
  }
};

// ============================================================================
// 2. MÓDULO 1: ELASTIC NET Y ESTIMACIÓN MAP (PUNTO 1)
// ============================================================================

const Module1 = {
  data: null,

  init() {
    this.bindEvents();
    this.generateData();
  },

  bindEvents() {
    const ids = ['m1-n', 'm1-p', 'm1-sigma', 'm1-rho', 'm1-sparsity', 'm1-lambda', 'm1-alpha'];
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', () => {
          document.getElementById(`${id}-val`).textContent = parseFloat(el.value).toFixed(
            id.includes('n') || id.includes('p') || id.includes('sparsity') ? 0 : 2
          );
          if (id === 'm1-p') {
            const pVal = parseInt(el.value, 10);
            const sparsityEl = document.getElementById('m1-sparsity');
            sparsityEl.max = pVal;
            if (parseInt(sparsityEl.value, 10) > pVal) sparsityEl.value = pVal;
            document.getElementById('m1-sparsity-val').textContent = sparsityEl.value;
          }
          if (id === 'm1-lambda' || id === 'm1-alpha') {
            this.updateEstimatesAndRender();
          } else {
            this.generateData();
          }
        });
      }
    });

    document.getElementById('m1-btn-resample')?.addEventListener('click', () => {
      this.generateData();
    });

    document.querySelectorAll('input[name="m1-path-mode"]').forEach(r => {
      r.addEventListener('change', () => this.renderPath());
    });
  },

  generateData() {
    const N = parseInt(document.getElementById('m1-n').value, 10);
    const P = parseInt(document.getElementById('m1-p').value, 10);
    const sigma = parseFloat(document.getElementById('m1-sigma').value);
    const rho = parseFloat(document.getElementById('m1-rho').value);
    const s = parseInt(document.getElementById('m1-sparsity').value, 10);

    // Vector de pesos verdaderos wTrue
    const wTrue = new Array(P).fill(0);
    for (let j = 0; j < s; j++) {
      wTrue[j] = (j % 2 === 0 ? 1 : -1) * (1.5 + 0.5 * j);
    }

    // Generar matriz de covarianza Toeplitz: Sigma_ij = rho^|i - j|
    const Sigma = Array.from({ length: P }, (_, i) =>
      Array.from({ length: P }, (_, j) => Math.pow(rho, Math.abs(i - j)))
    );

    // Factorización de Cholesky para dimensiones pequeñas
    const L = Array.from({ length: P }, () => new Array(P).fill(0));
    for (let i = 0; i < P; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
        if (i === j) {
          L[i][j] = Math.sqrt(Math.max(1e-10, Sigma[i][i] - sum));
        } else {
          L[i][j] = (Sigma[i][j] - sum) / L[j][j];
        }
      }
    }

    // Generar X ~ N(0, Sigma)
    const X = [];
    for (let i = 0; i < N; i++) {
      const z = Array.from({ length: P }, () => rng.normal(0, 1));
      const row = new Array(P).fill(0);
      for (let j = 0; j < P; j++) {
        for (let k = 0; k <= j; k++) {
          row[j] += L[j][k] * z[k];
        }
      }
      X.push(row);
    }

    // Estandarizar columnas de X (media 0, varianza 1)
    for (let j = 0; j < P; j++) {
      let mean = 0;
      for (let i = 0; i < N; i++) mean += X[i][j];
      mean /= N;
      let variance = 0;
      for (let i = 0; i < N; i++) {
        X[i][j] -= mean;
        variance += X[i][j] * X[i][j];
      }
      const std = Math.sqrt(variance / N) || 1;
      for (let i = 0; i < N; i++) X[i][j] /= std;
    }

    // Generar y = X * wTrue + epsilon
    const y = [];
    for (let i = 0; i < N; i++) {
      let yVal = 0;
      for (let j = 0; j < P; j++) yVal += X[i][j] * wTrue[j];
      yVal += rng.normal(0, sigma);
      y.push(yVal);
    }

    // Centrar y
    let yMean = y.reduce((a, b) => a + b, 0) / N;
    for (let i = 0; i < N; i++) y[i] -= yMean;

    // Calcular OLS estimado para P = 2
    let wOls = [0, 0];
    if (P === 2) {
      // (X^T X)^-1 X^T y
      let xtx = [[0, 0], [0, 0]];
      let xty = [0, 0];
      for (let i = 0; i < N; i++) {
        xtx[0][0] += X[i][0] * X[i][0];
        xtx[0][1] += X[i][0] * X[i][1];
        xtx[1][0] += X[i][1] * X[i][0];
        xtx[1][1] += X[i][1] * X[i][1];
        xty[0] += X[i][0] * y[i];
        xty[1] += X[i][1] * y[i];
      }
      const invXtx = Matrix.inv2x2(xtx);
      if (invXtx) {
        wOls = Matrix.matVecMul(invXtx, xty);
      }
    }

    this.data = { N, P, sigma, rho, s, wTrue, X, y, wOls };
    this.updateEstimatesAndRender();
  },

  // Algoritmo de Descenso por Coordenadas (Coordinate Descent)
  // Minimiza: 1/(2N) * ||y - Xw||^2 + lambda * [ alpha ||w||_1 + (1 - alpha)/2 ||w||_2^2 ]
  coordinateDescent(X, y, lambda, alpha, maxIter = 250, tol = 1e-6, wInit = null) {
    const N = X.length;
    const P = X[0].length;
    const w = wInit ? [...wInit] : new Array(P).fill(0);

    // Precalcular la norma de cada columna: z_j = (1/N) * sum_i X_ij^2 = 1 (ya estandarizado)
    const z = new Array(P).fill(0);
    for (let j = 0; j < P; j++) {
      for (let i = 0; i < N; i++) z[j] += X[i][j] * X[i][j];
      z[j] /= N;
    }

    // Residual inicial r_i = y_i - X_i * w
    const r = [...y];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < P; j++) r[i] -= X[i][j] * w[j];
    }

    for (let iter = 0; iter < maxIter; iter++) {
      let maxChange = 0;

      for (let j = 0; j < P; j++) {
        const oldWj = w[j];

        // Calcular rho_j = (1/N) * X_j^T (r + X_j * w_j) = (1/N) * X_j^T r + z_j * w_j
        let dotXr = 0;
        for (let i = 0; i < N; i++) dotXr += X[i][j] * r[i];
        const rho_j = (dotXr / N) + z[j] * oldWj;

        // Operador Soft-Thresholding: S(z, gamma) = sign(z) * max(0, |z| - gamma)
        const gamma = lambda * alpha;
        let newWj = 0;
        if (rho_j > gamma) {
          newWj = (rho_j - gamma) / (z[j] + lambda * (1 - alpha));
        } else if (rho_j < -gamma) {
          newWj = (rho_j + gamma) / (z[j] + lambda * (1 - alpha));
        } else {
          newWj = 0;
        }

        // Actualizar residual: r_i = r_i - X_ij * (newWj - oldWj)
        const delta = newWj - oldWj;
        if (Math.abs(delta) > 0) {
          for (let i = 0; i < N; i++) r[i] -= X[i][j] * delta;
        }

        w[j] = newWj;
        maxChange = Math.max(maxChange, Math.abs(delta));
      }

      if (maxChange < tol) break;
    }

    return w;
  },

  updateEstimatesAndRender() {
    if (!this.data) return;
    const lambda = parseFloat(document.getElementById('m1-lambda').value);
    const alpha = parseFloat(document.getElementById('m1-alpha').value);

    const wMap = this.coordinateDescent(this.data.X, this.data.y, lambda, alpha);
    this.data.wMap = wMap;

    // Actualizar resumen de métricas
    const summary = document.getElementById('m1-map-summary');
    if (summary) {
      summary.innerHTML = `
        <div><strong>OLS:</strong> [${this.data.wOls.map(v => v.toFixed(3)).join(', ')}]</div>
        <div><strong>MAP:</strong> [${wMap.slice(0, 2).map(v => v.toFixed(3)).join(', ')}]</div>
        <div><strong>Sparsity (w=0):</strong> ${wMap.filter(v => Math.abs(v) < 1e-4).length} de ${this.data.P}</div>
        <div><strong>Norma L1:</strong> ${wMap.reduce((s, v) => s + Math.abs(v), 0).toFixed(3)}</div>
        <div><strong>Norma L2:</strong> ${Math.sqrt(wMap.reduce((s, v) => s + v * v, 0)).toFixed(3)}</div>
      `;
    }

    this.renderContours(lambda, alpha);
    this.renderPath();
  },

  // Renderizar curvas de nivel para P = 2
  renderContours(lambda, alpha) {
    const canvas = document.getElementById('m1-contour-canvas');
    if (!canvas || this.data.P !== 2) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Límites de coordenadas en espacio w: [-3.0, 3.0]
    const range = 3.0;
    const toCanvasX = (w1) => width / 2 + (w1 / range) * (width * 0.45);
    const toCanvasY = (w2) => height / 2 - (w2 / range) * (height * 0.45);

    // Dibujar cuadrícula y ejes cartesianos
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let v = -3; v <= 3; v += 1) {
      ctx.moveTo(toCanvasX(v), 0);
      ctx.lineTo(toCanvasX(v), height);
      ctx.moveTo(0, toCanvasY(v));
      ctx.lineTo(width, toCanvasY(v));
    }
    ctx.stroke();

    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), height);
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(width, toCanvasY(0));
    ctx.stroke();

    // Rejilla de evaluación
    const gridRes = 70;
    const priorGrid = [];
    const likGrid = [];
    const postGrid = [];
    const N = this.data.N;
    const sigmaSq = this.data.sigma * this.data.sigma;

    for (let i = 0; i < gridRes; i++) {
      const w1 = -range + (2 * range * i) / (gridRes - 1);
      priorGrid[i] = [];
      likGrid[i] = [];
      postGrid[i] = [];
      for (let j = 0; j < gridRes; j++) {
        const w2 = -range + (2 * range * j) / (gridRes - 1);

        // -ln p(w) ~ lambda * [ alpha * (|w1| + |w2|) + (1-alpha)/2 * (w1^2 + w2^2) ]
        const prior = lambda * (alpha * (Math.abs(w1) + Math.abs(w2)) + 0.5 * (1 - alpha) * (w1 * w1 + w2 * w2));

        // -ln p(y|X, w) ~ 1/(2N) * sum_i (y_i - x_i1*w1 - x_i2*w2)^2
        let rss = 0;
        for (let k = 0; k < N; k++) {
          const res = this.data.y[k] - (this.data.X[k][0] * w1 + this.data.X[k][1] * w2);
          rss += res * res;
        }
        const lik = rss / (2 * N * sigmaSq);

        priorGrid[i][j] = prior;
        likGrid[i][j] = lik;
        postGrid[i][j] = lik + prior;
      }
    }

    // Función auxiliar para dibujar curvas de nivel básicas mediante thresholding discreto
    const drawContourLines = (grid, color, levels, isDashed = false) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      if (isDashed) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);

      levels.forEach(val => {
        ctx.beginPath();
        for (let i = 0; i < gridRes - 1; i++) {
          for (let j = 0; j < gridRes - 1; j++) {
            const v00 = grid[i][j];
            const v10 = grid[i + 1][j];
            const v01 = grid[i][j + 1];
            // Ver si cruza el nivel
            if ((v00 - val) * (v10 - val) < 0 || (v00 - val) * (v01 - val) < 0) {
              const w1 = -range + (2 * range * (i + 0.5)) / (gridRes - 1);
              const w2 = -range + (2 * range * (j + 0.5)) / (gridRes - 1);
              ctx.rect(toCanvasX(w1), toCanvasY(w2), 2, 2);
            }
          }
        }
        ctx.stroke();
      });
      ctx.setLineDash([]);
    };

    // Contornos de Prior (Ámbar / Naranja)
    drawContourLines(priorGrid, '#f59e0b', [0.15, 0.4, 0.8, 1.4, 2.2], true);

    // Contornos de Verosimilitud (Cian)
    const minLik = Math.min(...likGrid.flat());
    drawContourLines(likGrid, '#06b6d4', [minLik + 0.3, minLik + 0.9, minLik + 2.0, minLik + 3.8]);

    // Contornos de la Posterior (Púrpura)
    const minPost = Math.min(...postGrid.flat());
    drawContourLines(postGrid, '#8b5cf6', [minPost + 0.4, minPost + 1.2, minPost + 2.5]);

    // Dibujar punto OLS
    const olsX = toCanvasX(this.data.wOls[0]);
    const olsY = toCanvasY(this.data.wOls[1]);
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(olsX, olsY, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#f1f5f9';
    ctx.font = '11px sans-serif';
    ctx.fillText('OLS', olsX + 8, olsY - 4);

    // Dibujar punto MAP
    const mapX = toCanvasX(this.data.wMap[0]);
    const mapY = toCanvasY(this.data.wMap[1]);
    ctx.fillStyle = '#ef4444';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(mapX, mapY, 7, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.fillText('MAP', mapX + 9, mapY + 4);

    // Etiquetas de ejes
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('w₁', width - 20, toCanvasY(0) - 8);
    ctx.fillText('w₂', toCanvasX(0) + 8, 20);
  },

  // Renderizar Camino de Regularización (Path vs lambda)
  renderPath() {
    const canvas = document.getElementById('m1-path-canvas');
    if (!canvas || !this.data) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const mode = document.querySelector('input[name="m1-path-mode"]:checked')?.value || 'comparison';
    const numPoints = 60;
    const lambdaMax = 2.5;
    const lambdaMin = 0.005;

    // Generar valores de lambda en escala logarítmica
    const lambdas = [];
    for (let i = 0; i < numPoints; i++) {
      const frac = i / (numPoints - 1);
      lambdas.push(lambdaMax * Math.pow(lambdaMin / lambdaMax, frac));
    }

    const padLeft = 55, padRight = 30, padTop = 25, padBottom = 35;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Dibujar ejes
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    // Línea cero w = 0
    const zeroY = padTop + plotH / 2;
    ctx.strokeStyle = '#1e293b';
    ctx.beginPath();
    ctx.moveTo(padLeft, zeroY);
    ctx.lineTo(padLeft + plotW, zeroY);
    ctx.stroke();

    const currentAlpha = parseFloat(document.getElementById('m1-alpha').value);
    const maxWeightVal = 2.5;

    const toPlotX = (idx) => padLeft + (idx / (numPoints - 1)) * plotW;
    const toPlotY = (wVal) => zeroY - (wVal / maxWeightVal) * (plotH * 0.45);

    if (mode === 'comparison') {
      // Comparar w_1 bajo Lasso (alpha=1), Elastic Net (alpha=current) y Ridge (alpha=0)
      const series = [
        { name: 'Lasso (α=1.0)', alpha: 1.0, color: '#f43f5e', dash: [] },
        { name: `Elastic Net (α=${currentAlpha.toFixed(2)})`, alpha: currentAlpha, color: '#3b82f6', dash: [] },
        { name: 'Ridge (α=0.0)', alpha: 0.0, color: '#10b981', dash: [4, 4] }
      ];

      series.forEach(s => {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.setLineDash(s.dash);
        ctx.beginPath();

        let wWarm = null;
        for (let i = 0; i < numPoints; i++) {
          const l = lambdas[i];
          const wSol = this.coordinateDescent(this.data.X, this.data.y, l, s.alpha, 150, 1e-5, wWarm);
          wWarm = wSol;
          const x = toPlotX(i);
          const y = toPlotY(wSol[0]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
      ctx.setLineDash([]);

      // Leyenda
      let legX = padLeft + 15;
      series.forEach((s, idx) => {
        ctx.fillStyle = s.color;
        ctx.fillRect(legX, padTop + 8, 14, 3);
        ctx.fillStyle = '#cbd5e1';
        ctx.font = '11px sans-serif';
        ctx.fillText(s.name, legX + 18, padTop + 12);
        legX += 160;
      });

    } else {
      // Dibujar todas las variables w_j para el modelo actual
      const P = this.data.P;
      const palette = ['#3b82f6', '#f43f5e', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16'];
      
      const allPaths = Array.from({ length: P }, () => []);
      let wWarm = null;
      for (let i = 0; i < numPoints; i++) {
        const l = lambdas[i];
        const wSol = this.coordinateDescent(this.data.X, this.data.y, l, currentAlpha, 150, 1e-5, wWarm);
        wWarm = wSol;
        for (let j = 0; j < P; j++) allPaths[j].push(wSol[j]);
      }

      for (let j = 0; j < P; j++) {
        ctx.strokeStyle = palette[j % palette.length];
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let i = 0; i < numPoints; i++) {
          const x = toPlotX(i);
          const y = toPlotY(allPaths[j][i]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }

    // Etiquetas de los ejes
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px sans-serif';
    ctx.fillText('Valor del Coeficiente w', 10, padTop - 8);
    ctx.fillText('← λ Mayor (Más regularización)               λ Menor →', padLeft + plotW * 0.25, padTop + plotH + 25);
  }
};

// ============================================================================
// 3. MÓDULO 2: CLASIFICACIÓN BAYESIANA Y DISCRIMINATIVA (PUNTO 2)
// ============================================================================

const Module2 = {
  // Parámetros canónicos del examen
  defaults: {
    N: 300,
    pi1: 0.50,
    mu1: [-1.0, 0.5],
    Sigma1: [[0.8, 0.5], [0.5, 0.6]],
    mu2: [1.5, -0.5],
    Sigma2: [[0.6, -0.4], [-0.4, 1.8]],
    // Coeficientes teóricos de referencia derivados del examen
    A_ref: [[0.0854, 1.2888], [1.2888, -1.4596]],
    b_ref: [-5.5745, 3.6025],
    c_ref: -0.2060
  },

  state: null,

  init() {
    this.restoreDefaults();
    this.bindEvents();
    this.generateDatasetAndTrain();
  },

  restoreDefaults() {
    this.state = JSON.parse(JSON.stringify(this.defaults));
    this.syncControlsToState();
  },

  syncControlsToState() {
    const s = this.state;
    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) {
        el.value = val;
        const badge = document.getElementById(`${id}-val`);
        if (badge) badge.textContent = parseFloat(val).toFixed(2);
      }
    };

    setVal('m2-n', s.N);
    setVal('m2-pi1', s.pi1);
    setVal('m2-mu11', s.mu1[0]);
    setVal('m2-mu12', s.mu1[1]);
    setVal('m2-s11', s.Sigma1[0][0]);
    setVal('m2-s12', s.Sigma1[0][1]);
    setVal('m2-s22', s.Sigma1[1][1]);

    setVal('m2-mu21', s.mu2[0]);
    setVal('m2-mu22', s.mu2[1]);
    setVal('m2-s2_11', s.Sigma2[0][0]);
    setVal('m2-s2_12', s.Sigma2[0][1]);
    setVal('m2-s2_22', s.Sigma2[1][1]);
  },

  bindEvents() {
    const inputs = [
      'm2-n', 'm2-pi1',
      'm2-mu11', 'm2-mu12', 'm2-s11', 'm2-s12', 'm2-s22',
      'm2-mu21', 'm2-mu22', 'm2-s2_11', 'm2-s2_12', 'm2-s2_22'
    ];

    inputs.forEach(id => {
      const el = document.getElementById(id);
      el?.addEventListener('input', () => {
        document.getElementById(`${id}-val`).textContent = parseFloat(el.value).toFixed(2);
        this.readControlsIntoState();
        this.generateDatasetAndTrain();
      });
    });

    document.getElementById('m2-irls-quad')?.addEventListener('change', () => {
      this.trainModels();
    });

    document.getElementById('m2-btn-reset-exam')?.addEventListener('click', () => {
      this.restoreDefaults();
      this.generateDatasetAndTrain();
    });

    document.getElementById('m2-btn-resample')?.addEventListener('click', () => {
      this.generateDatasetAndTrain();
    });
  },

  readControlsIntoState() {
    const s = this.state;
    s.N = parseInt(document.getElementById('m2-n').value, 10);
    s.pi1 = parseFloat(document.getElementById('m2-pi1').value);

    s.mu1 = [
      parseFloat(document.getElementById('m2-mu11').value),
      parseFloat(document.getElementById('m2-mu12').value)
    ];
    const s12 = parseFloat(document.getElementById('m2-s12').value);
    s.Sigma1 = [
      [parseFloat(document.getElementById('m2-s11').value), s12],
      [s12, parseFloat(document.getElementById('m2-s22').value)]
    ];

    s.mu2 = [
      parseFloat(document.getElementById('m2-mu21').value),
      parseFloat(document.getElementById('m2-mu22').value)
    ];
    const s2_12 = parseFloat(document.getElementById('m2-s2_12').value);
    s.Sigma2 = [
      [parseFloat(document.getElementById('m2-s2_11').value), s2_12],
      [s2_12, parseFloat(document.getElementById('m2-s2_22').value)]
    ];
  },

  generateDatasetAndTrain() {
    const s = this.state;
    const N1 = Math.round(s.N * s.pi1);
    const N2 = s.N - N1;

    const L1 = Matrix.cholesky2x2(s.Sigma1);
    const L2 = Matrix.cholesky2x2(s.Sigma2);

    const samples = [];

    // Muestreo Clase 1 (label = 1)
    for (let i = 0; i < N1; i++) {
      const z1 = rng.normal(0, 1);
      const z2 = rng.normal(0, 1);
      const x1 = s.mu1[0] + L1[0][0] * z1 + L1[0][1] * z2;
      const x2 = s.mu1[1] + L1[1][0] * z1 + L1[1][1] * z2;
      samples.push({ x: [x1, x2], y: 1 });
    }

    // Muestreo Clase 2 (label = 0 para regresión logística, 2 para QDA)
    for (let i = 0; i < N2; i++) {
      const z1 = rng.normal(0, 1);
      const z2 = rng.normal(0, 1);
      const x1 = s.mu2[0] + L2[0][0] * z1 + L2[0][1] * z2;
      const x2 = s.mu2[1] + L2[1][0] * z1 + L2[1][1] * z2;
      samples.push({ x: [x1, x2], y: 0 });
    }

    this.samples = samples;
    this.trainModels();
  },

  trainModels() {
    const samples = this.samples;
    const N = samples.length;
    const s = this.state;

    // 1. QDA: Parámetros analíticos con las covarianzas completas
    const invS1 = Matrix.inv2x2(s.Sigma1);
    const invS2 = Matrix.inv2x2(s.Sigma2);
    const detS1 = Matrix.det2x2(s.Sigma1);
    const detS2 = Matrix.det2x2(s.Sigma2);
    const pi1 = s.pi1;
    const pi2 = 1 - pi1;

    // Función discriminante delta_k(x) = -1/2 ln|Sigma_k| - 1/2 (x - mu_k)^T inv(Sigma_k) (x - mu_k) + ln(pi_k)
    const qdaScore = (x) => {
      const d1 = [x[0] - s.mu1[0], x[1] - s.mu1[1]];
      const d2 = [x[0] - s.mu2[0], x[1] - s.mu2[1]];

      const qf1 = d1[0] * (invS1[0][0] * d1[0] + invS1[0][1] * d1[1]) +
                  d1[1] * (invS1[1][0] * d1[0] + invS1[1][1] * d1[1]);

      const qf2 = d2[0] * (invS2[0][0] * d2[0] + invS2[0][1] * d2[1]) +
                  d2[1] * (invS2[1][0] * d2[0] + invS2[1][1] * d2[1]);

      const delta1 = -0.5 * Math.log(Math.max(1e-10, detS1)) - 0.5 * qf1 + Math.log(pi1);
      const delta2 = -0.5 * Math.log(Math.max(1e-10, detS2)) - 0.5 * qf2 + Math.log(pi2);
      return delta1 - delta2; // > 0 clasifica como Clase 1
    };

    // 2. Naive Bayes Gaussiano: Asumiendo matriz de covarianza diagonal (Sigma_12 = 0)
    const var11 = s.Sigma1[0][0], var12 = s.Sigma1[1][1];
    const var21 = s.Sigma2[0][0], var22 = s.Sigma2[1][1];

    const gnbScore = (x) => {
      const qf1 = Math.pow(x[0] - s.mu1[0], 2) / var11 + Math.pow(x[1] - s.mu1[1], 2) / var12;
      const qf2 = Math.pow(x[0] - s.mu2[0], 2) / var21 + Math.pow(x[1] - s.mu2[1], 2) / var22;

      const logDet1 = Math.log(var11 * var12);
      const logDet2 = Math.log(var21 * var22);

      const delta1 = -0.5 * logDet1 - 0.5 * qf1 + Math.log(pi1);
      const delta2 = -0.5 * logDet2 - 0.5 * qf2 + Math.log(pi2);
      return delta1 - delta2;
    };

    // 3. Regresión Logística con IRLS (Newton-Raphson)
    const useQuad = document.getElementById('m2-irls-quad')?.checked ?? true;
    
    // Mapeo de características phi(x)
    const phi = (x) => {
      if (useQuad) {
        return [1, x[0], x[1], x[0] * x[0], x[0] * x[1], x[1] * x[1]];
      }
      return [1, x[0], x[1]];
    };

    const d = useQuad ? 6 : 3;
    let wIRLS = new Array(d).fill(0);

    // Iteraciones Newton-Raphson
    for (let iter = 0; iter < 15; iter++) {
      // Calcular probabilidades p_i = sigmoid(w^T phi(x_i))
      const p = [];
      for (let i = 0; i < N; i++) {
        const phi_i = phi(samples[i].x);
        const z = Matrix.dot(wIRLS, phi_i);
        p.push(1 / (1 + Math.exp(-Math.max(-25, Math.min(25, z)))));
      }

      // Gradiente: g = Phi^T (p - y)
      const g = new Array(d).fill(0);
      for (let i = 0; i < N; i++) {
        const phi_i = phi(samples[i].x);
        const err = p[i] - samples[i].y;
        for (let j = 0; j < d; j++) g[j] += phi_i[j] * err;
      }

      // Hessiano: H = Phi^T W Phi + delta * I (regularización de Levenberg-Marquardt)
      const H = Array.from({ length: d }, () => new Array(d).fill(0));
      for (let i = 0; i < N; i++) {
        const phi_i = phi(samples[i].x);
        const w_weight = p[i] * (1 - p[i]);
        for (let j = 0; j < d; j++) {
          for (let k = 0; k < d; k++) {
            H[j][k] += phi_i[j] * phi_i[k] * w_weight;
          }
        }
      }
      // Ridge regularizer para estabilidad numérica
      for (let j = 0; j < d; j++) H[j][j] += 1e-4;

      const invH = Matrix.invert(H);
      if (!invH) break;

      // Paso de actualización: w_new = w - inv(H) * g
      const deltaW = Matrix.matVecMul(invH, g);
      let stepSize = 0;
      for (let j = 0; j < d; j++) {
        wIRLS[j] -= deltaW[j];
        stepSize += deltaW[j] * deltaW[j];
      }

      if (Math.sqrt(stepSize) < 1e-5) break;
    }

    const irlsScore = (x) => Matrix.dot(wIRLS, phi(x));

    // Evaluar métricas y renderizar
    this.models = { qdaScore, gnbScore, irlsScore, wIRLS, useQuad };
    this.computeMetrics();
    this.renderCanvas();
  },

  computeMetrics() {
    const samples = this.samples;
    const N = samples.length;
    const m = this.models;

    const evalModel = (scoreFn, isLogistic = false) => {
      let correct = 0;
      let logLoss = 0;
      for (let i = 0; i < N; i++) {
        const score = scoreFn(samples[i].x);
        const pred = score >= 0 ? 1 : 0;
        if (pred === samples[i].y) correct++;

        // Probabilidad p(C1|x)
        const p1 = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, score))));
        const p = samples[i].y === 1 ? p1 : 1 - p1;
        logLoss -= Math.log(Math.max(1e-12, p));
      }
      return {
        accuracy: (correct / N) * 100,
        loss: logLoss / N
      };
    };

    const metQDA = evalModel(m.qdaScore);
    const metGNB = evalModel(m.gnbScore);
    const metIRLS = evalModel(m.irlsScore, true);

    const tbody = document.getElementById('m2-metrics-table');
    if (tbody) {
      tbody.innerHTML = `
        <tr>
          <td><strong>QDA (Generativo Completo)</strong></td>
          <td>Cuadrática (Matrices $\\Sigma_k$ completas)</td>
          <td>11 parámetros ($\mu_k, \\Sigma_k, \\pi_k$)</td>
          <td><span class="badge-acc badge-acc-high">${metQDA.accuracy.toFixed(1)}%</span></td>
          <td>${metQDA.loss.toFixed(4)}</td>
        </tr>
        <tr>
          <td><strong>Naive Bayes Gaussiano</strong></td>
          <td>Cuadrática Eje-Alineada ($\Sigma_{12} = 0$)</td>
          <td>7 parámetros (independencia condicional)</td>
          <td><span class="badge-acc">${metGNB.accuracy.toFixed(1)}%</span></td>
          <td>${metGNB.loss.toFixed(4)}</td>
        </tr>
        <tr>
          <td><strong>Regresión Logística (IRLS)</strong></td>
          <td>${m.useQuad ? 'Cuadrática (Base $\\phi(x)$ Grado 2)' : 'Lineal (Plano)'}</td>
          <td>${m.useQuad ? '6 pesos ($w_0 \\dots w_5$)' : '3 pesos ($w_0, w_1, w_2$)'}</td>
          <td><span class="badge-acc badge-acc-high">${metIRLS.accuracy.toFixed(1)}%</span></td>
          <td>${metIRLS.loss.toFixed(4)}</td>
        </tr>
      `;
      // Renderizar KaTeX en la tabla si está disponible
      if (window.renderMathInElement) {
        window.renderMathInElement(tbody);
      }
    }
  },

  renderCanvas() {
    const canvas = document.getElementById('m2-bayes-canvas');
    if (!canvas || !this.samples) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Mapeo espacial: espacio x in [-4.5, 4.5] x [-3.5, 3.5]
    const xMin = -4.5, xMax = 4.5;
    const yMin = -3.5, yMax = 3.5;

    const toCanvasX = (x) => ((x - xMin) / (xMax - xMin)) * width;
    const toCanvasY = (y) => height - ((y - yMin) / (yMax - yMin)) * height;
    const toSpaceX = (cx) => xMin + (cx / width) * (xMax - xMin);
    const toSpaceY = (cy) => yMin + ((height - cy) / height) * (yMax - yMin);

    // 1. Dibujar Mapa de Calor de Posterior p(C1|x) de QDA
    const gridRes = 60;
    const cellW = width / gridRes;
    const cellH = height / gridRes;

    const qdaGrid = [];
    const gnbGrid = [];
    const irlsGrid = [];

    for (let i = 0; i < gridRes; i++) {
      qdaGrid[i] = [];
      gnbGrid[i] = [];
      irlsGrid[i] = [];
      const sx = toSpaceX((i + 0.5) * cellW);
      for (let j = 0; j < gridRes; j++) {
        const sy = toSpaceY((j + 0.5) * cellH);
        const qScore = this.models.qdaScore([sx, sy]);
        const gScore = this.models.gnbScore([sx, sy]);
        const iScore = this.models.irlsScore([sx, sy]);

        qdaGrid[i][j] = qScore;
        gnbGrid[i][j] = gScore;
        irlsGrid[i][j] = iScore;

        // Probabilidad posterior para color de fondo
        const p1 = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, qScore))));
        
        // Degradado suave: Azul (Clase 1) vs Rojo/Rosa (Clase 2)
        const r = Math.round(244 * (1 - p1) + 30 * p1);
        const g = Math.round(63 * (1 - p1) + 64 * p1);
        const b = Math.round(94 * (1 - p1) + 210 * p1);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.22)`;
        ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
      }
    }

    // 2. Extraer y dibujar las tres fronteras de decisión (Score = 0)
    const drawZeroContour = (grid, color, lineWidth = 2, isDashed = false) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      if (isDashed) ctx.setLineDash([6, 5]); else ctx.setLineDash([]);
      ctx.beginPath();

      for (let i = 0; i < gridRes - 1; i++) {
        for (let j = 0; j < gridRes - 1; j++) {
          const v00 = grid[i][j];
          const v10 = grid[i + 1][j];
          const v01 = grid[i][j + 1];

          if (v00 * v10 <= 0 || v00 * v01 <= 0) {
            const cx = (i + 0.5) * cellW;
            const cy = (j + 0.5) * cellH;
            ctx.rect(cx - 1, cy - 1, 3, 3);
          }
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // Frontera QDA (Azul continuo)
    drawZeroContour(qdaGrid, '#3b82f6', 2.5);

    // Frontera Naive Bayes (Ámbar discontinuo)
    drawZeroContour(gnbGrid, '#f59e0b', 2.0, true);

    // Frontera Regresión Logística IRLS (Verde Esmeralda)
    drawZeroContour(irlsGrid, '#10b981', 2.2);

    // 3. Dibujar Muestras
    this.samples.forEach(pt => {
      const cx = toCanvasX(pt.x[0]);
      const cy = toCanvasY(pt.x[1]);

      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      if (pt.y === 1) {
        ctx.fillStyle = '#3b82f6'; // Clase 1
        ctx.strokeStyle = '#bfdbfe';
      } else {
        ctx.fillStyle = '#f43f5e'; // Clase 2
        ctx.strokeStyle = '#fecdd3';
      }
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
    });

    // 4. Dibujar Centros de Masa mu_1 y mu_2
    const drawMean = (mu, label, color) => {
      const cx = toCanvasX(mu[0]);
      const cy = toCanvasY(mu[1]);
      ctx.fillStyle = color;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(label, cx + 9, cy + 4);
    };

    drawMean(this.state.mu1, 'μ₁', '#3b82f6');
    drawMean(this.state.mu2, 'μ₂', '#f43f5e');
  }
};

// ============================================================================
// 4. MÓDULO 3: SVM Y GEOMETRÍA DEL MARGEN EN RKHS (PUNTO 3)
// ============================================================================

const Module3 = {
  points: [],
  model: null,
  isDragging: false,
  draggedIdx: -1,

  init() {
    this.bindEvents();
    this.loadPreset('separable');
  },

  bindEvents() {
    const canvas = document.getElementById('m3-svm-canvas');
    if (!canvas) return;

    // Sliders de C y Gamma
    document.getElementById('m3-c')?.addEventListener('input', (e) => {
      document.getElementById('m3-c-val').textContent = e.target.value;
      this.trainAndRender();
    });

    document.getElementById('m3-gamma')?.addEventListener('input', (e) => {
      document.getElementById('m3-gamma-val').textContent = parseFloat(e.target.value).toFixed(2);
      this.trainAndRender();
    });

    document.getElementById('m3-kernel')?.addEventListener('change', (e) => {
      const isRBF = e.target.value === 'rbf';
      document.getElementById('m3-gamma-container').style.display = isRBF ? 'block' : 'none';
      this.trainAndRender();
    });

    // Presets
    document.querySelectorAll('.m3-preset').forEach(btn => {
      btn.addEventListener('click', () => {
        this.loadPreset(btn.dataset.preset);
      });
    });

    // Botón limpiar
    document.getElementById('m3-btn-clear')?.addEventListener('click', () => {
      this.points = [];
      this.trainAndRender();
    });

    // Interacción con mouse en Canvas: Añadir o arrastrar puntos
    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const cy = (e.clientY - rect.top) * (canvas.height / rect.height);
      const [sx, sy] = this.toSpaceCoords(cx, cy, canvas);

      // Ver si hizo clic cerca de un punto existente para arrastrarlo
      const clickRadius = 0.35;
      const foundIdx = this.points.findIndex(p => 
        Math.hypot(p.x[0] - sx, p.x[1] - sy) < clickRadius
      );

      if (foundIdx !== -1) {
        this.isDragging = true;
        this.draggedIdx = foundIdx;
      } else {
        // Añadir nuevo punto con la clase seleccionada
        const label = parseInt(document.querySelector('input[name="m3-click-class"]:checked')?.value || '1', 10);
        this.points.push({ x: [sx, sy], y: label });
        this.trainAndRender();
      }
    });

    canvas.addEventListener('mousemove', (e) => {
      if (!this.isDragging || this.draggedIdx === -1) return;
      const rect = canvas.getBoundingClientRect();
      const cx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const cy = (e.clientY - rect.top) * (canvas.height / rect.height);
      const [sx, sy] = this.toSpaceCoords(cx, cy, canvas);
      this.points[this.draggedIdx].x = [sx, sy];
      this.trainAndRender();
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
      this.draggedIdx = -1;
    });
  },

  toSpaceCoords(cx, cy, canvas) {
    const range = 4.0;
    const sx = ((cx - canvas.width / 2) / (canvas.width * 0.45)) * range;
    const sy = -((cy - canvas.height / 2) / (canvas.height * 0.45)) * range;
    return [sx, sy];
  },

  toCanvasCoords(sx, sy, canvas) {
    const range = 4.0;
    const cx = canvas.width / 2 + (sx / range) * (canvas.width * 0.45);
    const cy = canvas.height / 2 - (sy / range) * (canvas.height * 0.45);
    return [cx, cy];
  },

  loadPreset(type) {
    this.points = [];
    if (type === 'separable') {
      // Dos clusters perfectamente separables
      for (let i = 0; i < 16; i++) {
        this.points.push({ x: [rng.normal(-1.6, 0.45), rng.normal(1.2, 0.45)], y: 1 });
        this.points.push({ x: [rng.normal(1.6, 0.45), rng.normal(-1.2, 0.45)], y: -1 });
      }
    } else if (type === 'overlap') {
      // Clusters con solapamiento suave
      for (let i = 0; i < 20; i++) {
        this.points.push({ x: [rng.normal(-0.8, 0.8), rng.normal(0.6, 0.8)], y: 1 });
        this.points.push({ x: [rng.normal(0.8, 0.8), rng.normal(-0.6, 0.8)], y: -1 });
      }
    } else if (type === 'circles') {
      // Círculos concéntricos
      document.getElementById('m3-kernel').value = 'rbf';
      document.getElementById('m3-gamma-container').style.display = 'block';
      for (let i = 0; i < 24; i++) {
        const theta = (2 * Math.PI * i) / 24;
        const r1 = rng.normal(1.1, 0.15);
        this.points.push({ x: [r1 * Math.cos(theta), r1 * Math.sin(theta)], y: 1 });
        const r2 = rng.normal(2.6, 0.2);
        this.points.push({ x: [r2 * Math.cos(theta), r2 * Math.sin(theta)], y: -1 });
      }
    }
    this.trainAndRender();
  },

  // Kernel Evaluation
  kernel(x1, x2, type, gamma) {
    if (type === 'linear') {
      return x1[0] * x2[0] + x1[1] * x2[1];
    }
    // RBF / Gaussian: exp(-gamma * ||x1 - x2||^2)
    const distSq = Math.pow(x1[0] - x2[0], 2) + Math.pow(x1[1] - x2[1], 2);
    return Math.exp(-gamma * distSq);
  },

  // Optimizador SMO Simplificado para SVM Dual
  // Maximiza: sum(alpha_i) - 0.5 * sum_ij (alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
  // sujeto a: 0 <= alpha_i <= C y sum(alpha_i * y_i) = 0
  trainSMO(points, kernelType, C, gamma, maxPasses = 15, tol = 1e-3) {
    const N = points.length;
    if (N < 2) return null;

    const alpha = new Array(N).fill(0);
    let b = 0;

    // Precalcular matriz de Kernel para acelerar
    const K = Array.from({ length: N }, (_, i) =>
      Array.from({ length: N }, (_, j) => this.kernel(points[i].x, points[j].x, kernelType, gamma))
    );

    const f = (idx) => {
      let sum = 0;
      for (let j = 0; j < N; j++) {
        if (alpha[j] > 0) sum += alpha[j] * points[j].y * K[j][idx];
      }
      return sum + b;
    };

    let passes = 0;
    while (passes < maxPasses) {
      let numChangedAlphas = 0;

      for (let i = 0; i < N; i++) {
        const Ei = f(i) - points[i].y;

        // Comprobar violaciones de las condiciones KKT
        if ((points[i].y * Ei < -tol && alpha[i] < C) || (points[i].y * Ei > tol && alpha[i] > 0)) {
          // Seleccionar j != i aleatoriamente
          let j = Math.floor(rng.random() * (N - 1));
          if (j >= i) j++;

          const Ej = f(j) - points[j].y;
          const oldAlphaI = alpha[i];
          const oldAlphaJ = alpha[j];

          // Calcular cotas L y H
          let L = 0, H = C;
          if (points[i].y !== points[j].y) {
            L = Math.max(0, alpha[j] - alpha[i]);
            H = Math.min(C, C + alpha[j] - alpha[i]);
          } else {
            L = Math.max(0, alpha[i] + alpha[j] - C);
            H = Math.min(C, alpha[i] + alpha[j]);
          }

          if (L >= H) continue;

          // Derivada segunda: eta = 2*K_ij - K_ii - K_jj
          const eta = 2 * K[i][j] - K[i][i] - K[j][j];
          if (eta >= 0) continue; // Problema no estrictamente cóncavo

          // Actualizar alpha_j no recortado
          let newAlphaJ = oldAlphaJ - (points[j].y * (Ei - Ej)) / eta;

          // Recortar alpha_j al intervalo [L, H]
          if (newAlphaJ > H) newAlphaJ = H;
          else if (newAlphaJ < L) newAlphaJ = L;

          if (Math.abs(newAlphaJ - oldAlphaJ) < 1e-5) continue;

          // Actualizar alpha_i correspondiente
          const newAlphaI = oldAlphaI + points[i].y * points[j].y * (oldAlphaJ - newAlphaJ);

          // Actualizar sesgo b
          const b1 = b - Ei - points[i].y * (newAlphaI - oldAlphaI) * K[i][i] - points[j].y * (newAlphaJ - oldAlphaJ) * K[i][j];
          const b2 = b - Ej - points[i].y * (newAlphaI - oldAlphaI) * K[i][j] - points[j].y * (newAlphaJ - oldAlphaJ) * K[j][j];

          if (newAlphaI > 0 && newAlphaI < C) b = b1;
          else if (newAlphaJ > 0 && newAlphaJ < C) b = b2;
          else b = (b1 + b2) / 2;

          alpha[i] = newAlphaI;
          alpha[j] = newAlphaJ;
          numChangedAlphas++;
        }
      }

      if (numChangedAlphas === 0) passes++;
      else passes = 0;
    }

    // Identificar vectores de soporte
    const supportVectors = [];
    for (let i = 0; i < N; i++) {
      if (alpha[i] > 1e-4) {
        supportVectors.push({ idx: i, point: points[i], alpha: alpha[i] });
      }
    }

    return { alpha, b, supportVectors, K };
  },

  trainAndRender() {
    const kernelType = document.getElementById('m3-kernel')?.value || 'linear';
    const C = parseFloat(document.getElementById('m3-c')?.value || '1000');
    const gamma = parseFloat(document.getElementById('m3-gamma')?.value || '0.5');

    const result = this.trainSMO(this.points, kernelType, C, gamma);
    this.model = result;

    this.verifyIdentities(result, kernelType, C, gamma);
    this.renderCanvas(kernelType, gamma);
  },

  // Verificación Numérica de Identidades del Margen
  verifyIdentities(result, kernelType, C, gamma) {
    if (!result || !result.supportVectors.length) {
      document.getElementById('m3-w-norm-sq').textContent = '0.0000';
      document.getElementById('m3-sum-alpha').textContent = '0.0000';
      document.getElementById('m3-margin-width').textContent = '0.0000';
      document.getElementById('m3-diff-ratio').textContent = '0.00%';
      return;
    }

    const { alpha } = result;
    const N = this.points.length;

    // 1. ||w||_H^2 = sum_i sum_j alpha_i * alpha_j * y_i * y_j * K(x_i, x_j)
    let wNormSq = 0;
    let sumAlpha = 0;

    for (let i = 0; i < N; i++) {
      sumAlpha += alpha[i];
      if (alpha[i] > 1e-6) {
        for (let j = 0; j < N; j++) {
          if (alpha[j] > 1e-6) {
            wNormSq += alpha[i] * alpha[j] * this.points[i].y * this.points[j].y * result.K[i][j];
          }
        }
      }
    }

    const wNorm = Math.sqrt(Math.max(1e-12, wNormSq));
    const marginWidth = 2 / wNorm;

    // Error relativo entre ||w||_H^2 y sum(alpha_i)
    const diff = Math.abs(wNormSq - sumAlpha);
    const relError = sumAlpha > 0 ? (diff / sumAlpha) * 100 : 0;

    document.getElementById('m3-w-norm-sq').textContent = wNormSq.toFixed(4);
    document.getElementById('m3-sum-alpha').textContent = sumAlpha.toFixed(4);
    document.getElementById('m3-margin-width').textContent = marginWidth.toFixed(4);
    document.getElementById('m3-diff-ratio').textContent = `${relError.toFixed(3)}%`;

    const badge = document.getElementById('m3-identity-badge');
    if (badge) {
      if (C >= 500 && relError < 0.5) {
        badge.className = 'badge-verification badge-success';
        badge.innerHTML = `✓ Margen Duro Verificado Exactamente: $||w||_{\\mathcal{H}}^2 = \\sum \\alpha_n$ (Error < 0.5%)`;
      } else {
        badge.className = 'badge-verification badge-warning';
        badge.innerHTML = `Margen Suave con Variables de Holgura: $\\sum \\alpha_n \\ge ||w||_{\\mathcal{H}}^2$ (Diferencia atribuible a holgura $\\xi_n > 0$)`;
      }
      if (window.renderMathInElement) window.renderMathInElement(badge);
    }
  },

  renderCanvas(kernelType, gamma) {
    const canvas = document.getElementById('m3-svm-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Ejes cartesianos
    const [origX, origY] = this.toCanvasCoords(0, 0, canvas);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(origX, 0); ctx.lineTo(origX, height);
    ctx.moveTo(0, origY); ctx.lineTo(width, origY);
    ctx.stroke();

    // Si hay modelo entrenado, dibujar frontera f(x)=0 y márgenes f(x)=+-1
    if (this.model && this.points.length >= 2) {
      const gridRes = 60;
      const cellW = width / gridRes;
      const cellH = height / gridRes;
      const grid = [];

      for (let i = 0; i < gridRes; i++) {
        grid[i] = [];
        for (let j = 0; j < gridRes; j++) {
          const cx = (i + 0.5) * cellW;
          const cy = (j + 0.5) * cellH;
          const [sx, sy] = this.toSpaceCoords(cx, cy, canvas);

          // Evaluar f(x) = sum_k alpha_k * y_k * K(x_k, x) + b
          let score = this.model.b;
          for (let k = 0; k < this.points.length; k++) {
            if (this.model.alpha[k] > 1e-6) {
              score += this.model.alpha[k] * this.points[k].y * this.kernel(this.points[k].x, [sx, sy], kernelType, gamma);
            }
          }
          grid[i][j] = score;

          // Sombreado de decisión de fondo
          const p = 1 / (1 + Math.exp(-Math.max(-10, Math.min(10, score))));
          const r = Math.round(59 * p + 244 * (1 - p));
          const g = Math.round(130 * p + 63 * (1 - p));
          const b = Math.round(246 * p + 94 * (1 - p));
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.16)`;
          ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
        }
      }

      // Dibujar curvas de contorno f(x) = 0 y f(x) = +-1
      const drawIsoLevel = (targetVal, color, isDashed = false, lw = 2) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        if (isDashed) ctx.setLineDash([5, 4]); else ctx.setLineDash([]);
        ctx.beginPath();
        for (let i = 0; i < gridRes - 1; i++) {
          for (let j = 0; j < gridRes - 1; j++) {
            const v00 = grid[i][j] - targetVal;
            const v10 = grid[i + 1][j] - targetVal;
            const v01 = grid[i][j + 1] - targetVal;
            if (v00 * v10 <= 0 || v00 * v01 <= 0) {
              ctx.rect((i + 0.5) * cellW - 1, (j + 0.5) * cellH - 1, 3, 3);
            }
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      };

      // Decisión f(x) = 0
      drawIsoLevel(0, '#38bdf8', false, 2.5);

      // Márgenes f(x) = +1 y f(x) = -1
      drawIsoLevel(1.0, '#a855f7', true, 1.8);
      drawIsoLevel(-1.0, '#a855f7', true, 1.8);
    }

    // Dibujar Puntos de Muestra
    this.points.forEach((pt, idx) => {
      const [cx, cy] = this.toCanvasCoords(pt.x[0], pt.x[1], canvas);

      // Si es vector de soporte, resaltar con anillo dorado
      const isSV = this.model?.alpha[idx] > 1e-4;
      if (isSV) {
        ctx.strokeStyle = '#eab308';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      if (pt.y === 1) {
        ctx.fillStyle = '#3b82f6'; // Clase +1 (Azul)
        ctx.strokeStyle = '#93c5fd';
      } else {
        ctx.fillStyle = '#f43f5e'; // Clase -1 (Rojo)
        ctx.strokeStyle = '#fecdd3';
      }
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });
  }
};

// ============================================================================
// 5. CONTROLADOR GLOBAL Y GESTIÓN DE PESTAÑAS (TABS)
// ============================================================================

function initGlobalController() {
  // Manejo de Semilla Aleatoria
  const seedInput = document.getElementById('global-seed');
  const btnRandomSeed = document.getElementById('btn-random-seed');
  const btnResetDefaults = document.getElementById('btn-reset-defaults');

  const applySeed = (newSeed) => {
    seedInput.value = newSeed;
    rng.setSeed(newSeed);
    Module1.generateData();
    Module2.generateDatasetAndTrain();
    Module3.loadPreset('separable');
  };

  seedInput?.addEventListener('change', () => {
    const val = parseInt(seedInput.value, 10) || 42;
    applySeed(val);
  });

  btnRandomSeed?.addEventListener('click', () => {
    const randomSeed = Math.floor(Math.random() * 900000) + 100000;
    applySeed(randomSeed);
  });

  btnResetDefaults?.addEventListener('click', () => {
    applySeed(42);
    Module2.restoreDefaults();
    Module2.generateDatasetAndTrain();
  });

  // Cambio de Pestañas
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPanel = document.getElementById(btn.dataset.tab);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }

      // Re-renderizado de gráficos al cambiar a la pestaña visible
      if (btn.dataset.tab === 'tab-m1') {
        Module1.updateEstimatesAndRender();
      } else if (btn.dataset.tab === 'tab-m2') {
        Module2.renderCanvas();
      } else if (btn.dataset.tab === 'tab-m3') {
        Module3.trainAndRender();
      }
    });
  });
}

// Inicialización general cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
  initGlobalController();
  Module1.init();
  Module2.init();
  Module3.init();
});

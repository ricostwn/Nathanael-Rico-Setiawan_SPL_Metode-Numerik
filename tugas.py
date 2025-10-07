import math

# ==========================================
# Sistem persamaan dan fungsi iterasi
# ==========================================
def f1(x, y): 
    return x**2 + x*y - 10

def f2(x, y): 
    return y + 3*x*y**2 - 57

def g1B(x, y):
    val = 10 - x * y
    if val < 0:
        raise ValueError("sqrt domain error pada g1B (10 - xy < 0)")
    return math.sqrt(val)

def g2B(x, y):
    denom = 3 * x
    if denom == 0:
        raise ValueError("pembagian nol pada g2B (3x = 0)")
    val = (57 - y) / denom
    if val < 0:
        raise ValueError("sqrt domain error pada g2B ((57 - y)/(3x) < 0)")
    return math.sqrt(val)

# Turunan parsial untuk Newton-Raphson
def dux(x, y): return 2 * x + y
def duy(x, y): return x
def dvx(x, y): return 3 * y**2
def dvy(x, y): return 1 + 6 * x * y

# ==========================================
# Metode Iterasi Titik Tetap - Jacobi
# ==========================================
def iterasi_jacobi(x0, y0, epsilon=1e-6, max_iter=100):
    x = [x0]
    y = [y0]
    print("\n=== Iterasi Titik Tetap (Jacobi) ===")
    print("r\t x\t\t y\t\t deltaX\t\t deltaY")
    print("---------------------------------------------------")
    
    for i in range(max_iter):
        try:
            x_new = g1B(x[i], y[i])
            y_new = g2B(x[i], y[i])
        except Exception as e:
            print(f"{i}\t {x[i]:.6f}\t {y[i]:.6f}\t ----\t ----\t ERROR: {e}")
            break
        
        deltaX = abs(x_new - x[i])
        deltaY = abs(y_new - y[i])
        print(f"{i}\t {x[i]:.6f}\t {y[i]:.6f}\t {deltaX:.6f}\t {deltaY:.6f}")
        
        if deltaX < epsilon and deltaY < epsilon:
            break
        x.append(x_new)
        y.append(y_new)
    
    print(f"\nHasil akhir Jacobi: x = {x[-1]:.6f}, y = {y[-1]:.6f}")

# ==========================================
# Metode Iterasi Titik Tetap - Seidel
# ==========================================
def iterasi_seidel(x0, y0, epsilon=1e-6, max_iter=100):
    x = [x0]
    y = [y0]
    print("\n=== Iterasi Titik Tetap (Seidel) ===")
    print("r\t x\t\t y\t\t deltaX\t\t deltaY")
    print("---------------------------------------------------")
    
    for i in range(max_iter):
        x_new = g1B(x[i], y[i])
        y_new = g2B(x_new, y[i])  # Gunakan nilai x terbaru
        
        deltaX = abs(x_new - x[i])
        deltaY = abs(y_new - y[i])
        print(f"{i}\t {x[i]:.6f}\t {y[i]:.6f}\t {deltaX:.6f}\t {deltaY:.6f}")
        
        if deltaX < epsilon and deltaY < epsilon:
            break
        x.append(x_new)
        y.append(y_new)
    
    print(f"\nHasil akhir Seidel: x = {x[-1]:.6f}, y = {y[-1]:.6f}")

# ==========================================
# Metode Newton-Raphson
# ==========================================
def newton_raphson(x0, y0, epsilon=1e-6, max_iter=100):
    x = [x0]
    y = [y0]
    print("\n=== Iterasi Newton-Raphson ===")
    print("r\t x\t\t y\t\t deltaX\t\t deltaY")
    print("---------------------------------------------------")
    
    for i in range(max_iter):
        u = f1(x[i], y[i])
        v = f2(x[i], y[i])
        det = dux(x[i], y[i]) * dvy(x[i], y[i]) - duy(x[i], y[i]) * dvx(x[i], y[i])
        
        x_new = x[i] - (u * dvy(x[i], y[i]) - v * duy(x[i], y[i])) / det
        y_new = y[i] + (u * dvx(x[i], y[i]) - v * dux(x[i], y[i])) / det
        
        deltaX = abs(x_new - x[i])
        deltaY = abs(y_new - y[i])
        print(f"{i}\t {x[i]:.6f}\t {y[i]:.6f}\t {deltaX:.6f}\t {deltaY:.6f}")
        
        if deltaX < epsilon and deltaY < epsilon:
            break
        x.append(x_new)
        y.append(y_new)
    
    print(f"\nHasil akhir Newton-Raphson: x = {x[-1]:.6f}, y = {y[-1]:.6f}")

# ==========================================
# Metode Secant
# ==========================================
def secant(x0, y0, x1, y1, epsilon=1e-6, max_iter=100):
    print("\n=== Iterasi Secant ===")
    print("r\t x\t\t y\t\t deltaX\t\t deltaY")
    print("---------------------------------------------------")
    
    for i in range(max_iter):
        f1_x1y1 = f1(x1, y1)
        f1_x0y0 = f1(x0, y0)
        f2_x1y1 = f2(x1, y1)
        f2_x0y0 = f2(x0, y0)
        
        x2 = x1 - f1_x1y1 * (x1 - x0) / (f1_x1y1 - f1_x0y0)
        y2 = y1 - f2_x1y1 * (y1 - y0) / (f2_x1y1 - f2_x0y0)
        
        deltaX = abs(x2 - x1)
        deltaY = abs(y2 - y1)
        print(f"{i}\t {x1:.6f}\t {y1:.6f}\t {deltaX:.6f}\t {deltaY:.6f}")
        
        if deltaX < epsilon and deltaY < epsilon:
            break
        x0, y0 = x1, y1
        x1, y1 = x2, y2
    
    print(f"\nHasil akhir Secant: x = {x1:.6f}, y = {y1:.6f}")

# ==========================================
# EKSEKUSI SEMUA METODE
# ==========================================
x0, y0 = 1.5, 3.5
epsilon = 1e-6

iterasi_jacobi(x0, y0, epsilon)
iterasi_seidel(x0, y0, epsilon)
newton_raphson(x0, y0, epsilon)
secant(1.5, 3.5, 1.6, 3.4, epsilon)
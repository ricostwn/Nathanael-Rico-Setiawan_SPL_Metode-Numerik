# DEMONSTRASI: Sistem nonlinear dua variabel
# f1(x,y) = x^2 + x*y - 10
# f2(x,y) = y + 3*x*y^2 - 57
#
# Methods implemented:
# - Fixed-point iteration (Jacobi and Seidel) with g1B & g2B (sqrt-forms)
# - Newton-Raphson (2D)
# - Secant-like (Broyden) multivariate (treated as "Secant" method here)
#
# Initial guess: x0=1.5, y0=3.5 ; tol=1e-6
import math
import numpy as np

def f1(x,y): return x*x + x*y - 10.0
def f2(x,y): return y + 3.0*x*(y**2) - 57.0

# g1B and g2B (safe versions)
def g1B_safe(x,y):
    arg = 10.0 - x*y
    return None if arg < 0 else math.sqrt(arg)

def g2B_safe(x_next,y):
    arg = 57.0 - 3.0 * x_next * (y**2)
    return None if arg < 0 else math.sqrt(arg)

# Jacobi (both updates computed from previous iterates)
def fixed_point_jacobi_safe(g1, g2, x0, y0, tol=1e-6, maxiter=200):
    table = [(0, x0, y0, 0.0, 0.0)]
    x, y = x0, y0
    for r in range(1, maxiter+1):
        x_new = g1(x,y)
        # for Jacobi we use the previous x,y for both g1 and g2
        y_new = g2(x,y)
        if x_new is None or y_new is None:
            table.append((r, x_new, y_new, None, None))
            break
        dx = abs(x_new - x)
        dy = abs(y_new - y)
        table.append((r, x_new, y_new, dx, dy))
        x, y = x_new, y_new
        if dx < tol and dy < tol:
            break
    return table

# Seidel (use x_{r+1} when computing y_{r+1})
def fixed_point_seidel_safe(g1, g2, x0, y0, tol=1e-6, maxiter=200):
    table = [(0, x0, y0, 0.0, 0.0)]
    x, y = x0, y0
    for r in range(1, maxiter+1):
        x_new = g1(x,y)
        if x_new is None:
            table.append((r, x_new, None, None, None)); break
        y_new = g2(x_new,y)
        if y_new is None:
            table.append((r, x_new, y_new, None, None)); break
        dx = abs(x_new - x); dy = abs(y_new - y)
        table.append((r, x_new, y_new, dx, dy))
        x, y = x_new, y_new
        if dx < tol and dy < tol:
            break
    return table

# Newton-Raphson (2D)
def newton_raphson_safe(x0,y0,tol=1e-6,maxiter=200):
    table = [(0, x0, y0, 0.0, 0.0)]
    x, y = x0, y0
    for r in range(1, maxiter+1):
        fu = f1(x,y); fv = f2(x,y)
        J11 = 2*x + y; J12 = x
        J21 = 3*(y**2);   J22 = 1 + 6*x*y
        det = J11*J22 - J12*J21
        if abs(det) < 1e-14:
            table.append((r, None, None, None, None)); break
        # Solve J * [dx,dy] = -F
        dx = (-fu * J22 - (-fv) * J12) / det
        dy = (J11*(-fv) - J21*(-fu)) / det
        x_new, y_new = x + dx, y + dy
        table.append((r, x_new, y_new, abs(x_new-x), abs(y_new-y)))
        x, y = x_new, y_new
        if abs(x_new-x) < tol and abs(y_new-y) < tol:
            break
    return table

# Broyden's method (secant-like, multivariate)
def broyden_safe(x0,y0,tol=1e-6,maxiter=200):
    table = []
    x = np.array([x0,y0], dtype=float)
    def F(v): return np.array([f1(v[0],v[1]), f2(v[0],v[1])], dtype=float)
    B = np.eye(2)   # initial approximate Jacobian
    Fx = F(x)
    table.append((0, x[0], x[1], 0.0, 0.0))
    for r in range(1, maxiter+1):
        try:
            s = np.linalg.solve(B, -Fx)
        except np.linalg.LinAlgError:
            table.append((r, None, None, None, None)); break
        x_new = x + s
        Fx_new = F(x_new)
        y_vec = Fx_new - Fx
        denom = s.dot(s)
        if abs(denom) < 1e-14:
            B = np.eye(2)
        else:
            B = B + np.outer((y_vec - B.dot(s)), s) / denom
        table.append((r, x_new[0], x_new[1], abs(x_new[0]-x[0]), abs(x_new[1]-x[1])))
        x, Fx = x_new, Fx_new
        if abs(table[-1][3]) < tol and abs(table[-1][4]) < tol:
            break
    return table

# Utility to print a table
def print_table(tab, title):
    print(title); print("-"*80)
    print(f"{'r':>3} {'x':>15} {'y':>15} {'deltaX':>15} {'deltaY':>15}"); print("-"*80)
    for r,x,y,dx,dy in tab:
        xs = f"{x:.9f}" if (x is not None) else str(x)
        ys = f"{y:.9f}" if (y is not None) else str(y)
        dxs = f"{dx:.9f}" if dx is not None else str(dx)
        dys = f"{dy:.9f}" if dy is not None else str(dy)
        print(f"{r:3d} {xs:>15} {ys:>15} {dxs:>15} {dys:>15}")
    print("-"*80 + "\nConverged in {} iterations\n".format(len(tab)-1 if len(tab)>1 else 0))

# Run with x0=1.5, y0=3.5
x0,y0 = 1.5, 3.5
tol = 1e-6
tab_jacobi = fixed_point_jacobi_safe(g1B_safe, lambda x,y: g2B_safe(g1B_safe(x,y), y), x0,y0,tol=tol)
tab_seidel = fixed_point_seidel_safe(g1B_safe, g2B_safe, x0,y0,tol=tol)
tab_nr = newton_raphson_safe(x0,y0,tol=tol)
tab_broyden = broyden_safe(x0,y0,tol=tol)

print_table(tab_jacobi, "Jacobi (g1B, g2B)")
print_table(tab_seidel, "Seidel (g1B, g2B)")
print_table(tab_nr, "Newton-Raphson")
print_table(tab_broyden, "Broyden (Secant-like)")

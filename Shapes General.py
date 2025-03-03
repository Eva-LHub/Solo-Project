#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:44:11 2025

@author: evaloughridge
"""

import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


n_list = []
div_list = []
x_list = []
rs_list = []
p_list = []

# Circle and Triangle
x=sp.Symbol('x') 

r = x/(2*sp.pi)
Pc = 2*sp.pi*r
Ac = sp.pi*(r**2)

P = 2-x

def solve(Ac,Pc, P, n, x):
    #print(f"\nShape of {n} sides:")
    
    s = P/n
    a = s/(2*sp.tan(sp.pi/n))
    As = (P* a)/2 
    
    sum_cs = Ac + As
    diff_sum = sp.diff(sum_cs, x)
    x1 = sp.solve(diff_sum, x)
    
    #print(f"x1 = {x1}")
    
    ratio = sp.Eq(Pc / P, Ac / As)

    x2 = sp.solve(ratio, x)
    #print(f"x2 = {x2}")
    
    div_s = x2[1]/x1[0]
    #print(f"div= {div_s:.4f}")
    
    r_x = r.subs(x,x1[0])
    #(f"r_x = {r_x}")
    
    s_x = s.subs(x,x1[0])
    #print(f"s_x = {s_x}")
    
    r_s = r_x/s_x
    #print(f"Relation = {r_s}")
    
    percent = Ac/As

    rs_list.append(r_s)
    n_list.append(n)
    div_list.append(div_s)
    x_list.append(x1[0])
    p_list.append(percent)
    
    return x1, x2, div_s


for n in range(3,21,1):
    x1t, x2t, div_t = solve(Ac, Pc, P, n, x)

print("-------------------------------------------------------------------------")

#-------------------------------------------------------------------------


n_values_numeric = [float(x.evalf() if isinstance(x, sp.Basic) else x) for x in n_list]
rs_values_numeric = [float(y.evalf() if isinstance(y, sp.Basic) else y) for y in rs_list]
x_values_numeric = [float(y.evalf() if isinstance(y, sp.Basic) else y) for y in x_list]
p_values_numeric = [float(p.subs(x, x_value).evalf()) if isinstance(p, sp.Basic) else float(p) for p, x_value in zip(p_list, x_values_numeric)]

n_array = np.array(n_values_numeric)
p_array = np.array(p_values_numeric)
x_array = np.array(x_values_numeric)
"""
plt.plot(n_values_numeric, x_values_numeric, label='Minimum x Values',  linestyle=':',color = 'DarkMagenta', marker='o', mec = 'DarkMagenta', mfc = 'DarkMagenta')
plt.title("Minimum Values of Circumference of Circle (x)")
plt.xlabel("Number of Sides in Polygon (n)")
plt.ylabel("Length of Circumference (x)")
plt.grid(True)
plt.legend(loc = 'lower right')
plt.xticks(range(int(min(n_values_numeric)), int(max(n_values_numeric)) + 1))
plt.show()
plt.close()
"""
coefficients = np.polyfit(n_values_numeric, rs_values_numeric, 1)  
trendline = np.poly1d(coefficients)
print(f"Trendline equation:{trendline}")
x_trend = np.linspace(min(n_values_numeric), max(n_values_numeric), 1000)
y_trend = trendline(x_trend)
plt.plot(x_trend, y_trend, color='HotPink', label='Trendline', zorder=1)
plt.scatter(n_values_numeric, rs_values_numeric, label='Ratio of Length of Sides and Radius', zorder=2)
plt.xticks(range(int(min(n_values_numeric)), int(max(n_values_numeric)) + 1))
#plt.title("Relationship between Length of Sides and Radius")
plt.xlabel("Number of Sides in Polygon (n)", fontsize = 14)
plt.ylabel("Value of Ratio", fontsize = 14)
plt.grid(True, zorder=0)
plt.legend(fontsize = 12)
plt.show()
plt.close()

"""
plt.plot(n_values_numeric, p_values_numeric , label='Percentages',  linestyle=':',color = 'PaleVioletRed', marker='o', mec = 'PaleVioletRed', mfc = 'PaleVioletRed')
plt.title("Percentage of the Polygon Filled by Incircle")
plt.xlabel("Number of Sides in Polygon (n)")
plt.ylabel("Percentage of the Polygon Filled (%)")
plt.grid(True)
plt.legend(loc = 'lower right')
plt.xticks(range(int(min(n_values_numeric)), int(max(n_values_numeric)) + 1))
plt.show()
plt.close()
"""
print("-------------------------------------------------------------------------")


x=sp.Symbol('x') #incircle
y=sp.Symbol('y') #excircle
z= 1-x-y # n-polygon
x3_list = []
y3_list = []
z3_list = []
r1_list = []
r2_list = []


def three_piece(r1,ss,r2,n):
    theta = ((n-2)*(sp.pi))/(2*n)
    ratio_1 = sp.Eq(r1,(ss/2)*sp.tan(theta))
    x1_3 = sp.solve(ratio_1, x)
    ratio_2 = sp.Eq(r2, ss/(2*sp.cos(theta)))
    x2_3 = sp.solve(ratio_2, x)
    y_sol = sp.Eq(x1_3[0], x2_3[0])
    y1 = sp.solve(y_sol, y)
    
    y_value = y1[0].evalf()  
    x_value = x1_3[0].subs(y, y_value).evalf()  
    z_value = z.subs({x: x_value, y: y_value}).evalf()
    
    x3_list.append(x_value)
    y3_list.append(y_value)
    z3_list.append(z_value)
    return x_value, y_value, z_value
    
for n in range(3,21,1):
    r1 = x/(2*sp.pi)
    ss = z/n
    r2 = y/(2*sp.pi)
    x_value, y_value, z_value = three_piece(r1,ss,r2,n)
    
    r1_x = r1.subs(x,x_value).evalf() 
    r2_y = r2.subs(y,y_value).evalf() 
    r1_list.append(r1_x)
    r2_list.append(r2_y)


plt.plot(n_list, x3_list, label='Incircle Circumference',  linestyle=':',color = 'SteelBlue', marker='.', mec = 'SteelBlue', mfc = 'SteelBlue')
plt.plot(n_list, y3_list, label='Excircle Circumference',  linestyle=':',color = 'Green', marker='.', mec = 'Green', mfc = 'Green')
plt.plot(n_list, z3_list, label='n-polygon Perimeter',  linestyle=':',color = 'Magenta', marker='.', mec = 'Magenta', mfc = 'Magenta')
#plt.title("Minimum Perimeter Values for a Wire of Unit Length")
plt.xlabel("Number of Sides in Polygon (n)", fontsize = 14)
plt.ylabel("Length", fontsize = 14)
plt.grid(True)
plt.legend(loc = 'upper right', fontsize = 12)
plt.xticks(range(int(min(n_list)), int(max(n_list)) + 1))
plt.show()
plt.close()    

rel_r = np.array(r1_list)/np.array(r2_list)
"""
plt.plot(n_list, rel_r, label='Relationship',  linestyle=':',color = 'Tomato', marker='o', mec = 'Tomato', mfc = 'Tomato')
plt.title("Ratio of Incircle & Excircle Radii")
plt.xlabel("Number of Sides in Polygon (n)")
plt.ylabel("Value")
plt.grid(True)
plt.legend(loc = 'lower right')
plt.xticks(range(int(min(n_list)), int(max(n_list)) + 1))
plt.show()
plt.close() 
"""
#-------------------------------------------------------------------------

# Define the fitting function: 1 - A/n^k
def model(n, L, A, k):
    return L - A / n**k


params_1, covariance_1 = opt.curve_fit(model, n_array, p_array, p0=[1,0.1, 0.1])

L_fit_1, A_fit_1, k_fit_1 = params_1

n_fit_1 = np.linspace(min(n_array), max(n_array), 100)
p_fit_1 = model(n_fit_1, L_fit_1, A_fit_1, k_fit_1)
"""
plt.plot(n_array, p_array, 'o', label='Original Data',linestyle=':', color='PaleVioletRed')

plt.plot(n_fit_1, p_fit_1, label=f'Fitted Curve: ${L_fit_1:.2f} - {A_fit_1:.2f}/n^{{{k_fit_1:.2f}}}$', linestyle='-', color='green')

plt.title("Percentage of the Polygon Filled by Incircle")
plt.xlabel("Number of Sides in Polygon (n)")
plt.ylabel("Percentage of the Polygon Filled (%)")
plt.grid(True)
plt.legend(loc='lower right')
plt.xticks(range(int(min(n_array)), int(max(n_array)) + 1))
plt.show()
"""
print("Curve Fit Parameters: L = {:.2f} A = {:.2f}, k = {:.2f}".format(*params_1))


print("-------------------------------------------------------------------------")

#-------------------------------------------------------------------------

params_2, covariance_2 = opt.curve_fit(model, n_array, x_array, p0=[0.5, 0.1, 0.1])

L_fit_2, A_fit_2, k_fit_2 = params_2

n_fit_2 = np.linspace(min(n_array), max(n_array), 100)
x_fit_2 = model(n_fit_2, L_fit_2, A_fit_2, k_fit_2)

"""
plt.plot(n_array, x_array, 'o', label='Original Data', linestyle=':',color='DarkMagenta')

plt.plot(n_fit_2, x_fit_2, label=f'Fitted Curve: ${L_fit_2:.2f} - {A_fit_2:.2f}/n^{{{k_fit_2:.2f}}}$', linestyle='-', color='green')

plt.title("Minimum Values of Circumference of Circle (x)")
plt.xlabel("Number of Sides in Polygon (n)")
plt.ylabel("Length of Circumference (x)")
plt.grid(True)
plt.legend(loc = 'lower right')
plt.xticks(range(int(min(n_array)), int(max(n_array)) + 1))
plt.show()
plt.close()
"""
print("Curve Fit Parameters: L = {:.2f} A = {:.2f}, k = {:.2f}".format(*params_2))

print("-------------------------------------------------------------------------")

#-------------------------------------------------------------------------

params_3, covariance_3 = opt.curve_fit(model, n_array, rel_r, p0=[1, 0.1, 0.1])

L_fit_3, A_fit_3, k_fit_3 = params_3

n_fit = np.linspace(min(n_array), max(n_array), 100)
x_fit_3 = model(n_fit, L_fit_3, A_fit_3, k_fit_3)

"""
plt.plot(n_array, rel_r, 'o', label='Original Data', linestyle=':',color='Tomato')

plt.plot(n_fit, x_fit_3, label=f'Fitted Curve: ${L_fit_3:.2f} - {A_fit_3:.2f}/n^{{{k_fit_3:.2f}}}$', linestyle='-', color='green')

plt.title("Ratio of Incircle & Excircle Radii")
plt.xlabel("Number of Sides in Polygon (n)")
plt.ylabel("Value")
plt.grid(True)
plt.legend(loc = 'lower right')
plt.xticks(range(int(min(n_list)), int(max(n_list)) + 1))
plt.show()
plt.close()
"""
print("Curve Fit Parameters: L = {:.2f} A = {:.2f}, k = {:.2f}".format(*params_3))

print("-------------------------------------------------------------------------")

#-------------------------------------------------------------------------


correlation = np.corrcoef(p_fit_1, x_fit_2)[0, 1]
print(f"Pearson Correlation Coefficient p & x: {correlation}")
correlation = np.corrcoef(p_fit_1, x_fit_3)[0, 1]
print(f"Pearson Correlation Coefficient p & x1: {correlation}")
correlation = np.corrcoef(x_fit_2, x_fit_3)[0, 1]
print(f"Pearson Correlation Coefficient x & x1: {correlation}")



#plotting the curve fits
plt.plot(n_fit_2, x_fit_2, label='Minimum $x$ Value for 2 unit wire', linestyle=':', color='magenta')
plt.plot(n_fit_1, p_fit_1, label='Percentage of Polygon Filled by Incircle', linestyle=':', color='darkorange')
plt.plot(n_fit_2, x_fit_3, label='Ratio of Incircle & Excircle Radii', linestyle=':', color='dodgerblue')

p_act_list = []
x2_act_list = []
x3_act_list = []


for n in range(3,21,1):
    p_act = model(n,L_fit_1,A_fit_1,k_fit_1)
    x2_act = model(n,L_fit_2,A_fit_2,k_fit_2)
    x3_act = model(n,L_fit_3,A_fit_3,k_fit_3)
    
    p_act_list.append(p_act)
    x2_act_list.append(x2_act)
    x3_act_list.append(x3_act)
    
plt.plot(n_array,np.array(p_act_list), '.',color='darkorange' )
plt.plot(n_array,np.array(x2_act_list), '.',color='magenta' )
plt.plot(n_array,np.array(x3_act_list), '.',color='dodgerblue' )
    
#plt.title("Relationship to Number of Sides in Polygon")
plt.xlabel("Number of Sides in Polygon (n)", fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.grid(True)
plt.legend(loc = 'lower right', fontsize = 12)
plt.xticks(range(int(min(n_array)), int(max(n_array)) + 1))
plt.show()

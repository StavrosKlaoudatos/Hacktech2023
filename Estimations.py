# -*- coding: utf-8 -*-
"""Hacktech 2023.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NHCDwHU8jI8gEzFGFqwOVf559MxqosEx

# **Hacktech Code**


---


**Ania Freymond**  

**David Hou**  

**Toyesh Jayaswal**  

**Stavros Klaoudatos**
"""

import numpy as np
import matplotlib.pyplot as plt

cost_kwh = 0.29
hours = [12,1,2,3,4,5]
total_hours_active = len(hours)

footprint_factor = 0.85
velocity = 20 #m/s

light_spacing = 200 #m
unit_length = 1000  #m This is how much we control

unit_time = unit_length/velocity
print(unit_time)



lights_per_unit_length =  2*(unit_length/light_spacing) #Two sides

P_consumption = 20 #watts
Light_consumnption = 450 #watts

COPL = total_hours_active*(Light_consumnption*lights_per_unit_length) * 365 * 100#Consumption_per_year for 100 km

COPL_cost = COPL/1000*cost_kwh

print(COPL,"Watt Hours per year per 100km of Highway")

cars_space = []
time_active_space = []
time_saved = []
energy_used_per_year_per_100kilometers = []
money_program = []
for i in range(163):
  cars_space.append(i)
  time_active_space.append(unit_time * i)
  time_saved.append(3600-unit_time * i)

  energy_used_program = total_hours_active*(((3600-time_saved[i])/3600)*(Light_consumnption*lights_per_unit_length)+P_consumption) * 365 * 100

  

  energy_used_per_year_per_100kilometers.append(energy_used_program)
  money_program.append(cost_kwh *energy_used_program/1000)

n=3
labels= ['Always On','Our program']
values = [COPL_cost,money_program[n]]
fig = plt.figure(figsize = (10, 5))
 

plt.bar(labels, values, color ='maroon',
        width = 0.4)
 

plt.ylabel("Money Spent on Energy per 100km of Highway per Year")
plt.title("Comparison of our program with the current model: Our Program is " + str(100*((values[0]-values[1])/values[0])) + "% Cheaper")
plt.show()
print(money_program[n])
print("Assuming 140,000 miles of Highways with these evenly spaced out lights, every year, if this model was implemented, we would save: ", 2225*(COPL_cost - money_program[n]),"$",", which corresponds to saving: ",2225/1000*(COPL_cost - money_program[n])/cost_kwh*footprint_factor,"Metric Tons of CO2")
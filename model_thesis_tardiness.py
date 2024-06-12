import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import gurobipy as gp
from gurobipy import GRB
import random
import time


random.seed(103093)

def data_callback(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        gap = (abs(cur_bd - cur_obj)/abs(cur_obj))*100
        
        # Change in obj value or bound?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._gap = gap
            model._data.append([time.time() - model._start, cur_obj, cur_bd, gap])

def plotGap(data):
        dfResult = pd.DataFrame(data, columns=['time', 'cur_obj','cur_bd','gap'])
        dfResult = dfResult.drop(dfResult[dfResult.cur_obj >= 100000000].index)
        
        fig, axes = plt.subplots()
        
        axes.set_xlabel('time')
        axes.set_ylabel('value')
        axes.set_xlim(dfResult['time'].values.min(), dfResult['time'].values.max())
        axes.set_ylim(0, dfResult['cur_obj'].values.max() * 1.1)
        line1, = axes.plot(dfResult['time'].values, dfResult['cur_obj'].values, color = 'navy', label='Current ObjValue')    
        line2, = axes.plot(dfResult['time'].values, dfResult['cur_bd'].values, color = 'blue', label='Current DB')    
        plt.fill_between(dfResult['time'].values, dfResult['cur_obj'].values, dfResult['cur_bd'].values, lw=0, color='lightsteelblue')
        
        axes2 = axes.twinx()
        axes2.set_ylabel('%gap')
        axes2.set_ylim(0, 100)
        line3, = axes2.plot(dfResult['time'].values, dfResult['gap'].values, color = 'red', label='Current Gap')
        axes.legend(handles=[line1, line2, line3], bbox_to_anchor=(0.5, 1.1), frameon=False, loc='upper center', ncol=3)
        
        plt.show()

def plotGranttChart():
    df = pd.DataFrame(x.keys(), columns=["Task", "l", "j"])
    df["value"] = model.getAttr("X", x).values()
    df["Start"] = model.getAttr("X", s).values()
    df["Finish"] = model.getAttr("X", f).values()
    df = df.drop(df[df.value < 0.9].index)
    df["diff"] = df["Finish"] - df["Start"]
    df["Resource"] = "Stage: " + df["l"].astype(str) + " Machine: " + df["j"].astype(str)
    # df["Task"] = df["Task"].astype(str)

    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Task", text="Task", color_continuous_scale = "viridis")
    fig.layout.xaxis.type = 'linear'
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(width=0.9)
    
    fig.data[0].x = df["diff"].tolist()

    fig.show()
    
## Data Preparation
NUMBER_OF_jOBS = 13
NUMBER_OF_STAGES = 2
NUMBER_OF_MACHINES_STAGE_1 = 1
NUMBER_OF_MACHINES_STAGE_2 = 3
NUMBER_OF_MACHINES = NUMBER_OF_MACHINES_STAGE_1 + NUMBER_OF_MACHINES_STAGE_2

I = [_ for _ in range(NUMBER_OF_jOBS)]
R = [_ for _ in range(NUMBER_OF_jOBS)]

K1 = [_ for _ in range(NUMBER_OF_MACHINES_STAGE_1)]
K2 = [_ for _ in range(NUMBER_OF_MACHINES_STAGE_2)]

L = {}  # Set of stages for each job
for job in range(NUMBER_OF_jOBS):
    # L.update({(job) : [l for l in range(random.choices([1,2], [0.1, 0.9], k=1)[0])]}) # for skipping
    L.update({(job): [l for l in range(NUMBER_OF_STAGES)]})

M = {}  # Set of eligible machines for each job at each stage
for i in I:
    for l in L.get((i)):
        if l == 0:
            M.update({(i, l): [j for j in K1]})
        else:
            M.update({(i, l): [j for j in K2]})

P = {}  # Processing times of job i on machine j in stage l
for i in I:
    for l in L.get((i)):
        for j in M.get((i, l)):
            P.update({(i, l, j): random.choices([10, 15], [0.8, 0.2], k=1)[0]})

D = {}  # Due dates of job i
for i in I:
    D.update({(i): random.choices([100, 150, 200], [0.7, 0.10, 0.20], k=1)[0]})

B = [(1, 2), (2, 3)]  # Set of incompatible pairs

BigM = NUMBER_OF_jOBS * 2 * (15 + 10)  # 2 stages, maximum processing time at a machine is 15 (s), maximum setup-time is 10

SDST = {}  # SDST values
for i1 in I:
    for i2 in I:
        if i1 != i2:
            for l in list(set(L.get((i1))) & set(L.get((i2)))):
                for j in list(set(M.get((i1, l))) & set(M.get((i2, l)))):
                    if (i1, i2) in B:
                        SDST.update({(i1, i2, l, j): BigM})
                    else:
                        SDST.update({(i1, i2, l, j): random.randint(5, 10)})

x_index = [(i, l, j) for (i, l, j) in P.keys()]
y_index = [(i, l, j, r) for (i, l, j) in P.keys() for r in R]
s_index = [(i, l, j) for (i, l, j) in P.keys()]
f_index = [(i, l, j) for (i, l, j) in P.keys()]
c_index = [i for i in I]
t_index = [i for i in I]

### Model
model = gp.Model()

x = model.addVars(x_index, vtype=GRB.BINARY, name="x")
y = model.addVars(y_index, vtype=GRB.BINARY, name="y")
s = model.addVars(s_index, vtype=GRB.CONTINUOUS, lb=0.0, name="s")
f = model.addVars(f_index, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
c = model.addVars(c_index, vtype=GRB.CONTINUOUS, lb=0.0, name="c")
t = model.addVars(t_index, vtype=GRB.CONTINUOUS, lb=0.0, name="t")
t_aux = model.addVars(t_index, vtype=GRB.CONTINUOUS, lb=0.0, name="t_aux")

model.setObjective(sum(t[i] for i in I), GRB.MINIMIZE)

## Constraints in thesis
for i in I:
    model.addConstr(t_aux[i] == c[i] - D.get((i)), name="Constr 1.1")
    model.addConstr(t[i] == gp.max_(t_aux[i], constant=0.0), name="Constr 1.2")

    for l in L.get((i)):
        model.addConstr(sum(x[i, l, j] for j in M.get((i, l))) == 1, name="Constr 2")

        for j in M.get((i, l)):
            model.addConstr(sum(y[i, l, j, r] for r in R) == x[i, l, j], name="Constr 3")

for l in range(NUMBER_OF_STAGES):
    if l == 0:
        for j in K1:
            for r in I:
                model.addConstr(sum(y[i, l, j, r] for i in I if (i, l, j) in P) <= 1, name="Constr 4")
    else:
        for j in K2:
            for r in I:
                model.addConstr(sum(y[i, l, j, r] for i in I if (i, l, j) in P) <= 1, name="Constr 4")

for i in I:
    for k in I:
        if k != i:
            if (i, k) in B:
                for l in list(set(L.get((i))) & set(L.get((k)))):
                    for j in list(set(M.get((i, l))) & set(M.get((i, l)))):
                        for r in range(NUMBER_OF_jOBS - 1):
                            model.addConstr(y[i, l, j, r] + y[k, l, j, r + 1] <= 1, name="Constr 5")

for i in I:
    for l1 in L.get((i)):
        for l2 in L.get((i)):
            if l2 < l1:
                for j1 in M.get((i, l1)):
                    for j2 in M.get((i, l2)):
                        # model.addConstr(s[i,l1,j1] >= f[i,l2,j2]*x[i,l1,j1], name="Constr 6")
                        model.addConstr(s[i, l1, j1] >= f[i, l2, j2] - BigM * (1 - x[i, l1, j1]), name="Constr 6")

for i in I:
    for k in I:
        if k != i:
            # if (i,k) not in B:
            for l in L.get((i)):
                for j in list(set(M.get((i, l))) & set(M.get((k, l)))):
                    for r in range(NUMBER_OF_jOBS - 1):
                        model.addConstr(
                            s[i, l, j] >= f[k, l, j] + SDST.get((k, i, l, j)) - BigM * (2 - y[k, l, j, r] - y[i, l, j, r + 1]),name="Constr 7")

for i in I:
    for l in L.get((i)):
        for j in M.get((i, l)):
            model.addConstr(f[i, l, j] == s[i, l, j] + P.get((i, l, j)) * x[i, l, j], name="Constr 8")

for i in I:
    model.addConstr(c[i] == gp.max_([f[i, l, j] for l in L.get((i)) for j in M.get((i, l))]), name="Constr 9")

## Additional Constraints
# Ensuring jobs must be assigned to the machine positions in sequential order.
for l in range(NUMBER_OF_STAGES):
    if l == 0:
        for j in K1:
            for r in range(NUMBER_OF_jOBS - 1):
                model.addConstr(sum(y[i, l, j, r] for i in I if (i, l, j) in P) >= sum(y[i, l, j, r + 1] for i in I if (i, l, j) in P), name="Constr 4")
    else:
        for j in K2:
            for r in range(NUMBER_OF_jOBS - 1):
                model.addConstr(sum(y[i, l, j, r] for i in I if (i, l, j) in P) >= sum(y[i, l, j, r + 1] for i in I if (i, l, j) in P), name="Constr 4")


model._obj = None
model._bd = None
model._gap = None
model._data = []
model._start = time.time()
model.Params.TimeLimit = 60*60
model.update()

model.optimize(callback=data_callback)
# model.computeIIS()
# model.write("model.ilp")

xResult = pd.DataFrame(x.keys(), columns=["i", "l", "j"])
xResult["value"] = model.getAttr("X", x).values()
yResult = pd.DataFrame(y.keys(), columns=["i", "l", "j", "r"])
yResult["value"] = model.getAttr("X", y).values()

plotGap(model._data)
plotGranttChart()
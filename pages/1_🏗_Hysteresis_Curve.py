# -*- coding: UTF-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

font_Times_New_Roman={"family":"Times New Roman",
                      # "style": "italic",
                      "weight":"heavy",
                      "size":16}
font_Song={"family":"SimSun",
           "style":"italic",
           "weight":"heavy",
           "size":15}
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rc('axes', unicode_minus=False)

class Hysteresis_loop:
    def __init__(self,disp,force,filter):
        self.disp=disp
        self.force=force
        self.Arg_dispzero = []
        self.length=disp.shape[0]

        self.filter=filter*0.9
        self.disp_positive_points=[]
        self.disp_negative_points=[]
        self.force_positive_points = []
        self.force_negative_points = []
        self.half_loop_list=[]
        self.points_half_cycle=[]
        self.disp_disassemble=[]
        self.force_disassemble = []
        self.Arg_dispzero_energy=[]
        self.S=[]

    def zero_division(self):
        self.Arg_dispzero.clear()
        for i in range(self.length - 1):
            if self.disp[i] * self.disp[i+1] <= 0 and self.disp[i+1] >= 0:
                self.Arg_dispzero.append(i + 1)
        return self.Arg_dispzero

    def zero_division_energy(self):
        self.Arg_dispzero_energy.clear()
        for i in range(self.length - 1):
            if self.disp[i] * self.disp[i+1] <= 0 :
                self.Arg_dispzero_energy.append(i + 1)
        return self.Arg_dispzero_energy

    def extreme_point(self):
        self.disp_positive_points.clear()
        self.force_positive_points.clear()
        self.disp_negative_points.clear()
        self.force_negative_points.clear()
        start_point=0
        for zero_point in self.zero_division():
            hysteresis_loop_disp = self.disp[start_point:zero_point]
            hysteresis_loop_force = self.force[start_point:zero_point]

            Arg_force_positive = np.argmax(hysteresis_loop_force)
            Arg_force_negative = np.argmin(hysteresis_loop_force)

            displacement_positive = hysteresis_loop_disp[Arg_force_positive]
            if abs(displacement_positive)<=self.filter:
                continue
            force_positive=hysteresis_loop_force[Arg_force_positive]
            self.disp_positive_points.append(displacement_positive)
            self.force_positive_points.append(force_positive)

            displacement_negative = hysteresis_loop_disp[Arg_force_negative]
            if abs(displacement_negative)<=self.filter:
                continue
            force_negative = hysteresis_loop_force[Arg_force_negative]
            self.disp_negative_points.append(displacement_negative)
            self.force_negative_points.append(force_negative)

            start_point = zero_point
        return self.disp_positive_points,self.force_positive_points,self.disp_negative_points,self.force_negative_points

    def permutation(self):
        A,B,C,D=self.extreme_point()
        positive_direction=np.hstack([np.array([A]).reshape(-1,1),np.array([B]).reshape(-1,1)])
        negative_direction = np.hstack([np.array([C]).reshape(-1,1),np.array([D]).reshape(-1,1)])
        return positive_direction,negative_direction

    def energy(self,end_point_initial,weight):
        self.zero_division_energy()
        self.half_loop_list.clear()
        self.points_half_cycle.clear()
        self.disp_disassemble.clear()
        self.force_disassemble.clear()
        self.S.clear()
        self.weight=weight

        self.end_point_initial=self.Arg_dispzero_energy[end_point_initial]
        E_start_point=0

        force_first_half_cycle = self.disp[E_start_point:self.end_point_initial]
        disp_first_half_cycle = self.force[E_start_point:self.end_point_initial]
        s_loop_first_half_cycle = abs(trapz(force_first_half_cycle, disp_first_half_cycle, dx=0.001))

        for E_zero_point in self.zero_division_energy():

            E_hysteresis_loop_force = self.force[E_start_point:E_zero_point]
            E_hysteresis_loop_disp = self.disp[E_start_point:E_zero_point]
            s_loop = abs(trapz(E_hysteresis_loop_force,E_hysteresis_loop_disp, dx=0.001))
            if s_loop < s_loop_first_half_cycle*self.weight:
                continue
            self.half_loop_list.append(s_loop)
            self.S.append(s_loop)

            E_start_point=E_zero_point

            self.points_half_cycle.append(E_zero_point)

        point_every_cycle1 = self.points_half_cycle[1::2][:]
        point_every_cycle2 = self.points_half_cycle[1::2][:]
        point_every_cycle1.insert(0, 0)

        for u in point_every_cycle2:
            v = point_every_cycle1.pop(0)
            self.disp_disassemble.append(self.disp[v:u])
            self.force_disassemble.append(self.force[v:u])

        return self.disp_disassemble,self.force_disassemble,self.S

st.set_page_config(page_title="Hysteresis Curve", page_icon="🏗")

st.markdown("# Hysteresis Curve")
st.sidebar.header("Hysteresis Curve")
uploaded_file = st.file_uploader("导入数据的第一列为位移，第二列为力，文件为.txt格式。")

first_cycle_disp_enter = st.number_input('请输入加载制度第一级对应的位移值',value=1)
# st.write('第一级循环的最大位移为:', first_cycle_disp_enter)

Initial1=st.slider('调整此数字直到图中第一圈循环显示',value=int(0),min_value=int(0),max_value=100,step=1)

Initial2 = st.slider('拖拽至对应的需要展示的循环数',value=0,min_value=0,max_value=100,step=1)
# st.write('Loop N.O.', Initial2)

Initial3 = st.slider('第一圈能量权重(默认0.5)',value=0.5,min_value=0.0,max_value=5.0,step=0.1)
# st.write('The energy weight is:', Initial3)

# 数据展示
col1, col2 ,col3,col4= st.columns(4)

if uploaded_file is not None:
    data = np.loadtxt(uploaded_file)
    dataframe=pd.DataFrame(data,columns=['Displacement','Force'])

    # p = figure(title='Hysteresis Curve',x_axis_label='Displacement',y_axis_label='Force')
    # p.line(dataframe.iloc[:,0], dataframe.iloc[:,1], legend_label='Hysteresis curve', line_width=2)
    # st.bokeh_chart(p, use_container_width=True)

    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=dataframe.iloc[:,0], y=dataframe.iloc[:,1],
                             mode='lines',
                             name='Hysteresis curve'))
    #=================================================提取骨架曲线
    HL = Hysteresis_loop(data[:,0], data[:,1], first_cycle_disp_enter)
    S_positive, S_negative = HL.permutation()
    skeleton_curve = np.vstack([S_negative[::-1], S_positive])

    with col1:
        st.header('滞回曲线')
        st.write(dataframe)

    skeleton_curve = pd.DataFrame(skeleton_curve, columns=['Displacement', 'Force'])
    with col2:
        st.header('骨架曲线')
        st.write(skeleton_curve)

    # fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,10),dpi=300)
    # fig.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    # ax.set_xlabel('位移', fontproperties=font_Song)
    # ax.set_ylabel('力', fontproperties=font_Song)
    # ax.tick_params(axis='x', which='major', direction='in', labelsize=14, length=4, width=1.5)  # 主刻度x：朝向、长短、大小
    # ax.tick_params(axis='x', which='minor', direction='in', color='#393e46', labelsize=14, length=2, width=1)  # 副刻度x
    # ax.tick_params(axis='y', which='major', direction='in', labelsize=14, length=4, width=1.5)  # 主刻度x
    # ax.tick_params(axis='y', which='minor', direction='in', color='#393e46', labelsize=14, length=2, width=1)  # 副刻度y
    # ax.grid(linestyle='--', color='#c9d6df')
    #
    # ax.plot(dataframe.iloc[:, 0], dataframe.iloc[:, 1], label='Hysteresis Curve',linewidth=5,color='#d3d4d8')
    # ax.plot(skeleton_curve.iloc[:,0],skeleton_curve.iloc[:,1], marker="s", markerfacecolor='red',markersize=12,
    #             markeredgewidth=1,markeredgecolor='#07689f',linestyle='-',label='Skeleton curve',linewidth=1)

    fig0.add_trace(go.Scatter(x=skeleton_curve.iloc[:,0], y=skeleton_curve.iloc[:,1],marker= {"size" :15,"color":"#c86b85",'symbol':'square'},
                             mode='markers',
                             name='skeleton curve'))
    # st.pyplot(fig)
    #=================================================提取能量

    disp_disassemble, force_disassemble, S = HL.energy(Initial1,Initial3)
    S=np.array(S)
    energy_out=np.vstack([np.arange(1,S.shape[0]+1),S]).T
    energy_out = pd.DataFrame(energy_out, columns=['Loop N.O.', 'Energy'])
    with col3:
        st.header('能量')
        st.write(energy_out)

    loop_out_disp=np.array([disp_disassemble[Initial2]])
    loop_out_force = np.array(force_disassemble[Initial2])

    loop_out = np.vstack([loop_out_disp,loop_out_force]).T
    loop_out = pd.DataFrame(loop_out, columns=['Displacement', 'Force'])
    with col4:
        st.header(f'第{Initial2}圈')
        st.write(loop_out)

    # ax.plot(disp_disassemble[Initial2], force_disassemble[Initial2],color='red',label='Loop')
    # ax.plot(disp_disassemble[0], force_disassemble[0], color="#14ffec", linewidth=2, linestyle='-', label='First loop')

    fig0.add_trace(go.Scatter(x=disp_disassemble[Initial2], y=force_disassemble[Initial2],
                             mode='lines+markers',
                             name=f'Loop {Initial2}',line_shape='spline'))
    fig0.add_trace(go.Scatter(x=disp_disassemble[0], y=force_disassemble[0],line=dict(dash='solid', width=3,color='#00fff5'),
                             mode='lines',
                             name='First loop'))

    # st.pyplot(fig)
    fig0.update_layout(autosize=False,width=1200,height=800,
                       title="滞回曲线展示",  # 主标题
                       xaxis_title="Displacement",  # 2个坐标轴的标题
                       yaxis_title="Force",
                       font=dict(
                           family="sans-serif",
                           size=20,
                           color="#7f7f7f"
                       ),
                       legend=dict(x=0, y=1,  # 图例的位置：将坐标轴看做是单位1
                                   traceorder="normal",
                                   font=dict(
                                       family="sans-serif",
                                       size=20,
                                       color="black"),
                                   bgcolor="LightSteelBlue",  # 背景颜色，边框颜色和宽度
                                   bordercolor="Black",
                                   borderwidth=2
                                   )
                       )
    st.plotly_chart(fig0)

else:
    st.error('导入数据有误！')


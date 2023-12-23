# -*- coding: UTF-8 -*-
import plotly.graph_objs as go
import streamlit as st
import numpy as np

st.set_page_config(page_title="宝儿，这是我画给你的圣诞树嘿嘿嘿！圣诞节快乐哦！", page_icon="❄")

st.markdown("# 宝儿画给宝儿的圣诞树🐷🐷🐷")
st.sidebar.header("宝儿，这是我画给你的圣诞树嘿嘿嘿！圣诞节快乐哦！")

b=np.linspace(0,2*np.pi,121)

a=np.linspace(0,3,121)

s,t=np.meshgrid(a,b)

x=s*np.cos(t)

y=s*np.sin(t)

z=-s**0.6*2+np.cos(t*12)*0.1

Data1=go.Surface(x=x,y=y,z=z,colorscale='greens',showscale=False,lighting=dict(ambient=0.7,diffuse=0.5,specular=0.05,roughness=0.5,fresnel=1.2))

Data2=go.Surface(x=x*0.8,y=y*0.8,z=z*0.8+1,colorscale='greens',showscale=False,lighting=dict(ambient=0.8,diffuse=0.8,specular=0.05,roughness=0.5,fresnel=2.2))

Data3=go.Surface(x=x*0.64,y=y*0.64,z=z*0.64+2,colorscale='greens',showscale=False,lighting=dict(ambient=1,diffuse=0.8,specular=0.05,roughness=0.5,fresnel=3.2))

Data4=go.Surface(x=np.cos(t)*0.8,y=np.sin(t)*0.8,z=s-5.4,surfacecolor=-s,showscale=False,colorscale='turbid')

lt=np.linspace(0,4*np.pi,6)

rt=np.linspace(0,1)

ka,kb=np.meshgrid(lt,rt)

x=(-abs(kb-1))/3

y=kb*np.sin(ka)

z=kb*np.cos(ka)+2

Data7=go.Surface(x=x,y=y,z=z,surfacecolor=kb,colorscale='ylorrd',showscale=False,lighting=dict(ambient=1,diffuse=0.8,specular=0.05,roughness=0.5,fresnel=2.2))

Data8=go.Surface(x=-x,y=y,z=z,surfacecolor=kb,colorscale='ylorrd',showscale=False,lighting=dict(ambient=1,diffuse=0.8,specular=0.05,roughness=0.5,fresnel=2.2))

Lay=go.Layout(barmode="group",width=960,height=600, scene=dict(bgcolor="white",xaxis=dict(backgroundcolor="lightblue"),
yaxis=dict(backgroundcolor="pink"),zaxis=dict(backgroundcolor="white"),camera=dict(projection=dict(type="orthographic"))))

Fig=go.Figure(data=[Data1,Data2,Data3,Data4,Data7,Data8],layout=Lay)
Fig.update_layout(autosize=False,width=800,height=1200)
st.plotly_chart(Fig)

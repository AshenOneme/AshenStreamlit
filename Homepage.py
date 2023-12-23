import streamlit as st

st.set_page_config(
    page_title="你好",
    page_icon="👋",
)

st.write("# 欢迎使用 Ashen's Streamlit! 👋")

st.sidebar.success("在上方选择一个演示。")

st.markdown(
    """
    Ashen's Streamlit 是一个专为结构抗震与振动控制领域构建的开源应用框架。      
    **👈 从侧边栏选择一个演示**，看看 Ashen's Streamlit 能做什么吧！
    ### 想了解更多吗？
    - 查看 [Ashen's Github](https://github.com/AshenOneme)
    ### 查看更复杂的示例
    - 滞回曲线 [处理程序exe](https://github.com/AshenOneme/Hysteresis-curve-processing-program)
"""
)
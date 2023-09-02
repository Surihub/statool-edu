import plotly.express as px

def 히스토그램(data):
    fig = px.histogram(data, nbins=10)
    fig.update_layout(xaxis_title='값', yaxis_title='빈도', title='히스토그램')
    fig.show()

def 상자그림(data):
    fig = px.box(data)
    fig.update_layout(yaxis_title='값', title='상자그림')
    fig.show()

def 막대그래프(x_labels, y_values):
    fig = px.bar(x=x_labels, y=y_values)
    fig.update_layout(xaxis_title='카테고리', yaxis_title='빈도', title='막대그래프')
    fig.show()

def 원그래프(labels, sizes):
    fig = px.pie(names=labels, values=sizes, hole=0.3)
    fig.update_layout(title='원그래프')
    fig.show()

# 히스토그램 함수 테스트
if __name__ == '__main__':
    import numpy as np
    data = np.random.randn(1000)  # 실제 데이터로 대체해주세요
    히스토그램(data)

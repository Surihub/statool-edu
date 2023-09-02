import seaborn as sns
import matplotlib.pyplot as plt

def 히스토그램(원자료, 열이름, 계급시작값, 계급의크기):
    최댓값 = max(원자료[열이름])  # 최댓값 계산
    bins = [계급시작값 + 계급의크기 * i for i in range(int((최댓값 - 계급시작값) / 계급의크기) + 2)]
    sns.histplot(data=원자료, x=열이름, bins=bins, kde=False, edgecolor='black')
    plt.title(f'{열이름}의 히스토그램')
    plt.xlabel(열이름)
    plt.ylabel('빈도')
    plt.show()

def 상자그림(원자료, 열이름):
    sns.boxplot(data = 원자료, x = 열이름)
    plt.ylabel('값')
    plt.title('상자그림')
    plt.show()

def 막대그래프(원자료, 열이름):
    order = 원자료[열이름].value_counts().sort_values(ascending = False).index.tolist()
    sns.countplot(data=원자료, x = 열이름, order = order)
    plt.ylabel('빈도')
    plt.title('막대그래프')
    plt.xticks(rotation=45)
    plt.show()

def 원그래프(원자료, 열이름):
    count = 원자료[열이름].value_counts().sort_values(ascending = False)
    plt.pie(count.tolist(), labels=count.index.tolist(), autopct='%1.1f%%', startangle=0)
    plt.title('원그래프')
    plt.show()



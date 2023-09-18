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


def 산점도(원자료, 가로열이름, 세로열이름, 구분=None):
    plt.figure(figsize=(8, 6))
    
    if 구분 is None:
        sns.scatterplot(data=원자료, x=가로열이름, y=세로열이름)
        plt.title(f'{가로열이름} vs. {세로열이름} 산점도')
    else:
        sns.scatterplot(data=원자료, x=가로열이름, y=세로열이름, hue=구분, palette='Set1')
        plt.title(f'{가로열이름} vs. {세로열이름} 산점도 (구분: {구분})')
    
    plt.xlabel(가로열이름)
    plt.ylabel(세로열이름)
    plt.legend(loc='best')
    plt.show()

# heatmap
# 전체 플롯




########################### 인공지능
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

dataset_dict = {
    '와인': datasets.load_wine(),
    '유방암': datasets.load_breast_cancer(),
    '당뇨병': datasets.load_diabetes(),
    '붓꽃': datasets.load_iris(),
    '손글씨': datasets.load_digits()
}

def 데이터불러오기(data):    
    if data in dataset_dict:
        dataset = dataset_dict[data]
    else:
        raise ValueError('지원하지 않는 데이터셋입니다.')
    
    # 데이터를 Pandas DataFrame으로 변환
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target  # 클래스 정보를 추가    
    return df



def 데이터분할하기(df):
    # 데이터를 특성(X)과 레이블(y)로 분할
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 훈련 세트와 테스트 세트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test



def 모델_훈련(X_train, y_train, 모델명):
    # 모델 선택
    모델 = 모델_선택(모델명)
    
    # 모델 훈련
    모델.fit(X_train, y_train)
    
    return 모델

def 모델_평가(X_test, y_test, model):
    # 범주형이면 혼동행렬
    if len(y_test.unique())<5 : 
        # 모델 예측
        y_pred = model.predict(X_test)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        # 정확도 계산
        cm = confusion_matrix(y_test, y_pred_rounded)

        실제_라벨 = [f'실제{i}' for i in range(len(cm))]
        예측_라벨 = [f'예측{i}' for i in range(len(cm))]
        confusion_df = pd.DataFrame(cm, columns=예측_라벨, index=실제_라벨)
        print(y_pred_rounded.tolist()[:10], '>>>> target을 예측한 값의 일부에요!')
        print(y_test.tolist()[:10], '>>>> target의 실제 값의 일부에요!')
        print('혼동행렬(confusion matrix)')
        display(confusion_df)
        print('\n')


        # 대각선 상의 셀 값을 합산하여 올바른 예측 수를 구함
        correct_predictions = np.sum(np.diag(confusion_df))

        # 전체 샘플 수를 구함 (혼동 행렬의 모든 값을 합산)
        total_samples = np.sum(confusion_df).sum()

        # 정확도 계산
        accuracy = correct_predictions / total_samples
        return np.round(accuracy, 4)
    else:
        # 모델 예측
        y_pred = model.predict(X_test)
        print(y_pred.tolist()[:10], '>>>> target을 예측한 값의 일부에요!')
        print(y_test.tolist()[:10], '>>>> target의 실제 값의 일부에요!')
 
        # 평균 제곱 오차(Mean Squared Error) 계산
        mse = mean_squared_error(y_test, y_pred)

        # 결정 계수(R-squared) 계산
        r_squared = r2_score(y_test, y_pred)

        print(f'평균 제곱 오차(MSE): {mse:.4f}')
        print(f'결정 계수(R-squared): {r_squared:.4f}')
        print('\n')

        return np.round(r_squared, 4)

def 모델_선택(모델명):
    모델_딕셔너리 = {
        '선형': LinearRegression(),
        '이차식': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        '랜덤 포레스트': RandomForestClassifier(),
        '서포트 벡터 머신': SVC(),
        '최근접 이웃': KNeighborsClassifier(),

        # 다른 모델을 추가할 수 있습니다.
    }
    
    if 모델명 in 모델_딕셔너리:
        return 모델_딕셔너리[모델명]
    else:
        raise ValueError('지원하지 않는 모델입니다.')
    


    
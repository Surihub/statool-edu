__version__ = '0.0.1'

# seaborn 설정하기
import seaborn as sns

sns.set_style("darkgrid")
sns.set_palette("colorblind")

# 플롯에 한글 설정하기
import koreanize_matplotlib



#####################################################
####################데이터 처리######################
#####################################################

def 데이터유형(df):
    return df
def 숫자로변환(df):
    # 모든 열을 순회하며 범주형 열만 선택
    categorical_columns = df.select_dtypes(include=['category', 'object']).columns

    # 범주형 열을 숫자로 매핑하기 위한 딕셔너리 초기화
    category_mapping = {}

    for col in categorical_columns:
        # 각 범주형 열의 고유한 값들을 확인하고 매핑 딕셔너리 생성
        category_mapping[col] = {category: index for index, category in enumerate(df[col].astype('category').cat.categories)}

        # 매핑 딕셔너리를 사용하여 범주형 데이터를 숫자로 변환
        df[col] = df[col].map(category_mapping[col])
    print(category_mapping)
    return df 

def 미리보기(원자료, n_row = None):
    n, m = 원자료.shape
    print(f"{n}행, {m}열로 이루어진 자료입니다.")

    if n_row is not None:
        display(원자료.head(n_row))
    else:
        display(원자료)


# 기초통계량
def 기초통계(원자료):
    df = 원자료.describe()
    df.index = ['개수', '평균', '표준편차', '최솟값', '제1사분위수', '중앙값', '제3사분위수','최댓값']
    display(df)

# 피벗, 역피벗 추가

# 분할표
# def 분할표(원자료, 행, 열):



# 역피벗
# def 원자료만들기():


### 기술통계
# def 대푯값():

# def 산포도():


#####################################################
#######################시각화########################
#####################################################
import seaborn as sns
import matplotlib.pyplot as plt

def 히스토그램(원자료, 열이름, 계급시작값, 계급의크기):
    최댓값 = max(원자료[열이름])  # 최댓값 계산
    bins = [계급시작값 + 계급의크기 * i for i in range(int((최댓값 - 계급시작값) / 계급의크기) + 2)]
    sns.histplot(data=원자료, x=열이름, bins=bins, kde=False)#, edgecolor='black')
    plt.title(f'{열이름}의 히스토그램')
    plt.xlabel(열이름)
    plt.ylabel('빈도')
    plt.show()

def 상자그림(원자료, 열이름, 구분=None):
    if 구분 is None:
        sns.boxplot(data = 원자료, y = 열이름)
        plt.title(f'{열이름} 상자그림')
    else:
        sns.boxplot(data=원자료, y=열이름, x=구분)
        plt.title(f'{구분}에 따른 {열이름} 상자그림')
    plt.show()

def 막대그래프(원자료, 열이름):
    order = 원자료[열이름].value_counts().sort_values(ascending=False).index.tolist()
    ax = sns.countplot(data=원자료, x=열이름, order=order)
    plt.ylabel('빈도')
    plt.title('막대그래프')
    plt.xticks(rotation=45)
    
    # 각 막대 위에 개수 표시
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

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


#####################################################
######################통계분석#######################
#####################################################


import numpy as np
import scipy.stats as stats
import seaborn as sns

def 모평균추정(data, column_name, confidence, population_stddev=None):
    """
    모표준편차를 입력하거나 입력하지 않는 경우에 따라 평균의 신뢰구간을 추정하는 함수

    Parameters:
    - data: 데이터프레임
    - column_name: 해당 열의 이름 또는 설명 (문자열)
    - confidence: 신뢰도 (0에서 1 사이의 실수)
    - population_stddev: 모표준편차 (모를 경우는 None)

    Returns:
    - None: 결과를 출력하고 반환하지 않습니다.
    """
    data_values = data[column_name].values
    n = len(data_values)
    mean = np.mean(data_values)

    if population_stddev is not None:
        # 모표준편차를 입력한 경우 (Z 분포 사용)
        margin_of_error = stats.norm.ppf(1 - (1 - confidence) / 2) * (population_stddev / np.sqrt(n))
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    else:
        # 모표준편차를 입력하지 않은 경우 (t 분포 사용)
        std_error = np.std(data_values, ddof=1) / np.sqrt(n)
        margin_of_error = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1) * std_error
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    # 결과 출력
    print("********************<신뢰구간 추정>************************")
    print(f"{column_name}에 대한 {confidence * 100}% 신뢰구간:: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}), 표본평균: {mean:.4f}")




def 모비율추정(data, column_name, target, confidence):
    """
    모비율의 신뢰구간을 추정하는 함수

    Parameters:
    - data: 데이터프레임
    - column_name: 해당 열의 이름 또는 설명 (문자열)
    - confidence: 신뢰도 (0에서 1 사이의 실수)

    Returns:
    - None: 결과를 출력하고 반환하지 않습니다.
    """
    data_values = data[column_name].values
    n = len(data_values)
    p_hat = (data[column_name] == target).mean()

    # Z 분포를 사용하여 모비율의 신뢰구간을 추정
    margin_of_error = stats.norm.ppf(1 - (1 - confidence) / 2) * np.sqrt((p_hat * (1 - p_hat)) / n)
    confidence_interval = (p_hat - margin_of_error, p_hat + margin_of_error)

    # 결과 출력
    print("********************<가설검정>************************")
    print(f"{column_name}에 대한 {confidence * 100}% 신뢰구간: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}), 표본비율: {p_hat:.4f}")




def 모평균검정(data, column_name, significance_level, reference_value, alternative='two-sided', population_stddev=None):
    """
    모표준편차를 입력하거나 입력하지 않는 경우에 따라 평균에 대한 가설 검정을 수행하는 함수

    Parameters:
    - data: 데이터프레임
    - column_name: 해당 열의 이름 또는 설명 (문자열)
    - significance_level: 유의수준 (0에서 1 사이의 실수)
    - reference_value: 기준값 (검정하고자 하는 평균 값)
    - population_stddev: 모표준편차 (모를 경우는 None)

    Returns:
    - result: 가설 검정 결과 (문자열)
    """
    data_values = data[column_name].values
    n = len(data_values)
    mean = np.mean(data_values)

    if population_stddev is not None:
        # 모표준편차를 입력한 경우 (Z 검정 사용)
        z_statistic = (mean - reference_value) / (population_stddev / np.sqrt(n))
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_statistic)))  # 양측 검정을 위해 2배를 곱함
    else:
        # 모표준편차를 입력하지 않은 경우 (t 검정 사용)
        std_error = np.std(data_values, ddof=1) / np.sqrt(n)
        t_statistic = (mean - reference_value) / std_error
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n - 1))  # 양측 검정을 위해 2배를 곱함
        elif alternative == 'less':
            p_value = stats.t.cdf(t_statistic, df=n - 1)
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_statistic, df=n - 1)

    # 귀무가설과 대립가설 출력
    if alternative == 'two-sided':
        null_hypothesis = f"귀무가설: {column_name}의 평균은 {reference_value}입니다."
        alternative_hypothesis = f"대립가설: {column_name}의 평균은 {reference_value}가 아닙니다."
    elif alternative == 'less':
        null_hypothesis = f"귀무가설: {column_name}의 평균은 {reference_value} 이상입니다."
        alternative_hypothesis = f"대립가설: {column_name}의 평균은 {reference_value} 미만입니다."
    elif alternative == 'greater':
        null_hypothesis = f"귀무가설: {column_name}의 평균은 {reference_value} 이하입니다."
        alternative_hypothesis = f"대립가설: {column_name}의 평균은 {reference_value} 초과입니다."

    # 가설 검정 결과 생성
    if p_value < significance_level:
        result = f"p값이 {p_value:.4f}로, 유의수준 {significance_level} 보다 작다. 따라서 귀무가설을 기각한다."
        # result = f"{column_name}의 평균은 {reference_value}와(과) 유의하게 다릅니다 (p-value={p_value:.4f})."
    else:
        result = f"p값이 {p_value:.4f}로, 유의수준 {significance_level} 보다 크다. 따라서 귀무가설을 기각할 수 없다."
        # result = f"{column_name}의 평균은 {reference_value}와(과) 유의하지 않게 다릅니다 (p-value={p_value:.4f})."

    # return result
    # p값이 ~로, 유의수준 ~보다크다. 따라서 귀무가설을 기각할 수 없다.
    print("********************************************")
    print(null_hypothesis)
    print(alternative_hypothesis)
    print(result)



def 모비율검정(data, column_name, target, significance_level, reference_value, alternative='two-sided'):
    """
    모비율에 대한 가설 검정을 수행하는 함수

    Parameters:
    - data: 데이터프레임
    - column_name: 해당 열의 이름 또는 설명 (문자열)
    - significance_level: 유의수준 (0에서 1 사이의 실수)
    - reference_value: 기준값 (검정하고자 하는 비율)
    - alternative: 검정 방법 ('two-sided', 'less', 'greater')

    Returns:
    - None: 결과를 출력하고 반환하지 않습니다.
    """
    data_values = data[column_name].values
    n = len(data_values)
    p_hat = (data[column_name] == target).mean()

    # Z 분포를 사용하여 모비율에 대한 가설 검정 수행
    z_statistic = (p_hat - reference_value) / np.sqrt((p_hat * (1 - p_hat)) / n)
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_statistic)))  # 양측 검정을 위해 2배를 곱함
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_statistic)
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_statistic)

    # 귀무가설과 대립가설 출력
    if alternative == 'two-sided':
        null_hypothesis = f"귀무가설: {column_name}의 비율은 {reference_value:.4f}입니다."
        alternative_hypothesis = f"대립가설: {column_name}의 비율은 {reference_value:.4f}가 아닙니다."
    elif alternative == 'less':
        null_hypothesis = f"귀무가설: {column_name}의 비율은 {reference_value:.4f} 이하입니다."
        alternative_hypothesis = f"대립가설: {column_name}의 비율은 {reference_value:.4f} 초과입니다."
    elif alternative == 'greater':
        null_hypothesis = f"귀무가설: {column_name}의 비율은 {reference_value:.4f} 이상입니다."
        alternative_hypothesis = f"대립가설: {column_name}의 비율은 {reference_value:.4f} 미만입니다."

    # 가설 검정 결과 생성
    if p_value < significance_level:
        result = f"p값이 {p_value:.4f}로, 유의수준 {significance_level} 보다 작다. 따라서 귀무가설을 기각한다."
    else:
        result = f"p값이 {p_value:.4f}로, 유의수준 {significance_level} 보다 크다. 따라서 귀무가설을 기각할 수 없다."

    # 결과 출력
    print("********************************************")
    print(null_hypothesis)
    print(alternative_hypothesis)
    print(result)



#####################################################
######################인공지능#######################
#####################################################
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
    '붓꽃': datasets.load_iris(),
    '와인': datasets.load_wine(),
    '당뇨병': datasets.load_diabetes(),
    # '유방암': datasets.load_breast_cancer(),
    # '손글씨': datasets.load_digits()
}

def 데이터불러오기(data):    
    if data in dataset_dict:
        dataset = dataset_dict[data]
        
        # 데이터를 Pandas DataFrame으로 변환
        df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        # display(df)
        df['target'] = dataset.target  # 클래스 정보를 추가    

    elif data == '펭귄':
        df = sns.load_dataset("penguins")
        df.columns = ['종류', '서식지', '부리 길이', '부리 깊이', '날개 길이', '몸무게', '성별']    
    elif data == '비디오게임':
        df = pd.read_csv("./Video_Games_Sales_kaggle.csv")
        df.columns = [
        '제목', '플랫폼', '출시년도', '장르', '배급사',
        '북미 판매량', '유럽 판매량', '일본 판매량', '기타 지역 판매량', '전체 글로벌 판매량',
        '평론가 평점', '평론가 리뷰 수', '사용자 평점', '사용자 리뷰 수', '개발사', '등급']
        df = df.dropna().reset_index(drop=True)
    else:
        raise ValueError('지원하지 않는 데이터셋입니다.')
    

    # 열 이름을 한글로 변경
    if data == '와인':
        df.columns =  ['물질명', '말산', '회분', '회분알칼리도', '마그네슘', '총_페놀', '플라보노이드', '비플라보노이드_페놀', '프로안토시아닌', '색상_강도', '색조', 'OD280_OD315_희석_와인', '프롤린', '타겟']
    elif data == '당뇨병':
        df.columns = ['연령', '성별', 'BMI', 's1','s2','s3', 's4', 's5', 's6', '타겟']
    elif data == '붓꽃':
        df.columns = ['꽃받침 길이', '꽃받침 너비', '꽃잎 길이', '꽃잎 너비', '품종']
    df = df.dropna()
    return df


def 데이터분할하기(df, target):
    # 데이터를 특성(X)과 레이블(y)로 분할
    X = df.drop(target, axis=1)
    y = df[target]
    
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
    if len(y_test.unique())<10 : 
        # 모델 예측
        y_pred = model.predict(X_test)
        y_pred_rounded = np.round(y_pred).astype(int)

        y_pred_label = list(set(y_pred_rounded))
        y_pred_label.sort()
        y_test_label = y_test.unique()
        y_test_label.sort()
        # 정확도 계산
        cm = confusion_matrix(y_test, y_pred_rounded)
        cm = cm[~np.all(cm == 0, axis=1)]
        # display(cm)

        실제_라벨 = [f'실제{i}' for i in y_test_label]
        예측_라벨 = [f'예측{i}' for i in y_pred_label]
        confusion_df = pd.DataFrame(cm, columns=예측_라벨, index=실제_라벨)
        print(y_pred_rounded.tolist()[:5], '>>>> target을 예측한 값의 일부에요!')
        print(y_test.tolist()[:5], '>>>> target의 실제 값의 일부에요!')
        print('혼동행렬(confusion matrix)')
        display(confusion_df)
        print('\n')
        # 정확도 계산
        accuracy = accuracy_score(y_test, y_pred_rounded)
        return np.round(accuracy, 4)
    else:
        # 모델 예측
        y_pred = model.predict(X_test)
        print(np.round(y_pred.tolist()[:10], 4), '>>>> target을 예측한 값의 일부에요!')
        print(np.round(y_test.tolist()[:10], 4), '>>>> target의 실제 값의 일부에요!')
 
        # 평균 제곱 오차(Mean Squared Error) 계산
        rmse = mean_squared_error(y_test, y_pred)**0.5
        print(f'RMSE(Root Mean Squared Error ): {rmse:.4f}')
        print('\n')

        return np.round(rmse, 4)

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
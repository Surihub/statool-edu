import numpy as np
import scipy.stats as stats
import seaborn as sns

# def 추정(원자료, 열이름, 신뢰도):
#     """
#     통계 분석을 위한 추정 검정을 수행하는 함수

#     Parameters:
#     - data: 원자료 (numpy 배열 또는 리스트)
#     - column_name: 해당 열의 이름 또는 설명 (문자열)
#     - confidence: 신뢰도 (0에서 1 사이의 실수, 기본값은 0.95)

#     Returns:
#     - None: 결과를 출력하고 반환하지 않습니다.
#     """
#     # 데이터의 크기 (샘플 크기)
#     n = len(원자료)

#     # 데이터의 평균과 표준 오차 계산
#     mean = np.mean(원자료[열이름])
#     std_error = np.std(원자료[열이름], ddof=1) / np.sqrt(n)


#     # 신뢰구간 계산
#     margin_of_error = stats.norm.ppf(1 - (1 - 신뢰도) / 2) * std_error
#     confidence_interval = (mean - margin_of_error, mean + margin_of_error)
#     # 결과 출력
#     print(f"평균 추정값: {mean:.4f}")
#     print(f"{열이름}에 대한 {신뢰도 * 100}% 신뢰구간: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
#     # print(f"신뢰구간의 길이": {margin_of_error})


# def 검정(원자료, 열이름, 유의수준, 기준값):
#     """
#     모평균 검정을 수행하는 함수

#     Parameters:
#     - data: 원자료 (numpy 배열 또는 리스트)
#     - column_name: 해당 열의 이름 또는 설명 (문자열)
#     - significance_level: 유의수준 (0에서 1 사이의 실수)
#     - reference_value: 기준값 (검정하고자 하는 평균 값)

#     Returns:
#     - result: 가설 검정 결과 (문자열)
#     """
#     # 데이터의 크기 (샘플 크기)
#     n = len(원자료)

#     # 데이터의 평균과 표준 오차 계산
#     mean = np.mean(원자료[열이름])
#     std_error = np.std(원자료[열이름], ddof=1) / np.sqrt(n)

#     # 검정 통계량과 p-value 계산
#     t_statistic = (mean - reference_value) / (std_error)
#     p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n - 1))  # 양측 검정을 위해 2배를 곱함

#     # 가설 검정
#     if p_value < significance_level:
#         result = f"{column_name}의 평균은 {reference_value}와(과) 유의하게 다릅니다 (p-value={p_value:.4f})."
#     else:
#         result = f"{column_name}의 평균은 {reference_value}와(과) 유의하지 않게 다릅니다 (p-value={p_value:.4f})."

#     return result







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


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


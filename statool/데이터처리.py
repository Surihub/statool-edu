

def 미리보기(원자료, n_row = None):
    n, m = 원자료.shape
    print(f"{n}행, {m}열로 이루어진 자료입니다.")

    if n_row is not None:
        display(원자료.head(n_row))
    else:
        display(원자료)

# 피벗, 역피벗 추가

# 분할표
# def 분할표(원자료, 행, 열):



# 역피벗
# def 원자료만들기():


### 기술통계
# def 대푯값():

# def 산포도():


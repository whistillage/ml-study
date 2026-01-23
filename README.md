# 📘 Machine Learning Study Archive

**머신러닝 이론과 실습을 체계적으로 학습**하기 위한 개인 공부 기록입니다.

- 📚 참고 교재: *파이썬 머신러닝 완벽 가이드 개정 2판*

---

## 🎯 Study Goals

- 머신러닝의 핵심 개념 이해
- 딥러닝/AI 응용 학습을 위한 기초 구축

---

## 🛠️ Environment

- Language: Python
- Libraries: NumPy, Pandas, Matplotlib, Scikit-learn 등
- Tooling: Jupyter Notebook / VS Code

---

## 📂 Repository Structure

---

## ✍️ 단원별 정리

### 01. 파이썬 기반의 머신러닝과 생태계 이해
- 머신러닝이란?
    - 애플레이케이션을 수정하지 않고도 데이터를 기반으로 패턴을 학습하고 결과를 예측하는 알고리즘 기법
- 머신러닝의 분류
    - 지도학습
        - 분류(Classification)
        - 회귀(Regression)
        - 추천 시스템
        - 시각/음성 감지/인지
        - 텍스트 분석, NLP
    - 비지도학습
        - 클러스터링
        - 차원 축소
        - 강화학습
- 머신러닝의 특성
    - 데이터에 의존적이다
- 언어
    - Python
        - 객체 지향형, 함수형 프로그래밍 모두를 포괄하는 아키텍처
        - 속도는 느리지만 뛰어난 확장성, 유연성, 호환성
        - 많은 라이브러리
        - 딥러닝 프레임워크(TensorFlow, Keras, PyTorch) 지원
    - R
        - 통계 전용 프로그램 언어
        - 많은 통계 패키지를 지원
- Numpy
    - Numerical Python
    - 빠른 배열 기반 연산을 지원하는 패키지
    - 숫자(int, unsigned int, float), 문자열, bool 모두 가능
    - 기초 사용법은 [notebook](01-basics/01_numpy.ipynb) 참고
- Pandas
    - 2차원 데이터를 효율적으로 가공/처리하는 라이브러리
        - numpy보다 유연함
    - 많은 부분이 numpy를 기반으로 작성됨
    - csv 등의 파일을 쉽게 **DataFrame**으로 변환해줌
        - DataFrame: 2차원 데이터를 담는 데이터 구조체
            - 여러 Series로 구성됨
            - Index + 여러 개의 Column
        - Series
            - Index + 1개의 Column
    - Index
        - DataFrame, Series의 레코드를 고유하게 식별하는 객체
        - Primary Key와 비슷
    - 데이터 Selection 및 Filtering
        - 행 추출에서 슬라이싱은 지양
            - 불리언 인덱싱 or 칼럼명 지정 권장
        - iloc
            - 위치(정수형) 기반
            - 불리언 인덱싱 불가
        - loc
            - 레이블 기반
            - 불리언 인덱싱 가능
    - 기초 사용법은 [notebook](01-basics/01_pandas.ipynb) 참고

### 02. 사이킷런으로 시작하는 머신러닝
- 사이킷런(scikit-learn)이란?
    - 파이썬 머신러닝 라이브러리
    - 머신러닝 모델을 구축하는 주요 프로세스를 지원하는 편리하고 다양한 모듈을 지원
    - 쉽고 직관적인 API 프레임워크 지원
- 사이킷런의 주요 모듈
    - 예제 데이터
    - 피처 처리
    - 피처 처리 & 차원 축소
    - 데이터 분리, 검증 & 파라미터 튜닝
    - 평가
    - ML 알고리즘
    - 유틸리티
- 예제 데이터 셋
    - 종류
        - 라이브러리 내장 데이터 셋
            - 분류용: datasests.load_iris() 등
            - 회귀용: datasets.load_boston() 등
        - fetch
            - 용량이 커서 인터넷에서 다운받아 사용
            - datasets.fetch_covtype() 등
        - 표본 데이터 랜덤 생성기
            - 분류용: datasets.make_classifications()
            - 클러스터링용: datasets.make_blobs()
    - 구성
        - data: feature 데이터 셋 (ndarray)
        - target (ndarray)
            - 분류: 레이블 값
            - 회귀: 숫자 결괏값 데이터 셋
        - target_names: 레이블 이름 (ndarray or list)
        - feature_names: feature 이름 (ndarray or list)
        - DESCR: 데이터 셋과 각 feature에 대한 설명 (string)
- 기타 개념
    - 지도 학습
        - 레이블이 주어진 데이터를 학습하여 미지의 데이터에 대한 레이블을 예측
        - 종류
            - 분류(Classification)
                - 분류 모델의 기초 예제는 [notebook](02-sklearn/02_ml_workflow.ipynb) 참고
            - 회귀(Regression)
        - Estimator
            - 지도 학습의 모든 알고리즘을 구현한 클래스
                - fit(): 학습
                - predict(): 예측
            - 종류
                - 분류: Classifier
                - 회귀: Regressor
    - 비지도학습
        - fit(): 입력 데이터의 형태에 맞춰 데이터를 변환하기 위한 사전 구조 구현
        - transform(): 입력 데이터의 차원 변환, 클러스터링, 피처 추출 등
    - 과적합(Overfitting)
        - 모델이 학습 데이터에만 과도하게 최적화되어 실제 예측을 다른 데이터로 수행할 경우에는 예측 성능이 과도하게 떨어지는 것
---
- 아래 전체 과정의 예제는 [notebook](02-sklearn/02_exercise_titanic.ipynb) 참고
1. 데이터 전처리
    - 모듈 사용법은 [notebook](02-sklearn/02_preprocessing.ipynb) 참고
    - 기본 규칙
        - NaN 값 허용 X
            - 다른 값으로 변환해야 함
        - 문자열 허용 X
            - 숫자형으로 변환해야 함
            - 종류
                - 카테고리형
                    - 코드 값으로 변환
                - 텍스트형
                    - 피처 벡터화 등의 기법으로 벡터화 또는 삭제
    - 데이터 인코딩
        1. 레이블 인코딩
            - 카테고리 피처를 숫자형으로 변환
            - 값의 크고 작음은 반영되지 않아야 함
                - 선형 회귀에서 사용되서는 안됨
                - 트리 계열 알고리즘은 사용되어도 됨
            - sklearn.preprocessing의 LabelEncoder 클래스 이용
        2. 원-핫 인코딩(One-Hot encoding)
            - 피처 값들을 열로 변환한 뒤, 해당하는 피처 값만 1, 나머지는 전부 0으로 표시
            - sklearn.preprocessing의 OneHotEncoder 클래스 이용
            - pandas의 get_dummies() 메소드 이용
                - 숫자형으로 변환하지 않음
    - 피처 스케일링
        - 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 것
        - 표준화(Standardizaiton)
            - 피처 데이터를 **N(0, 1)**로 변환하는 것
            - 각 데이터 x에 **(x-μ)/σ**를 적용
            - sklearn.preprocessing의 StandardScaler 클래스 이용
            - Support Vector Machine, Linear Regression, Logistic Regression에서 표준화된 데이터를 전제
        - 정규화(Normalization)
            - 개별 데이터의 크기를 모두 똑같은 단위로 변경하는 것
            - ex. sklearn.preprocessing의 MinMaxScaler 클래스 이용
                - 0 ~ 1의 값으로 변환
                - **x_i - min(x) / max(x) - min(x)**를 적용      
        - 벡터 정규화
            - Normalizer 모듈
            - **x_i / √(x_i^2 + y_i^2 + z_i^2)**를 적용
        - 유의 사항
            1. 가능하다면 전체 데이터의 스케일링을 적용한 뒤 학습과 테스트 데이터로 분리
            2. 1이 어렵다면, 학습 데이터로 fit()된 Scaler로 테스트 데이터를 transform()
                - 테스트 데이터에 새롭게 스케일링을 적용해서는 안됨
                - 테스트 데이터에 fit_transform() 적용해서는 안됨
2. 데이터 셋 분리
    - 모듈 사용법은 [notebook](02-sklearn/02_model_selection.ipynb) 참고
3. 모델 학습
4. 테스트 데이터 예측
    - 교차 검증
        - 모듈 사용법은 [notebook](02-sklearn/02_model_selection.ipynb) 참고
        - 교차 검증이란?
            데이터 편중을 막기 위해, 별도로 구성된 여러 학습 데이터 셋과 검증 데이터 셋에서 학습과 평가를 수행하는 것
        - 필요한 이유
            - 고정된 데이터 셋으로 학습을 하다보면 해당 테스트 데이터에 편향되게 모델을 유도하는 편향이 발생함
            - 다른 테스트 데이터 셋에는 성능이 저하됨
            - 각 셋에서 수행한 평가 결과에 따라 하이퍼파라미터 튜닝 등 모델 최적화를 쉽게 할 수 있음
        - 학습 데이터 셋을 학습 데이터 셋 + **검증 데이터 셋**으로 분리
        - K-폴드 교차 검증
            - KFold            
                - K개의 폴드 셋을 만들어서 각각 학습과 검증을 수행함
                - 데이터를 K등분하여 K개의 검증 데이터 셋을 정함
                - 최종 검증 평가 = K개의 검증 평가의 평균
            - Stratified KFold
                - 불균형하게 분포되어 있는 레이블 데이터 집합을 위한 K 폴드 방식
                    - ex. 매우 적은 비율로 있는 스팸 메일
                - 원본 데이터의 **레이블 분포**를 학습/테스트 셋에서도 보존함
            - cross_val_score
                - Stratified KFold로 추출한 인덱스를 기반으로 작성한 로직을 한꺼번에 수행
                    1. 폴드 셋 설정
                    2. for문에서 학습 및 테스트 데이터의 인덱스 추출
                    3. 학습 및 예측 수행 및 예측 성능 반환
                - 반환값: 평가 지표 배열
            - cross_validate
                - 여러 개의 평가 지표를 반환할 수 있음
        - GridSearchCV
            - 교차 검증 + 하이퍼파라미터 튜닝
            - 하이퍼파라미터 조합을 순차적으로 적용
                - 수행시간이 상대적으로 오래 걸림

### 03. 평가

### 04. 분류

### 05. 회귀

### 06. 차원 축소

### 07. 군집화

### 08. 텍스트 분석

### 09. 추천 시스템

### 10. 시각화

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
        - 분류
        - 회귀
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
    - 기초 사용법은 [notebook]()

### 02. 사이킷런으로 시작하는 머신러닝

### 03. 평가

### 04. 분류

### 05. 회귀

### 06. 차원 축소

### 07. 군집화

### 08. 텍스트 분석

### 09. 추천 시스템

### 10. 시각화

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
plt.rc('font', family='Malgun Gothic')


def top10(df):
    for year in df['year'].unique()[2:6]:
        print(f"\n{year}년도 상위 10명 선수")

        # 해당 연도의 데이터만 선택
        year_df = df[df['year'] == year]

        # 안타 상위 10명
        top_hit = year_df.sort_values(by='H', ascending=False).head(10)
        print("안타 상위 10명:\n", top_hit[['batter_name', 'H']])

        # 타율 상위 10명
        top_avg = year_df.sort_values(by='avg', ascending=False).head(10)
        print("\n타율 상위 10명:\n", top_avg[['batter_name', 'avg']])

        # 홈런 상위 10명
        top_hr = year_df.sort_values(by='HR', ascending=False).head(10)
        print("\n홈런 상위 10명:\n", top_hr[['batter_name', 'HR']])

        # 출루율 상위 10명
        top_obp = year_df.sort_values(by='OBP', ascending=False).head(10)
        print("\n출루율 상위 10명:\n", top_obp[['batter_name', 'OBP']])

def top_war(df):
    col_war = ['batter_name','cp','war']
    df_war =df[df['year'] == 2018]
    df_war = df[col_war]

    # 2018년 포지션(cp) 별로 war이 가장 높은 선수 출력
    top_players = df_war.loc[df_war.groupby('cp')['war'].idxmax()]

    print("2018년 포지션별 war이 가장 높은 선수들:\n", top_players[['cp', 'batter_name', 'war']])

def sal_corr(df):
    # 연도별 상관관계 히트맵 생성
    corr_matrix = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

    correlations = df.corr()['salary'].abs().sort_values(ascending=False)

    # 상관관계가 가장 높은 지표 출력
    top_correlation_indicator = correlations.index[1]  # 첫 번째는 'salary' 자신이므로 두 번째를 선택
    top_correlation_value = correlations.iloc[1]

    print(f"\nsalary와 상관관계가 가장 높은 지표:")
    print(f"{top_correlation_indicator}: {top_correlation_value:.3f}")

if __name__=='__main__':
    #DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    df = pd.read_csv('2019_kbo_for_kaggle_v2.csv', encoding='utf-8')
 
    top10(df)
    top_war(df)
    sal_corr(df)

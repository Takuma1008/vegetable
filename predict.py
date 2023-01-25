import statsmodels.api as sm
import numpy as np
import pandas as pd

def main():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    data = preprocess(train_df, test_df)

    kind_price_dict = {}
    for kind in test_df['kind'].unique():
        pred_price = predict(data, kind)
        kind_price_dict |= {kind: pred_price}

    test_df['mode_price'] = test_df['kind'].map(kind_price_dict)
    submit = test_df[['kind', 'date', 'mode_price']]
    submit.to_csv('submission.csv', index=False)

def preprocess(train_df, test_df):
    train_df['year'] = train_df['date']//10000
    test_df['year'] = test_df['date']//10000
    train_df['month'] = train_df['date'].apply(lambda x: int(str(x)[4:6]))
    test_df['month'] = test_df['date'].apply(lambda x: int(str(x)[4:6]))

    # 訓練データにしかない野菜は除く
    kinds = test_df['kind'].unique()
    train_df = train_df[train_df['kind'].isin(kinds)]

    all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    all_df.drop('weekno', axis=1, inplace=True)

    # 2005年は中途半端な月から始まっているので除く
    all_df = all_df.query('20060101 <= date').reset_index(drop=True)

    # 月毎に集計する
    vis_df = pd.pivot_table(all_df, index=['year', 'month'], columns='kind', values='mode_price').reset_index()
    vis_df = vis_df.fillna(0)

    # 2022年11月のデータ（予測対象月）のデータをnullにして入れておく
    november_2022 = vis_df.copy()
    november_2022.loc[:, november_2022.columns[2:]] = np.NaN
    november_2022 = pd.DataFrame(november_2022.iloc[-1]).T
    november_2022['month'] = 11
    vis_df = pd.concat([vis_df, november_2022]).reset_index(drop=True)

    # indexをdatetime型に変換
    vis_df.index = [str(int(year)) + str(month) + '1' for year, month in zip(vis_df['year'], vis_df['month'])]
    vis_df.index = pd.to_datetime(vis_df.index, format='%Y%m%d')

    return vis_df

def predict(data, kind):

    # train_data = data[data.index < "2022-10"]
    train_data = data[data.index < "2021-10"]
    
    # テストデータはテスト期間以前の日付も含まなければいけない
    # test_data = data[data.index >= "2022-01"]
    test_data = data[data.index >= "2021-01"]

    # 総当たりで、AICが最小となるSARIMAの次数を探す
    max_p = 3
    max_q = 3
    max_d = 2
    max_sp = 1
    max_sq = 1
    max_sd = 1

    pattern = max_p*(max_d + 1)*(max_q + 1)*(max_sp + 1)*(max_sq + 1)*(max_sd + 1)

    modelSelection = pd.DataFrame(index=range(pattern), columns=["model", "aic"])

    # 自動SARIMA選択
    num = 0

    for p in range(1, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                for sp in range(0, max_sp + 1):
                    for sd in range(0, max_sd + 1):
                        for sq in range(0, max_sq + 1):
                            sarima = sm.tsa.SARIMAX(
                                train_data[kind], order=(p,d,q), 
                                seasonal_order=(sp,sd,sq,12), 
                                enforce_stationarity = False, 
                                enforce_invertibility = False
                            ).fit(disp=False)
                            modelSelection.loc[num]["model"] = str(p) + "_" + str(d) + "_"+ str(q) + "," + str(sp) + "_" + str(sd) + "_" + str(sq)
                            modelSelection.loc[num]["aic"] = sarima.aic
                            num = num + 1

    order, seasonal_order = [x.split('_') for x in modelSelection.sort_values(by='aic').iloc[0]['model'].split(',')]
    order = [int(x) for x in order]
    seasonal_order = [int(x) for x in seasonal_order] + [12]

    SARIMA = sm.tsa.SARIMAX(train_data[kind], order=order, seasonal_order=seasonal_order).fit()
    # 予測
    # pred = SARIMA.predict('2022-01-01', '2022-11-01')
    pred = SARIMA.predict('2021-01-01', '2021-11-01')
    
    print(f"実際：{data[kind]['2021-11-01']}, 予測：{pred.iloc[-1]}")

    return pred.iloc[-1]

if __name__ == '__main__':
    main()
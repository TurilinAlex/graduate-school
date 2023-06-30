import multiprocessing
from datetime import datetime, timedelta

import pandas as pd
from requests import get

list_ticker = [
    'AUD/CAD', 'AUD/CHF', 'AUD/NZD', 'AUD/USD', 'EUR/AUD',
    'EUR/CHF', 'EUR/GBP', 'EUR/USD', 'GBP/AUD', 'EUR/CAD',
    'GBP/CHF', 'GBP/NZD', 'NZD/USD', 'USD/CAD', 'USD/CHF'
]


def history_price(order):
    try:
        dt = datetime.now()
        t_start = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0).timestamp()
        t_end = datetime(1990, dt.month, dt.day, dt.hour, dt.minute, 0).timestamp()
        url = f'https://sec7.intrade2.bar/fxhis/symbol={order}&resolution=1&from={int(t_end)}&to={int(t_start)}'
        print(url)
        r = get(url, timeout=1200)
        print(len(r.json()['candles']))
        ask_bid_price = pd.DataFrame(r.json()['candles'])

        candles = pd.DataFrame()
        candles['Date'] = pd.to_datetime(ask_bid_price[0], unit='s')
        candles['Open'] = round((ask_bid_price[1] + ask_bid_price[5]) / 2, 5)
        candles['High'] = round((ask_bid_price[3] + ask_bid_price[7]) / 2, 5)
        candles['Low'] = round((ask_bid_price[4] + ask_bid_price[8]) / 2, 5)
        candles['Close'] = round((ask_bid_price[2] + ask_bid_price[6]) / 2, 5)
        candles['Volume'] = ask_bid_price[9]
        stock = r.json()['instrument_id'].replace('/', '_')
        candles.to_csv(fr"/home/turilin/Documents/GitHub/graduate-school/data/{stock}.csv", index=False)

    except Exception as ex:
        print(ex)
        print(order)


if __name__ == '__main__':
    print('*' * 10 + 'START_HISTORY' + '*' * 10)
    print(datetime.now())

    # multiprocessing.set_start_method('spawn')
    # p = multiprocessing.Pool(processes=len(list_ticker))
    # p.map(history_price, list_ticker)
    # p.close()
    history_price(list_ticker[11])
    print(datetime.now())

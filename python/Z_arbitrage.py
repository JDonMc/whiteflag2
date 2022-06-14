import bitmex
import requests
import json
import datetime
from coinbase.wallet.client import Client
from time import sleep

bitmex_api_key = '8B2bo1xsLM3UGm0qeBX4lnSf' 
bitmex_api_secret = 'jzxim_5fmd7kBiIzRSloC2366-TMd_u738peJYe8FsRN8XN_'

coinbase_API_key = '01d671112a6cfd7a5ccb6e572e21095c'
coinbase_API_secret = 'Aq0AwNfvZILETL0II65R5gTsPWmN7PN0FQoAvcY/qzGE5XUR+ZNOgRVyDL92oRr6jvPjxvQGA+UJOnHKUN4y0g=='

client = bitmex.bitmex(test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret)

clientb = Client(coinbase_API_key, coinbase_API_secret)

while True:
    positions = client.Quote.Quote_get(filter=json.dumps({"symbol": 'XBTUSD'}), reverse=True).result()[0][0]
    print(positions)
    bitmex_btc = {}
    
    bitmex_btc["markPrice"] = positions["askPrice"]
    print('BitMex: ',bitmex_btc['markPrice'])
    
    coinbase_btc = clientb.get_spot_price(currency_pair= 'BTC-USD')
    print('Coinbase: ',coinbase_btc['amount'])
    
    percent = float(((float(coinbase_btc['amount']) - bitmex_btc['markPrice']) * 100) / bitmex_btc['markPrice']) 
    
    sleep (1)
    
    if percent < 1.5:
        print ('No arbitrage possibility')
        continue
    
    else:
        if percent == 1.5:
            print ('ARBITRAGE TIME')
            break
    sleep(1)
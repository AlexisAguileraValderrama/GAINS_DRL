import gym
from gym import spaces

from datetime import datetime
from random import randrange
from datetime import timedelta

import numpy as np
import pandas as pd

import random

# import pytz module for working with time zone
import pytz

import MetaTrader5 as mt5
from contract import Contract

import torch

#La clase FakeNewsEnv es una clase que implementa un gym environment
#Su objetivo es definir 

class ForexTrainEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, account=0, password = '',money = 300, quote = 'EURUSD', ticks_before = 5, brain_name = ''):
    super(ForexTrainEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:

    # now connect to another trading account specifying the password
    account=account
    authorized=mt5.login(account, password=password)
    if authorized:
        # display trading account data 'as is'
        print(mt5.account_info())
        # display trading account data in the form of a list
        print("Show account_info()._asdict():")
        account_info_dict = mt5.account_info()._asdict()
        for prop in account_info_dict:
            print("  {}={}".format(prop, account_info_dict[prop]))
    else:
        print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))

    print(torch.version.cuda)
 
    self.brain_name = brain_name

    self.total_min = 350
    self.steps_limit = 330

    self.ticks_before = ticks_before

    self.money_gained = 0
    self.money_lost = 0

    self.wallet = 300 # round(random.uniform(100.00, 700.00), 2) ##Ej 300
    self.quote = quote ##Ej EURUSD

    self.wallet_to_evaluate = self.wallet
    self.time_to_evaluate = 10 #Minutes (ticks)

    

    self.loss_limit_per = 1
    self.loss_limit = self.wallet*self.loss_limit_per/100 ##Ir actualizando

    self.profit = 0
    self.last_profit_per = 0

    self.last_gained = 0

    self.cancelled_contracts = 0
    self.con_canelled_contracts = 0
    self.abort = False

    print("Connecting with MetaTrader")

    print("MetaTrader5 package author: ",mt5.__author__)
    print("MetaTrader5 package version: ",mt5.__version__)
    
    # establish connection to the MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    
    # attempt to enable the display of the EURJPY symbol in MarketWatch
    selected=mt5.symbol_select(quote,True)
    if not selected:
        print(f'Faildes getting {quote}')
        mt5.shutdown()
        quit()
    
    # display EURJPY symbol properties
    symbol_info=mt5.symbol_info(quote)
    self.currency_bid = symbol_info.bid
    self.pip_digits = symbol_info.digits
    self.contract_size = symbol_info.margin_hedged

    self.pip_value = (1/10**(self.pip_digits-1))*self.contract_size/self.currency_bid

    self.limit_pip_value_per = 0.02

    self.max_lot_cap = self.wallet*self.limit_pip_value_per/self.pip_value ##Esto debe irse actualizando

    self.current_contract = None
    self.current_time_lapse_data = None
    self.current_minute = 0

    self.timezone = pytz.timezone("Etc/UTC")

    self.log = f'-------------------------------------------------- \n' + \
          f'Money: {self.wallet} \n' + \
          f'Ventana de tiempo: {self.total_min} \n' + \
          f'Ticks before: {self.ticks_before} \n' + \
          f'Loss limit: {self.loss_limit} \n' + \
          f'Quote: {self.quote} \n' + \
          f'pip value: {self.pip_value} \n' + \
          f'Max lot capacity: {self.max_lot_cap}' 


    print(self.log)

    self.reward_total = 0

    self.action_space = spaces.Box(low=-0.0, high=1.0,
                                    shape=(3,), dtype=np.float32)


    self.observation_space = spaces.Box(low=-5.0, high=5.0,
                                        shape=(3 + (self.ticks_before + 1)*2 + 5,), dtype=np.float32)

  def BuildObservation(self):
    #Grab the first sentence

    digits_from_wallet = []
    for i in list(range(3))[::-1]:
      digits_from_wallet.append((self.wallet // 10**i % 10)/10)

    window = self.current_time_lapse_data[self.current_minute - self.ticks_before:self.current_minute+1]
    window_values = window[["open","close"]].astype(np.float32).values
  
    window_values_flatten = []
    for l in window_values:
        window_values_flatten.extend(l)
    
    contract_data = []

    if self.current_contract is None:
      contract_data = [0,0,0,0,0]
    else:
      contract_data = [self.current_contract.type,
                       self.current_contract.vol_start,
                       self.current_contract.lot_size,
                       self.close_values[self.current_minute],
                       self.profit/self.wallet*100, ##Profit loss
                       ]
      
    obs = np.concatenate((digits_from_wallet,
                              window_values_flatten,
                              contract_data),
                              axis=None).astype(np.float32)

    return obs


  def step(self, action):

    # 0 - Action -> 0 - nada / 1 - Abrir compra / 2 - Abrir venta / 3 - Cerrar contrato
    # 1 - Buy
    # 2 - Sell

    isFinished = False
    reward = 0

    discrete_action = int(4*action[0])

    current_close = self.close_values[self.current_minute]

    if self.current_contract is not None:
      self.profit = self.current_contract.type * \
                         self.current_contract.lot_size * \
                         self.pip_value * \
                         (current_close-self.current_contract.vol_start) / \
                          (1/10**(self.pip_digits-1))
      self.profit_per = self.profit/self.wallet*100*4

      reward += self.profit_per - self.last_profit_per
      self.last_profit_per = self.profit

    if discrete_action == 0:
      if self.current_contract is None:
        reward += -0.005
      else:
        reward += -0.001
    
    if discrete_action == 1:
      if self.current_contract is None:
        reward += 0.01

        number = round(action[1]*self.max_lot_cap,2)
        number = number if number >= 0.01 else 0.01

        self.current_contract = Contract(1, 
                                       current_close,
                                       number)

      else:
        reward += -0.001

    if discrete_action == 2:
      if self.current_contract is None:
        reward += 0.01
        number = round(action[2]*self.max_lot_cap,2)
        number = number if number >= 0.01 else 0.01

        self.current_contract = Contract(-1, 
                                    current_close,
                                    number)
      else:
        reward += -0.001
    
    if self.profit < -self.loss_limit:
      reward += self.profit_per if self.profit_per > 0 else self.profit_per
      self.wallet += self.profit

      self.cancelled_contracts += 1

      if self.profit > 0:
          self.money_gained += self.profit
      else:
          self.money_lost += self.profit

      #reward += self.money_gained + self.money_lost

      self.last_gained = self.money_gained + self.money_lost

      self.max_lot_cap = self.wallet*self.limit_pip_value_per/self.pip_value
      self.loss_limit = self.wallet*self.loss_limit_per/100
      self.current_contract = None
      self.profit = 0
      self.profit_per = 0
      self.last_profit_per = 0

    if discrete_action >= 3:
      if self.current_contract is not None:
        reward += self.profit_per if self.profit_per > 0 else self.profit_per
        self.wallet += self.profit

        if self.profit > 0:
           self.money_gained += self.profit
        else:
           self.money_lost += self.profit
        
        #reward += self.money_gained + self.money_lost

        self.max_lot_cap = self.wallet*self.limit_pip_value_per/self.pip_value
        self.loss_limit = self.wallet*self.loss_limit_per/100
        self.current_contract = None
        self.profit = 0
        self.profit_per = 0
        self.last_profit_per = 0
      else:
         reward += -0.002
    
    # if self.current_minute%self.time_to_evaluate == 0:
    #    self.wallet_to_evaluate = self.wallet
    # else:
    #    dif = self.wallet - self.wallet_to_evaluate 
    #    por = -self.wallet_to_evaluate * 0.06
    #    self.abort = dif <= por
    #    #reward += -0.1
    
    obs = self.BuildObservation()
    self.reward_total += reward

    self.current_minute += 1
    self.steps += 1

    if self.steps >= self.steps_limit:
      isFinished = True
      self.last_gained = self.money_gained + self.money_lost

    self.log = f'-------------------------------------------------- \n' + \
               f'Brain name {self.brain_name} \n' + \
               f'Action: {action} \n' + \
               f'Action counter:  {discrete_action} \n' + \
               f'Wallet : {self.wallet} \n' + \
               f'Money Gained : {self.money_gained} \n' + \
               f'Money Lost : {self.money_lost} \n' + \
               f'Last Gained: {self.last_gained} \n' + \
               f'Current contract type:  {self.current_contract.type if self.current_contract is not None else 0} \n' + \
               f'Current contract lot_size:  {self.current_contract.lot_size if self.current_contract is not None else 0} \n' + \
               f'Current contract: {self.current_contract.vol_start if self.current_contract is not None else 0} \n' + \
               f'Cancelled contracts: {self.cancelled_contracts} \n' + \
               f'Abort: {self.abort} \n ' + \
               f'Step count (min): {self.steps} \n' + \
               f'Current minute: {self.current_minute} \n' + \
               f'Profit: {self.profit} \n' + \
               f'Max lot capacity: {self.max_lot_cap} \n' + \
               f'Loss limit: {self.loss_limit} \n' + \
               f'Reward step: {reward} \n' + \
               f'Reward total {self.reward_total}'


    print(self.log)
    info = {"abort":self.abort,
            "wallet":self.wallet}

    return obs, reward, isFinished, info


  def reset(self):

    #self.wallet = round(random.uniform(100.00, 700.00), 2) ##Ej 300
    self.max_lot_cap = self.wallet*self.limit_pip_value_per/self.pip_value
    self.loss_limit = self.wallet*self.loss_limit_per/100

    self.current_contract = None
    self.profit = 0
    self.steps = 0
    self.profit_per = 0
    self.last_profit_per = 0

    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    d1 = datetime.strptime('1/1/2019 06:00:00', '%m/%d/%Y %H:%M:%S')
    d2 = datetime.strptime('1/1/2023 06:00:00', '%m/%d/%Y %H:%M:%S')

    while True:
        fecha = self.random_date(d1, d2)
        if fecha.weekday() == 4 and fecha.hour > 16:
            continue
        if fecha.weekday() > 4:
            continue
        break

    fecha.replace(tzinfo=self.timezone)

    # get 10 EURUSD H4 bars starting from 01.10.2020 in UTC time zone
    rates = mt5.copy_rates_from(self.quote, mt5.TIMEFRAME_M1, fecha, self.total_min)

    # create DataFrame out of the obtained data
    self.current_time_lapse_data = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    self.current_time_lapse_data['time']=pd.to_datetime(self.current_time_lapse_data['time'], unit='s')

    print(self.current_time_lapse_data)
    self.current_minute = self.ticks_before

    self.close_values = self.current_time_lapse_data["close"]

    observation = self.BuildObservation()

    # input("Enter")

    return observation  # reward, done, info can't be included
  
  def random_date(self, start, end):
      """
      This function will return a random datetime between two datetime 
      objects.
      """
      delta = end - start
      int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
      random_second = randrange(int_delta)
      return start + timedelta(seconds=random_second)
    
  def render(self, mode='human'):
    pass
 
 
  def close (self):
    pass
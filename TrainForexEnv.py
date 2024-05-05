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

import pickle

#La clase FakeNewsEnv es una clase que implementa un gym environment
#Su objetivo es definir 

class ForexTrainEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, money,
                quote,
                ticks_before = 10,
                brain_name = '', 
                random_money = False, 
                begin_date = '',
                end_date = '' ):
    super(ForexTrainEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:

    print(torch.version.cuda)

    self.brain_name = brain_name

    self.ticks_span = 30

    self.ticks_before = ticks_before

    self.money_gained = 0
    self.money_lost = 0

    self.random_money = random_money
    if self.random_money:
      self.wallet = round(random.uniform(100.00, 500.00), 2) ##Ej 300
    else:
      self.wallet=money
    
    self.begin_date = begin_date 
    self.end_date = end_date

    self.quote = quote ##Ej EURUSD
    
    self.loss_limit_per = 1
    self.loss_limit = self.wallet*self.loss_limit_per/100 ##Ir actualizando

    self.profit = 0

    self.gained = 0

    self.buy_contracts_count = 0
    self.sell_contracts_count = 0

    self.cancelled_contracts = 0
    self.abort = False

    self.mas_stonks = 0
    self.menos_stonks = 0

    self.ticks_to_die = 3

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

    self.limit_pip_value_per = 0.005

    self.max_lot_cap = self.wallet*self.limit_pip_value_per/self.pip_value ##Esto debe irse actualizando

    self.current_contract = None


    self.current_time_lapse_data = None
    self.current_minute = 0

    self.timezone = pytz.timezone("Etc/UTC")

    self.log = f'-------------------------------------------------- \n' + \
          f'Money: {self.wallet} \n' + \
          f'Ventana de tiempo: {self.ticks_span} \n' + \
          f'Ticks before: {self.ticks_before} \n' + \
          f'Loss limit: {self.loss_limit} \n' + \
          f'Quote: {self.quote} \n' + \
          f'pip value: {self.pip_value} \n' + \
          f'Max lot capacity: {self.max_lot_cap}' 


    print(self.log)

    self.reward_total = 0

    self.total_days = 0

    self.action_space = spaces.Box(low=-0.0, high=1.0,
                                    shape=(3,), dtype=np.float32)


    self.observation_space = spaces.Box(low=-5.0, high=5.0,
                                        shape=(3 + (self.ticks_before + 2) + 5,), dtype=np.float32)

  def BuildObservation(self):
    #Grab the first sentence

    # 1ra parte Wallet 345
    # .3
    # .4
    # .5
    digits_from_wallet = []
    for i in list(range(3))[::-1]:
      digits_from_wallet.append((self.wallet // 10**i % 10)/10)

    # 2.da parte Velas / open - close
    window = self.current_time_lapse_data[self.current_minute - self.ticks_before:self.current_minute+1]
    window_values = window[["open"]].astype(np.float32).values
  
    window_values_flatten = []
    for l in window_values:
        window_values_flatten.extend(l)
    
    window_values_flatten.append(window['close'].astype(np.float32).values[-1])

    # Contrato activo
    contract_data = []

    if self.current_contract is None:
      contract_data = [0,0,0,0,0]
    else:
      contract_data = [self.current_contract.type, #-1 venta #1 compra 
                       self.current_contract.vol_start, #cuanto empezo
                       self.current_contract.lot_size, #cuanto volumen
                       self.close_values[self.current_minute], # cual es el valor actual
                       self.profit/self.wallet, ##Profit loss
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
    
    ## Checar contrato #################################################

    if self.current_contract is not None:
      self.profit = self.current_contract.type * \
                         self.current_contract.lot_size * \
                         self.pip_value * \
                         (current_close-self.current_contract.vol_start) / \
                          (1/10**(self.pip_digits-1))

    # Ignorar ##########################################################

    if discrete_action == 0:
      if self.current_contract is None:
        reward += -0.02
      else:
        reward += -0.01

    ###################################################################
    
    #Contrato de compra#################################################

    if discrete_action == 1:
      if self.current_contract is None:

        self.buy_contracts_count += 1

        reward += 0.01
        number = round(action[1]*self.max_lot_cap,2)
        number = number if number >= 0.01 else 0.01

        self.current_contract = Contract(1, 
                                       current_close,
                                       number)

      else:
        reward += -0.01
    
    ###################################################################

    #Contrato de venta#################################################

    if discrete_action == 2:
      if self.current_contract is None:
        self.sell_contracts_count += 1

        reward += 0.01
        number = round(action[2]*self.max_lot_cap,2)
        number = number if number >= 0.01 else 0.01

        self.current_contract = Contract(-1, 
                                    current_close,
                                    number)
      else:
        reward += -0.01

    ###################################################################


    #Cerrar contrato################################################

    if discrete_action >= 3:
      if self.current_contract is not None:
        
        reward += self.closeContract()

      else:
         reward += -0.002

    ###################################################################

    #Cancelar el contrato si no stonks xD ##############################

    if self.profit < -(self.wallet*5)/100:
      self.dying_ticks += 1
      if self.dying_ticks == self.ticks_to_die:

        reward += self.closeContract()

    else:
      self.dying_ticks = 0
    
    ################################################################

    if self.current_minute >= self.ticks_span - 1:
      isFinished = True
      if self.current_contract is not None:
        
        reward += self.closeContract()

    self.reward_total += reward

    self.log = f'-------------------------------------------------- \n' + \
               f'Brain name {self.brain_name} \n' + \
               f'Action: {action} \n' + \
               f'Action counter:  {discrete_action} \n' + \
               f'Wallet : {self.wallet} \n' + \
               f'Money Gained : {self.money_gained} \n' + \
               f'Money Lost : {self.money_lost} \n' + \
               f'Gained: {self.gained} \n' + \
               f'Current contract type:  {self.current_contract.type if self.current_contract is not None else 0} \n' + \
               f'Current contract lot_size:  {self.current_contract.lot_size if self.current_contract is not None else 0} \n' + \
               f'Current contract: {self.current_contract.vol_start if self.current_contract is not None else 0} \n' + \
               f'Current close: {current_close} \n' + \
               f'Buy contracts: {self.buy_contracts_count} \n' + \
               f'Sell contracts: {self.sell_contracts_count} \n' + \
               f'Cancelled contracts: {self.cancelled_contracts} \n' + \
               f'Abort: {self.abort} \n ' + \
               f'Step count (min): {self.steps} \n' + \
               f'Current minute: {self.current_minute} \n' + \
               f'Profit: {self.profit} \n' + \
               f'Max lot capacity: {self.max_lot_cap} \n' + \
               f'Loss limit: {self.loss_limit} \n' + \
               f'Mas stonks: {self.mas_stonks} \n' + \
               f'Menos Stonks: {self.menos_stonks} \n' + \
               f'Reward step: {reward} \n' + \
               f'Reward total {self.reward_total} \n'  + \
               f'Total days total {self.total_days}'


    print(self.log)
    info = {"abort":self.abort,
            "wallet":self.wallet}

    self.current_minute += 1
    self.steps += 1

    obs = self.BuildObservation()

    # print(self.current_time_lapse_data[self.current_minute - self.ticks_before:self.current_minute+1])
    # print(obs)

    return obs, reward, isFinished, info

  def closeContract(self):

    prof = self.profit

    self.wallet += self.profit

    if self.profit > self.mas_stonks:
        self.mas_stonks = self.profit
    if self.profit < self.menos_stonks:
        self.menos_stonks = self.profit

    if self.profit > 0:
        self.money_gained += self.profit
    else:
        self.money_lost += self.profit
      
    self.gained = self.money_gained + self.money_lost

    self.max_lot_cap = self.wallet*self.limit_pip_value_per/self.pip_value
    self.loss_limit = self.wallet*self.loss_limit_per/100

    self.current_contract = None
    self.dying_ticks = 0

    self.profit = 0

    return prof

  def reset(self):

    self.total_days = self.total_days + 1

    if self.random_money:
      self.wallet = round(random.uniform(100.00, 500.00), 2) ##Ej 300

    self.closeContract()

    self.steps = 0

    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    d1 = datetime.strptime(self.begin_date, '%m/%d/%Y %H:%M:%S')
    d2 = datetime.strptime(self.end_date, '%m/%d/%Y %H:%M:%S')

    while True:
        fecha = self.random_date(d1, d2)
        if fecha.weekday() == 5 and fecha.hour > 12:
            continue
        if fecha.weekday() > 5:
            continue
        break

    fecha.replace(tzinfo=self.timezone)
    # get 10 EURUSD H4 bars starting from 01.10.2020 in UTC time zone
    rates = mt5.copy_rates_from(self.quote, mt5.TIMEFRAME_M30, fecha, self.ticks_span+1)

    self.current_time_lapse_data = pd.DataFrame(rates)
    self.current_time_lapse_data['time']=pd.to_datetime(self.current_time_lapse_data['time'], unit='s')

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

  def write_report(self, path_name, agente):
    
    report = {"agent_id" : agente,
              "total_reward":self.reward_total,
              "money_gained":self.gained}
    
    file = open(path_name,'wb')
    pickle.dump(report,file)
    file.close()
    
  def render(self, mode='human'):
    pass
 
 
  def close (self):
    pass
import sys
import os

from datetime import datetime
import pickle

import time

username = "sonic"

now = datetime.now() # current date and time
fecha = now.strftime("%m %d %Y, %H-%M-%S")

modelo = "TD3"

num_agentes = 3
generaciones = 10





checkpoint_path = "None"
buffer_path = "None"

for generacion in range(generaciones):

    directorio = f'C:\\Users\\sonic\\OneDrive\\Escritorio\\GAINSRL\\modelos\\{modelo}\\{modelo} - {fecha}\\gen {generacion}'

    if not os.path.exists(directorio):
        os.makedirs(directorio)

    for agente in range(num_agentes):

        flags = f'-f "{fecha}" -d "{directorio}" -a {agente} -m {modelo} -b "{checkpoint_path}" -k "{buffer_path}" -g {generacion}'

        os.system('start cmd /k ; cd C:\\Users\\' + username + '\\Desktop\\GAINSRL ^& python train_core_multi.py '+ flags + ' ^& exit')


    while True:
        if len(os.listdir(directorio+"\\")) == num_agentes * 3: 
            break
        time.sleep(10)

    report_files = []

    for file in os.listdir(directorio+"\\"):
        if file.startswith("report"):
            report_files.append(os.path.join(directorio+"\\", file))

    report_list = []

    gen_log_file = open(directorio + "\\log.txt", "a")
    print_log = ''

    for file_string in report_files:
        file = open(file_string,'rb')
        data = pickle.load(file)
        report_list.append(data)
        print_log += "---------------------------\n"+\
                    f"Generacion: {generacion}" +\
                    f'{data} \n'
        file.close()
    
    agents_ord = sorted(report_list, key=lambda x: x['money_gained'], reverse=True)

    selected = agents_ord[0]["agent_id"]
    print_log += "++++++++++++++++++++++++++++++ \n"+\
                 f'El seleccionado fue {selected} \n'
    print(print_log)
    gen_log_file.write(print_log)
    gen_log_file.close()

    for file in os.listdir(directorio+"\\"):
        if file.endswith(f'{selected}.zip'):
            checkpoint_path = os.path.join(directorio+"\\", file)
            break

    for file in os.listdir(directorio+"\\"):
        if file.endswith(f'{selected}.pkl'):
            buffer_path = os.path.join(directorio+"\\", file)
            break




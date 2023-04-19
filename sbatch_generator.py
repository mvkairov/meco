import argparse

params = {
    'name': 'paramSearch',
    'mail': 'mkairov@hse.ru',
    'cpus': 32,
    'gpus': 0,
    'time': '8:00:00',
    'env': 'meco',
    'command': 'python3 run_cmd.py',
    'params': ''
}

script = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --mail-user={mail}
#SBATCH --mail-type=ALL
#SBATCH --error=runs/run-%j.err
#SBATCH --output=runs/run-%j.log
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={gpus}
#SBATCH --time={time}

module load Python/Anaconda_v11.2021
source deactivate
source activate {env}

{command} {params}
"""

CLI = argparse.ArgumentParser(prog='SBatch script generator')

CLI.add_argument('-n', '--name', type=str)
CLI.add_argument('--mail', type=str)
CLI.add_argument('--cpus', type=str)
CLI.add_argument('--gpus', type=str)
CLI.add_argument('-t', '--time', type=str)
CLI.add_argument('--env', type=str)
CLI.add_argument('-c', '--command', type=str)
CLI.add_argument('-p', '--params', type=str)

if __name__ == '__main__':
    args = CLI.parse_args()
    for name, value in vars(args).items():
        if value is not None:
            params[name] = value
    script = script.format(**params)
    print(script)
    file = open(f'{params["name"]}.sbatch', 'w')
    file.write(script)
    file.close()

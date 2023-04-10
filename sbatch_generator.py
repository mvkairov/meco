import argparse

params = {
    'name': 'test',
    'mail': 'mkairov@hse.ru',
    'cpus': 4,
    'gpus': 0,
    'time': '0:15:00',
    'env': 'meco',
    'langs': None,
    'classifiers': None,
    'input': 'Datasets/DataToUse/',
    'jobs': -1,
    'test': '',
    'send_to_tg': '-s'
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

python3 Classifiers/cmd_run_classifiers.py -l {langs} -i {input} -c {classifiers} -j {jobs} {test} {send_to_tg}
"""

CLI = argparse.ArgumentParser(prog='SBatch script generator')

CLI.add_argument('--name', type=str)
CLI.add_argument('--mail', type=str)
CLI.add_argument('--cpus', type=str)
CLI.add_argument('--gpus', type=str)
CLI.add_argument('--time', type=str)
CLI.add_argument('--env', type=str)

CLI.add_argument('-l', '--langs', nargs='*', type=str, required=True)
CLI.add_argument('-c', '--classifiers', nargs='*', type=str)
CLI.add_argument('-i', '--input', type=str)
CLI.add_argument('-j', '--jobs', type=int)
CLI.add_argument('-t', '--test', action='store_const', const='-t')
CLI.add_argument('-s', '--send_to_tg', action='store_const', const='-s')

if __name__ == '__main__':
    args = CLI.parse_args()
    for name, value in vars(args).items():
        if value is not None:
            params[name] = value
    params['langs'] = ' '.join(params['langs'])
    params['classifiers'] = ' '.join(params['classifiers'])
    script = script.format(**params)
    file = open(f'{params["name"]}.sbatch', 'w')
    file.write(script)
    file.close()

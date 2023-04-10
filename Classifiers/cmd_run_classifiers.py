from run_grid_search_cv import *
from default_values import *
import argparse
import requests
import time
import os
import numpy as np

langs_ = None
input_ = 'Datasets/DataToUse/'
n_jobs_ = 4
classifiers_ = default_classifiers.keys()


def parse_lang(lang_name):
    specs = lang_name.split('-')
    cur_langs = specs.pop(0).split('+')
    if cur_langs == ['all']:
        cur_langs = lang_list
    cur_cols = specs.pop(0)
    spec_dict = {}
    for spec in specs:
        col_name, col_values = spec.split('=')
        spec_dict[col_name] = col_values.split('+')
    lang_params = {
        'langs': cur_langs,
        'cols': cur_cols,
        'params': spec_dict
    }
    return lang_params


def make_msg(clf_name, lang_name, gs_params, cv_score, el_time):
    msg = f'For parameters used on {clf_name} with {lang_name}:\n'
    for name, value in gs_params.items():
        msg += f'\t{name} = {value}\n'
    msg += 'results below were obtained (mean, std):\n'
    for name, value in cv_score.items():
        msg += f'\t{name}: ({np.round(np.mean(value), 3)}, {np.round(np.std(value), 3)})\n'
    msg += f'\nElapsed time: {round(el_time, 3)} seconds\n'
    return msg


def send_tg_message(msg):
    BOT_TOKEN = os.environ.get('TG_BOT_TOKEN')
    CHAT_ID = os.environ.get('TG_CHAT_ID')

    URL = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    formatted_text = f'```\n{msg}\n```'
    msg_data = {
        'chat_id': CHAT_ID,
        'text': formatted_text,
        'parse_mode': 'markdown'
    }
    requests.post(url=URL, json=msg_data)


CLI = argparse.ArgumentParser(prog='GridSearch+CV results')
CLI.add_argument('-l', '--langs', nargs='*', type=str, required=True)
CLI.add_argument('-c', '--classifiers', nargs='*', type=str)
CLI.add_argument('-i', '--input', type=str)
CLI.add_argument('-j', '--jobs', type=int)
CLI.add_argument('-t', '--test', action='store_true')
CLI.add_argument('-s', '--send_to_tg', action='store_true')

if __name__ == '__main__':
    args = CLI.parse_args()
    if args.langs is not None:
        langs_ = args.langs
    if args.input is not None:
        input_ = args.input
    if args.classifiers is not None and len(args.classifiers) != 0:
        classifiers_ = args.classifiers
    if args.jobs is not None:
        n_jobs_ = args.jobs
    is_test_run = args.test
    send_to_tg = args.send_to_tg

    for clf in classifiers_:
        for lang in langs_:
            cur_params = parse_lang(lang)
            X, y = make_dataset(**cur_params, path_to_data=input_)

            st = time.time()
            if is_test_run:  # run on one set of parameters for debugging
                params, score = run_grid_search_cv(X, y, classifier=default_classifiers[clf],
                                                   params=test_params[clf], n_jobs=n_jobs_)
            else:
                params, score = run_grid_search_cv(X, y, classifier=default_classifiers[clf],
                                                   params=default_params[clf], n_jobs=n_jobs_)
            et = time.time()

            message = make_msg(clf, lang, params, score, et - st)
            print(message)
            if send_to_tg:
                send_tg_message(message)

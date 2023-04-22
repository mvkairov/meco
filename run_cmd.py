from Models.search_methods import search_cv
from Preprocessing.MECO_data_split import make_dataset, lang_list
import argparse
import requests
import time
import os

langs_ = models_ = None
input_ = 'Datasets/DataToUse/'
n_jobs_ = 4
method_ = 'Classification'
target_ = 'Target_Label'
search_ = 'Grid'


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


def make_msg(clf_name, lang_name, search_results, cv_results, el_time):
    msg = f'For parameters used on {clf_name} with {lang_name}:\n'
    for name, value in search_results.items():
        msg += f'\t{name} = {value}\n'
    msg += 'results below were obtained (mean, std):\n'
    for name, value in cv_results.items():
        msg += f'\t{name}: {value}\n'
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
CLI.add_argument('-m', '--models', nargs='*', type=str, required=True)
CLI.add_argument('--method', type=str)
CLI.add_argument('--target', type=str)
CLI.add_argument('--search', type=str)
CLI.add_argument('-i', '--input', type=str)
CLI.add_argument('-j', '--jobs', type=int)
CLI.add_argument('-d', '--debug', action='store_true')
CLI.add_argument('-s', '--send_to_tg', action='store_true')

if __name__ == '__main__':
    args = CLI.parse_args()
    if args.langs is not None:
        langs_ = args.langs
    if args.models is not None:
        models_ = args.models
    if args.method is not None:
        method_ = args.method
    if args.target is not None:
        target_ = args.target
    if args.search is not None:
        search_ = args.search
    if args.input is not None:
        input_ = args.input
    if args.jobs is not None:
        n_jobs_ = args.jobs
    debug = args.debug
    send_to_tg = args.send_to_tg

    for lang in langs_:
        cur_params = parse_lang(lang)
        X, y, value_idx = make_dataset(**cur_params, target=target_, path_to_data=input_)

        for model in models_:
            st = time.time()
            search_params, cv_score = search_cv(X, y, model, method_, search_, value_idx, n_jobs=n_jobs_, debug=debug)
            et = time.time()

            message = make_msg(model, lang, search_params, cv_score, et - st)
            print(message)
            if send_to_tg:
                send_tg_message(message)

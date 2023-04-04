from run_grid_search_cv import *
import argparse
import requests
import time

langs_ = None
input_ = 'Datasets/DataToUse/'
output_ = sys.stdout
output_name_ = None
cid_ = 381845635

BOT_TOKEN = '6048346687:AAHqaCJC7fYtwJVP1uBm7Q6zPH9TciI_oqA'
URL = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'

lang_list = ['du', 'ee', 'fi', 'ge', 'gr', 'he', 'it', 'no', 'ru', 'sp', 'it']

test_params = {
    'knn': {
        'n_neighbors': [10],
        'p': [2.5],
    },
    'mlp': {
        'hidden_layer_sizes': [100],
        'max_iter': [10000],
        'learning_rate_init': [10 ** (-3)],
        'activation': ['relu'],
        'solver': ['adam'],
    },
    'abc': {
        'learning_rate': [0.1],
        'n_estimators': [100],
    },
    'rfc': {
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    },
    'gbc': {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'min_samples_split': [5],
        'min_samples_leaf': [5],
    }
}

CLI = argparse.ArgumentParser(prog='GridSearch+CV computed on')
CLI.add_argument('-l', '--langs', nargs='*', type=str, required=True)
CLI.add_argument('-i', '--input', type=str)
CLI.add_argument('-o', '--output', type=str)
CLI.add_argument('-c', '--cid', type=str)
CLI.add_argument('-t', '--test', action='store_true')

if __name__ == '__main__':
    args = CLI.parse_args()
    if args.langs is not None:
        langs_ = args.langs
    if args.input is not None:
        input_ = args.input
    if args.output is not None:
        output_name_ = args.output
    if args.cid is not None:
        cid_ = args.cid
    is_test_run = args.test

    for lang in langs_:
        cur_params = {}
        specs = lang.split('-')
        cur_langs = specs.pop(0).split('+')
        if cur_langs == ['all']:
            cur_langs = lang_list
        cur_cols = specs.pop(0)
        spec_dict = {}
        for spec in specs:
            col_name, col_values = spec.split('=')
            spec_dict[col_name] = col_values.split('+')
        cur_params = {
            'langs': cur_langs,
            'cols': cur_cols,
            'params': spec_dict
        }

        if output_name_ is not None:
            output_ = open(args.output, 'w')
            print(file=output_)

        X, y = make_dataset(cur_params['langs'],
                            cols=cur_params['cols'],
                            params=cur_params['params'], path_to_data=input_)
        st = time.time()
        if is_test_run:  # run on one set of parameters for debugging
            run_check_grid_search_for_dataset(X, y, name=lang,
                                              classifiers={'knn': default_classifiers['knn']},
                                              params=test_params, output=output_)
        else:
            run_check_grid_search_for_dataset(X, y, name=lang, output=output_)
        et = time.time()

        if output_name_ is not None:
            output_.close()

        # Telegram bot for sending results
        if cid_ is not None and output_name_ is not None:
            res_file = open(output_name_, 'r')
            results = res_file.read()
            res_file.close()

            formatted_text = f'```{results}```\n' + f'Elapsed time: {round(et - st, 3)} seconds'
            msg_data = {
                'chat_id': cid_,
                'text': formatted_text,
                'parse_mode': 'markdown'
            }
            result = requests.post(url=URL, json=msg_data)

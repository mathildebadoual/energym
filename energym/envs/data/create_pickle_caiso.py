"""
Store CAISO requests into a dataframe stored into a pickle file
"""
from pyiso import client_factory
import pandas as pd
import datetime
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, default='20150601')
    parser.add_argument('--end_date', type=str, default='20181101')
    parser.add_argument('--which', type=str, default='all',
                        help="choose between 'all', 'gen' and 'load'")
    parser.add_argument('--test_train', help='divide the set in train and test test',
                        action='store_true')
    parser.add_argument('--percentage', type=float, default=0.1)
    args = parser.parse_args()

    caiso = client_factory('CAISO')

    start_date = datetime.datetime(int(args.start_date[0:4]),
                                   int(args.start_date[4:6]),
                                   int(args.start_date[6:8]))
    end_date = datetime.datetime(int(args.end_date[0:4]),
                                 int(args.end_date[4:6]),
                                 int(args.end_date[6:8]))

    if args.test_train:
        period = end_date - start_date
        train_period = int(period.days * (1 - args.percentage))
        test_period = period.days - train_period
        start_date_train = start_date
        end_date_train = start_date + datetime.timedelta(days=train_period)
        start_date_test = end_date_train
        end_date_test = start_date_test + datetime.timedelta(days=test_period)
        if args.which == 'all' or args.which == 'gen':
            download(start_date_train, end_date_train, caiso, 'gen', 'gen_caiso_train')
            download(start_date_test, end_date_test, caiso, 'gen', 'gen_caiso_test')

        if args.which == 'all' or args.which == 'load':
            download(start_date_train, end_date_train, caiso, 'load', 'dem_caiso_train')
            download(start_date_test, end_date_test, caiso, 'load', 'dem_caiso_test')

    else:
        if args.which == 'all' or args.which == 'gen':
            download(start_date, end_date, caiso, 'gen', 'gen_caiso')

        if args.which == 'all' or args.which == 'load':
            download(start_date, end_date, caiso, 'load', 'dem_caiso')


def download(start_date, end_date, caiso, type, name_picke):
    all_data = []
    date_0 = start_date
    date_1 = start_date + datetime.timedelta(days=30)
    pbar = tqdm(total=(end_date - start_date).days // 30)
    while date_1 <= end_date:
        if type == 'gen':
            all_data += caiso.get_generation(start_at=date_0, end_at=date_1)
        if type == 'load':
            all_data += caiso.get_load(start_at=date_0, end_at=date_1)
        date_0, date_1 = date_1, date_1 + datetime.timedelta(days=30)
        pbar.update(1)
    pbar.close()
    df = pd.DataFrame(all_data)
    df.to_pickle(name_picke + '.pkl')


if __name__ == "__main__":
    main()

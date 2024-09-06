from tbparse import SummaryReader
from matplotlib import pyplot as plt
import pandas as pd

def logs_dir_to_dataframe(logs_dir):
    reader = SummaryReader(logs_dir, extra_columns={'dir_name', 'wall_time'})
    raw_df = reader.scalars
    raw_df.to_csv('raw.csv')

    raw_df['col_name'] = raw_df[['dir_name', 'tag']].agg('/'.join, axis=1)

    raw_df.drop(columns=['dir_name', 'tag'], inplace=True)

    raw_df['time'] = pd.to_datetime(raw_df['wall_time'], unit='s')

    pivot_df = raw_df.pivot_table(index='step', columns='col_name', values='value')

    pivot_df.interpolate(method='linear', inplace=True)

    pivot_df.index.rename('step')

    return pivot_df

if __name__ == '__main__':
    df = logs_dir_to_dataframe('../../runs')
    df.to_csv('tensorboard_data.csv')

    df['estimate/bearing/altitude'].plot()
    plt.show()
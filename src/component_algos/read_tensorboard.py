from tensorflow.python.summary.summary_iterator import summary_iterator
# https://stackoverflow.com/a/40029298/14587004
from glob import glob
import pandas as pd


def logs_dir_to_dataframe(logs_dir: str) -> pd.DataFrame:
    '''
    Given a tensorboard log directory path, return a pandas dataframe
    where the columns are the scalar names plus columns for step and time,
    and the rows are the values at each step.
    '''

    if not logs_dir.endswith('/'):
        logs_dir += '/'

    # Get all the event files
    unique_steps = set()
    unique_cols = set()
    events_files = glob(logs_dir + "/**/*.tfevents.*", recursive=True)
    def iter_values():
        for logfile in events_files:
            if 'filter' in logfile:
                continue
            for event in summary_iterator(logfile):
                for value in event.summary.value:
                    yield logfile, event, value
        
    for logfile, event, value in iter_values():
        tag_base = logfile[len(logs_dir):logfile.index('/events.out')]
        col_name = f'{tag_base}/{value.tag}'
        unique_cols.add(col_name)
        unique_steps.add(event.step)
    print(f'unique steps: len {len(unique_steps)}')
    
    steps_ordered = sorted(list(unique_steps))
    output_df = pd.DataFrame(index=steps_ordered, columns=['time'] + list(unique_cols))
    for logfile, event, value in iter_values():
        tag_base = logfile[len(logs_dir):logfile.index('/events.out')]
        col_name = f'{tag_base}/{value.tag}'
        if 'launched' in col_name:
            print(tag_base, value.simple_value)
        output_df.loc[event.step, col_name] = value.simple_value
        output_df.loc[event.step, 'time'] = event.wall_time
    
    return output_df


if __name__ == '__main__':
    df = logs_dir_to_dataframe('../../runs')
    df = df.apply(pd.to_numeric)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.interpolate(method='time', inplace=True)
    df.reset_index(inplace=True)
    df.to_csv('tensorboard_data.csv')
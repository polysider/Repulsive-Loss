import os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_log_dir(log_dir_name):

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    base_dir = os.path.join(ROOT_DIR, log_dir_name)
    log_dir = os.path.join(base_dir, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    return log_dir


def save_run_info(arg_string, log_dir):

    # Store a text file in the log directory
    info_filename = os.path.join(log_dir, 'parameters_info.txt')
    with open(info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)


def main():
    log_dir = prepare_log_dir('test')
    test_args = ['foo', 'bar', 'baz']
    save_run_info(test_args, log_dir)


if __name__ == '__main__':
    main()
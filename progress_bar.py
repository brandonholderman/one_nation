from time import sleep
from progress.bar import Bar
# from progress.spinner import LineSpinner, MoonSpinner, Spinner


def run_nerd_bar():
    with Bar('Processing...' ) as bar:
        for i in range(100):
            sleep(0.02)
            bar.next()

# def run_nerd_bar():
#     with Spinner('Processing...' ) as bar:
#         for i in range(100):
#             sleep(0.05)
#             bar.next()

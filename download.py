from tcia_utils import nbia
import argparse

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--path', type=str, default='', help="manifest file path")
args = parser.parse_args()

nbia.downloadSeries(args.path, input_type='manifest')
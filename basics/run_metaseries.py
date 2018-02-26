import subprocess
import argparse
import numpy as np

'''
Run a code many times, each time with a one parameter given a different value
Example usage:

python run_metaseries.py -pro haldane_chern_class -opts ensure_chern/-LT/hexagonal/-shape/square/-AB/0.25/-Vpin/0.1 -metavar pinconf -metavals 2:6 -var N 20:5:44
'''

parser = argparse.ArgumentParser('Run a series of programs, one by one, with different arguments')
parser.add_argument('-pro', '--program', help='Name of program to run many times with different options',
                    type=str, default='gyro_lattice_1stO_torque')
parser.add_argument('-opts', '--static_options', help='String to run for every sim, with / chars instead of spaces',
                    type=str, default='')
parser.add_argument('-var', '--vary_var', help='Variable to change with each sim', type=str, default='check_empty')
parser.add_argument('vals', type=str, nargs='?',
                    help='values to use as var for each series in turn, as val0/val1/val2/etc, OR as int:int:int or' +
                         ' float:float:float or int:int or float:float',
                    default='check_string_for_empty')
parser.add_argument('-metavar', '--metavar', help='Variable to change with each sim', type=str, default='check_empty')
parser.add_argument('-metavals', '--metavals',
                    help='values to use as var for all sub-series, as val0/val1/val2/etc, OR as int:int:int or' +
                         ' float:float:float or int:int or float:float',
                    type=str, default='check_string_for_empty')
    
args = parser.parse_args()

if args.vals == 'check_string_for_empty':
    raise RuntimeError('ERROR! No values to assign.')
elif ':' in args.vals:
    values = args.vals.split(':')
    if '.' in values[0] or '.' in values[1]:
        print 'Supplied values are floats...'
        print 'values = ', values
        start = float(values[0].replace('n', '-'))
        if len(values) == 3:
            step = float(values[1].replace('n', '-'))
            end = float(values[2].replace('n', '-'))
        elif len(values) == 2:
            step = float(1)
            end = float(values[1].replace('n', '-'))
        else:
            raise RuntimeError('If : is used, vals must be ##:## or ##:##:##')
    else:
        print 'Values are integers...'
        start = int(values[0].replace('n', '-'))
        if len(values) == 3:
            step = int(values[1].replace('n', '-'))
            end = int(values[2].replace('n', '-'))
        elif len(values) == 2:
            print 'Incrementing by 1 between each subprocess call...'
            step = 1
            end = int(values[1].replace('n', '-'))
        else:
            raise RuntimeError('If : is used, vals must be ##:## or ##:##:##')
    print 'setting vals_nums to np.arange(', start, ',', end, ',', step, ')...'
    vals_nums = np.arange(start, end, step)
    vals = [str(vali) for vali in vals_nums]
else:
    vals = np.array([args.vals.split('/')[i] for i in range(len(args.vals.split('/')))])
    
# print vals
optstr = args.static_options
varstr = '/-' + args.vary_var
print "optstr = ", optstr
metalist = ['-var', args.metavar, args.metavals]

# An example string fed into subprocess is:
# python gyro_lattice_1stO_torque.py -NH 13 -NV 19 -split_spin -split_k 0.15 -freq 2.29 -modii 200 -excite_dist 4.0

for val in vals:
    if args.vary_var == 'check_empty':
        callstr = ['python', 'run_series.py', '-pro', args.program, '-opts'] + [optstr + '/' + val] + metalist
    else:
        callstr = ['python', 'run_series.py', '-pro', args.program, '-opts'] + [optstr + varstr + '/' + val] + metalist
    print 'Calling subprocess: \n', callstr
    subprocess.call(callstr)

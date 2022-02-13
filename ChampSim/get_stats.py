#!/usr/bin/env python3

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', help='Path to ChampSim results file')
    parser.add_argument('--cache-level', default='LLC', choices=('L2', 'LLC'), help='Cache level to compute stats for (default: %(default)s)')
    parser.add_argument('--base', default=None, help='Path to ChampSim base settings results file with no prefetcher for more accurate statistics')

    return parser.parse_args()


def read_file(path, cache_level):
    if path is None:
        return None

    expected_keys = ('ipc', 'total_miss', 'useful', 'useless', 'load_miss', 'rfo_miss', 'kilo_inst')
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if 'Finished CPU' in line:
                data['ipc'] = float(line.split()[9])
                data['kilo_inst'] = int(line.split()[4]) / 1000
            if cache_level not in line:
                continue
            line = line.strip()
            if 'LOAD' in line:
                data['load_miss'] = int(line.split()[-1])
            elif 'RFO' in line:
                data['rfo_miss'] = int(line.split()[-1])
            elif 'TOTAL' in line:
                data['total_miss'] = int(line.split()[-1])
            elif 'USEFUL' in line:
                data['useful'] = int(line.split()[-3])
                data['useless'] = int(line.split()[-1])

    if not all(key in data for key in expected_keys):
        return None

    return data

def main(args=None):
    print(args)
    results = read_file(args.results_file, args.cache_level)
    useful, useless, ipc, load_miss, rfo_miss, kilo_inst = (
        results['useful'], results['useless'], results['ipc'], results['load_miss'], results['rfo_miss'], results['kilo_inst']
    )
    results_total_miss = load_miss + rfo_miss + useful
    total_miss = results_total_miss

    results_mpki = (load_miss + rfo_miss) / kilo_inst

    base = read_file(args.base, args.cache_level)
    if base is not None:
        base_total_miss, base_ipc = base['total_miss'], base['ipc']
        base_mpki = base_total_miss / kilo_inst

    if useful + useless == 0:
        print('Accuracy: N/A [All prefetches were merged and were not useful or useless]')
    else:
        print('Accuracy:', useful / (useful + useless) * 100, '%')
    if total_miss == 0:
        print('Coverage: N/A [No misses. Did you run this simulation for long enough?]')
    else:
        print('Coverage:', useful / total_miss * 100, '%')
    print('MPKI:', results_mpki)
    if base is not None:
        print('MPKI Improvement:', (base_mpki - results_mpki) / base_mpki * 100, '%')
    print('IPC:', ipc)
    if base is not None:
        print('IPC Improvement:', (ipc - base_ipc) / base_ipc * 100, '%')

if __name__ == '__main__':
    main(args=get_args())

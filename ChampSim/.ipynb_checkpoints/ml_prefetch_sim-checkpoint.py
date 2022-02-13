#!/usr/bin/env python3

import argparse
import os
import sys

from model import Model

default_results_dir = './results'
default_output_file = './stats.csv'
default_spec_instrs = 40
default_gap_instrs = 40
default_warmup_instrs = 0

default_seed_file = './scripts/seeds.txt'

default_base_binary = 'bin/hashed_perceptron-no-no-no-no-lru-1core'
default_bo_binary = 'bin/hashed_perceptron-no-no-no-bo-lru-1core'
default_spp_binary = 'bin/hashed_perceptron-no-no-no-spp-lru-1core'
default_sisb_binary = 'bin/hashed_perceptron-no-no-no-sisb-lru-1core'
default_domino_binary = 'bin/hashed_perceptron-no-no-no-domino-lru-1core'
default_vldp_binary = 'bin/hashed_perceptron-no-no-no-vldp-lru-1core'
default_prefetcher_binary = 'bin/hashed_perceptron-no-no-no-from_file-lru-1core'

baseline_names = ['No Prefetcher', 'Best Offset', 'SPP', 'SISB','Domino','VLDP']
baseline_fns = ['no', 'bo','spp', 'sisb', 'domino','vldp']
baseline_binaries = [default_base_binary, default_bo_binary, default_spp_binary,default_sisb_binary, default_domino_binary,default_vldp_binary]

help_str = {
'help': '''usage: {prog} command [<args>]

Available commands:
    build            Builds base and prefetcher ChampSim binaries
    run              Runs ChampSim on specified traces
    eval             Parses and computes metrics on simulation results
    train            Trains your model
    generate         Generates the prefetch file
    help             Displays this help message. Command-specific help messages
                     can be displayed with `{prog} help command`
'''.format(prog=sys.argv[0]),

'build': '''usage: {prog} build [<target>]

Description:
    {prog} build [<target>]
        Builds <target> ChampSim binaries where <target> is one of:

            all            Builds all of the below binaries [default]
            base           Builds all the baseline binaries
            prefetcher     Builds just the prefetcher binary that reads from a file

        If <target> is unspecified, this will act as if `{prog} build all` was
        executed.

Notes:
    Barring updates to the GitHub repository, this will only need to be done once.
'''.format(prog=sys.argv[0]),

'run': '''usage: {prog} run <execution-trace> [--prefetch <prefetch-file>] [--no-base] [--results-dir <results-dir>]
                            [--num-instructions <num-instructions>] [--num-prefetch-warmup-instructions <num-warmup-instructions>]

Description:
    {prog} run <execution-trace>
        Runs the base ChampSim binary on the specified execution trace.

Options:
    --prefetch <prefetch-file>
        Additionally runs the prefetcher ChampSim binary that issues prefetches
        according to the file.

    --no-base
        When specified with --prefetch <prefetch-file>, run only the prefetcher
        ChampSim binary on the specified execution trace without the baseline
        ChampSim binaries.

    --results-dir <results-dir>
        Specifies what directory to save the ChampSim results file in. This
        defaults to `{default_results_dir}`.

    --num-instructions <num-instructions>
        Number of instructions to run the simulation for. Defaults to
        {default_spec_instrs}M instructions for the spec benchmarks and
        {default_gap_instrs}M instructions for the gap benchmarks.

    --num-prefetch-warmup-instructions <num-warmup-instructions>
        Number of instructions in millions to warm-up the simulator for before
        starting prefetching. Defaults to {default_warmup_instrs}M instructions.
        This would also be the number of instructions that you train your models
        on. By specifying this, these first instructions do not get included in
        the metric computation.

    --seed-file <seed-file>
        Path to seed file to load for ChampSim evaluation. Defaults to {seed_file}.
'''.format(prog=sys.argv[0], default_results_dir=default_results_dir,
    default_spec_instrs=default_spec_instrs, default_gap_instrs=default_gap_instrs,
    default_warmup_instrs=default_warmup_instrs, seed_file=default_seed_file),

'eval': '''usage: {prog} eval [--results-dir <results-dir>] [--output-file <output-file>]

Description:
    {prog} eval
        Runs the evaluation procedure on the ChampSim output found in the specified
        results directory and outputs a CSV at the specified output path.

Options:
    --results-dir <results-dir>
        Specifies what directory the ChampSim results files are in. This defaults
        to `{default_results_dir}`.

    --output-file <output-file>
        Specifies what file path to save the stats CSV data to. This defaults to
        `{default_output_file}`.

Note:
    To get stats comparing performance to a no-prefetcher baseline, it is necessary
    to have run the base ChampSim binary on the same execution trace.

    Without the base data, relative performance data comparing MPKI and IPC will
    not be available and the coverage statistic will only be approximate.
'''.format(prog=sys.argv[0], default_results_dir=default_results_dir, default_output_file=default_output_file),

'train': '''usage: {prog} train <load-trace> [--model <model-path>] [--generate <prefetch-file>] [--num-prefetch-warmup-instructions <num-warmup-instructions>]

Description:
    {prog} train <load-trace>
        Trains your model on the given load trace and optionally generates the
        prefetch file.

Options:
    --generate <prefetch-file>
        Outputs the prefetch file with your trained model

    --model <model-path>
        Saves model to this location. If not specified, the model is not
        explicitly saved.

    --num-prefetch-warmup-instructions <num-warmup-instructions>
        Number of instructions in millions to warm-up the simulator for before
        starting prefetching. Defaults to {default_warmup_instrs}M instructions.
        This would also be the number of instructions that you train your models
        on. By specifying this, these first instructions do not get included in
        the metric computation.
'''.format(prog=sys.argv[0], default_warmup_instrs=default_warmup_instrs),

'generate': '''usage: {prog} generate <load-trace> <prefetch-file> [--model <model-path>] [--num-prefetch-warmup-instructions <num-warmup-instructions>]

Description:
    {prog} generate <load-trace> <prefetch-file> --model <model-path>
        Generates the prefetch file using the specified model

Options:
    --num-prefetch-warmup-instructions <num-warmup-instructions>
        Number of instructions in millions to warm-up the simulator for before
        starting prefetching. Defaults to {default_warmup_instrs}M instructions.
        This would also be the number of instructions that you train your models
        on. By specifying this, these first instructions do not get included in
        the metric computation.
'''.format(prog=sys.argv[0], default_warmup_instrs=default_warmup_instrs),
}

def build_command():
    build = 'all'
    if len(sys.argv) > 2:
        if sys.argv[2] not in ['all', 'base', 'prefetcher']:
            print('Invalid build target')
            exit(-1)
        build = sys.argv[2]

    # Build no prefetcher baseline
    if build in ['all', 'base']:
        for name, fn in zip(baseline_names, baseline_fns):
            print('Building ' + name + ' ChampSim binary')
            os.system('./build_champsim.sh hashed_perceptron no no no ' + fn + ' lru 1')

    # Build prefetcher
    if build in ['all', 'prefetcher']:
        print('Building prefetcher ChampSim binary')
        os.system('./build_champsim.sh hashed_perceptron no no no from_file lru 1')

def run_command():
    if len(sys.argv) < 3:
        print(help_str['run'])
        exit(-1)

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('execution_trace', default=None)
    parser.add_argument('--prefetch', default=None)
    parser.add_argument('--no-base', default=False, action='store_true')
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--num-instructions', default=None) #default_spec_instrs if execution_trace[0].isdigit() else default_gap_instrs)
    parser.add_argument('--num-prefetch-warmup-instructions', default=default_warmup_instrs)
    parser.add_argument('--seed-file', default=default_seed_file)

    args = parser.parse_args(sys.argv[2:])

    execution_trace = args.execution_trace

    if args.num_instructions is None:
        args.num_instructions = default_spec_instrs if execution_trace[0].isdigit() else default_gap_instrs

    if not os.path.exists(args.seed_file):
        print('Seed file "' + args.seed_file + '" does not exist')
        seed = None
    else:
        with open(args.seed_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.split()[0] in os.path.basename(execution_trace):
                    seed = line.split()[1]
                    break
            else:
                print('Could not find execution trace "{}" in seed file "{}"'.format(execution_trace, args.seed_file))
                seed = None

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    if not args.no_base:
        for name, binary in zip(baseline_names, baseline_binaries):
            if not os.path.exists(binary):
                print(name + ' ChampSim binary not found')
                exit(-1)

            if seed is not None:
                cmd = '{binary} -prefetch_warmup_instructions {warm}000000 -simulation_instructions {sim}000000 -seed {seed} -traces {trace} > {results}/{base_trace}-{base_binary}.txt 2>&1'.format(
                    binary=binary, warm=args.num_prefetch_warmup_instructions, sim=args.num_instructions,
                    trace=execution_trace, seed=seed, results=args.results_dir,
                    base_trace=os.path.basename(execution_trace), base_binary=os.path.basename(binary))
            else:
                cmd = '{binary} -prefetch_warmup_instructions {warm}000000 -simulation_instructions {sim}000000 -traces {trace} > {results}/{base_trace}-{base_binary}.txt 2>&1'.format(
                    binary=binary, warm=args.num_prefetch_warmup_instructions, sim=args.num_instructions,
                    trace=execution_trace, results=args.results_dir, base_trace=os.path.basename(execution_trace),
                    base_binary=os.path.basename(binary))

            print('Running "' + cmd + '"')

            os.system(cmd)

    if args.prefetch is not None:
        if not os.path.exists(default_prefetcher_binary):
            print('Prefetcher ChampSim binary not found')
            exit(-1)

        if seed is not None:
            cmd = '<{prefetch} {binary} -prefetch_warmup_instructions {warm}000000 -simulation_instructions {sim}000000 -seed {seed} -traces {trace} > {results}/{base_trace}-{base_binary}.txt 2>&1'.format(
                prefetch=args.prefetch, binary=default_prefetcher_binary, warm=args.num_prefetch_warmup_instructions, sim=args.num_instructions,
                trace=execution_trace, seed=seed, results=args.results_dir,
                base_trace=os.path.basename(execution_trace), base_binary=os.path.basename(default_prefetcher_binary))
        else:
            cmd = '<{prefetch} {binary} -prefetch_warmup_instructions {warm}000000 -simulation_instructions {sim}000000 -traces {trace} > {results}/{base_trace}-{base_binary}.txt 2>&1'.format(
                prefetch=args.prefetch, binary=default_prefetcher_binary, warm=args.num_prefetch_warmup_instructions, sim=args.num_instructions,
                trace=execution_trace, results=args.results_dir, base_trace=os.path.basename(execution_trace),
                base_binary=os.path.basename(default_prefetcher_binary))

        print('Running "' + cmd + '"')

        os.system(cmd)

def read_file(path, cache_level='LLC'):
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

def compute_stats(trace, prefetch=None, base=None, baseline_name=None):
    if prefetch is None:
        return None

    pf_data = read_file(prefetch)

    useful, useless, ipc, load_miss, rfo_miss, kilo_inst = (
        pf_data['useful'], pf_data['useless'], pf_data['ipc'], pf_data['load_miss'], pf_data['rfo_miss'], pf_data['kilo_inst']
    )
    pf_total_miss = load_miss + rfo_miss + useful
    total_miss = pf_total_miss

    pf_mpki = (load_miss + rfo_miss) / kilo_inst

    if base is not None:
        b_data = read_file(base)
        b_total_miss, b_ipc = b_data['total_miss'], b_data['ipc']
        b_mpki = b_total_miss / kilo_inst

    if useful + useless == 0:
        acc = 'N/A'
    else:
        acc = str(useful / (useful + useless) * 100)
    if total_miss == 0:
        cov = 'N/A'
    else:
        cov = str(useful / total_miss * 100)
    if base is not None:
        mpki_improv = str((b_mpki - pf_mpki) / b_mpki * 100)
        ipc_improv = str((ipc - b_ipc) / b_ipc * 100)
    else:
        mpki_improv = 'N/A'
        ipc_improv = 'N/A'

    return '{trace},{baseline_name},{acc},{cov},{mpki},{mpki_improv},{ipc},{ipc_improv}'.format(
        trace=trace, baseline_name=baseline_name, acc=acc, cov=cov, mpki=str(pf_mpki),
        mpki_improv=mpki_improv, ipc=str(ipc), ipc_improv=ipc_improv,
    )

def eval_command():
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('--results-dir', default=default_results_dir)
    parser.add_argument('--output-file', default=default_output_file)

    args = parser.parse_args(sys.argv[2:])

    traces = {}
    for fn in os.listdir(args.results_dir):
        trace = fn.split('-hashed_perceptron-')[0]
        if trace not in traces:
            traces[trace] = {}
        if 'from_file' in fn:
            traces[trace]['prefetch'] = os.path.join(args.results_dir, fn)
        else:
            for base_fn in baseline_fns:
                if base_fn == fn.split('-hashed_perceptron-')[1].split('-')[3]:
                    traces[trace][base_fn] = os.path.join(args.results_dir, fn)

    stats = ['Trace,Baseline,Accuracy,Coverage,MPKI,MPKI_Improvement,IPC,IPC_Improvement']
    for trace in traces:
        d = traces[trace]
        if 'no' in d:
            stats.append(compute_stats(trace, d['no'], baseline_name='no'))
            stats.append(compute_stats(trace, d['prefetch'], d['no'], baseline_name='yours'))
        else:
            stats.append(compute_stats(trace, d['prefetch'], baseline_name='No Baseline'))
        for fn in baseline_fns:
            if fn in d:
                trace_stats = None
                if fn != 'no' and 'no' in d:
                    trace_stats = compute_stats(trace, d[fn], d['no'], baseline_name=fn)
                if trace_stats is not None:
                    stats.append(trace_stats)

    with open(args.output_file, 'w') as f:
        print('\n'.join(stats), file=f)

def generate_prefetch_file(path, prefetches):
    with open(path, 'w') as f:
        for instr_id, pf_addr in prefetches:
            print(instr_id, hex(pf_addr)[2:], file=f)

def read_load_trace_data(load_trace, num_prefetch_warmup_instructions):
    
    def process_line(line):
        split = line.strip().split(', ')
        return int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

    train_data = []
    eval_data = []
    with open(load_trace, 'r') as f:
        for line in f:
            pline = process_line(line)
            if pline[0] < int(num_prefetch_warmup_instructions) * 1000000:
                train_data.append(pline)
            else:
                eval_data.append(pline)

    return train_data, eval_data

def train_command():
    if len(sys.argv) < 3:
        print(help_str['train'])
        exit(-1)
    #'train': '''usage: {prog} train <load-trace> [--model <model-path>] [--generate <prefetch-file>] [--num-prefetch-warmup-instructions <num-warmup-instructions>]

    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('load_trace', default=None)
    parser.add_argument('--generate', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--num-prefetch-warmup-instructions', default=default_warmup_instrs)

    args = parser.parse_args(sys.argv[2:])

    train_data, eval_data = read_load_trace_data(args.load_trace, args.num_prefetch_warmup_instructions)

    model = Model()
    model.train(train_data)

    if args.model is not None:
        model.save(args.model)

    if args.generate is not None:
        prefetches = model.generate(eval_data)
        generate_prefetch_file(args.generate, prefetches)

def generate_command():
    if len(sys.argv) < 3:
        print(help_str['generate'])
        exit(-1)

    #'generate': '''usage: {prog} generate <load-trace> <prefetch-file> [--model <model-path>] [--num-prefetch-warmup-instructions <num-warmup-instructions>]
    parser = argparse.ArgumentParser(usage=argparse.SUPPRESS, add_help=False)
    parser.add_argument('load_trace', default=None)
    parser.add_argument('prefetch_file', default=None)
    parser.add_argument('--model', default=None, required=True)
    parser.add_argument('--num-prefetch-warmup-instructions', default=default_warmup_instrs)

    args = parser.parse_args(sys.argv[2:])

    model = Model()
    model.load(args.model)

    _, data = read_load_trace_data(args.load_trace, args.num_prefetch_warmup_instructions)

    prefetches = model.generate(data)

    generate_prefetch_file(args.prefetch_file, prefetches)

def help_command():
    # If one of the available help strings, print and exit successfully
    if len(sys.argv) > 2 and sys.argv[2] in help_str:
        print(help_str[sys.argv[2]])
        exit()
    # Otherwise, invalid subcommand, so print main help string and exit
    else:
        print(help_str['help'])
        exit(-1)

commands = {
    'build': build_command,
    'run': run_command,
    'eval': eval_command,
    'train': train_command,
    'generate': generate_command,
    'help': help_command,
}

def main():
    # If no subcommand specified or invalid subcommand, print main help string and exit
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(help_str['help'])
        exit(-1)

    # Run specified subcommand
    commands[sys.argv[1]]()

if __name__ == '__main__':
    main()

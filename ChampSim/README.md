# Modified ChampSim for ML Prefetching Competition

We will use ChampSim to evaluate the effectiveness of your ML prefetchers.  You
prefetching models will be trained using the Load Traces that we provide (details below), 
and they will generate an Ouput File with a list of prefetches that will be fed back into 
ChampSim to compute coverage, accuracy and instructions per cycle (IPC).

## Traces:

The traces can be found at [this link](https://utexas.box.com/s/2k54kp8zvrqdfaa8cdhfquvcxwh7yn85).
Alternatively, the `download.sh` file can be used to download all of the files to
avoid bulk download restrictions from Box. You can also use the information found
in the `download_links` file to download the data in another fashion.

There are two types of traces that can be found here:
- Load traces under the folder LoadTraces that you will use to train your ML models.  The 
  load trace is a series of program's LLC accesses, and the trace format is as follows: 
```
Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
```
  The load traces are plain text CSV.

- Execution traces under the folder ChampSimTraces that ChampSim will need to
  compute IPC.  You do not need these traces to train your models, they are
  only provided to facilitate an evaluation using IPCs.  Note that you do not
  unzip execution traces as ChampSim expects it to be in the zipped format. 

## Output File

For a given Load Trace, your code should generate an output file that contains one
prefetch per line.  Each line should consist of two space-separated integral
values, the unique instruction ID for which you want to issue a prefetch and the
load address you want to prefetch.  The unique instruction ID corresponds to
the ID of the triggering load in the input Load Trace.  You can include up to two 
prefetches per load listed in the Load Trace.  You can choose not to prefetch
for a load.  Note that the prefetches should be in the order that they occur in the trace.
Should you exceed the maximum number of prefetches per load, the first two will
be kept and the remaining excess prefetches for that load will be discarded.

For example, consider a Load Trace as follows:
```
3659 cycle1 A ip1 1
5433 cycle2 B ip2 0
6928 cycle3 C ip3 0
```

Your output file could look something like this:
```
3659 A+1    # Issue first prefetch for Instruction 3569
3659 A+2    # Issue second prefetch for Instruction 3569
5433 B+8    # Issue only one prefetch for Instruction 5433
```

## Your Code:

Your code should have two modes of functioning:

1. Taking in a Training Load Trace that your model trains on
2. Taking in a Test Load Trace for which your model will produce predictions in
   the format explained above.

## Building, Running, and Evaluating

This has been rolled into one script `ml_prefetch_sim.py`. Below there are some
common use cases highlighted, but more information can be found for each of the
subcommands by running:

```
./ml_prefetch_sim.py help subcommand
```

where subcommand is any of `build|run|eval`

### Building

The following command will compile two ChampSim binaries: (1) A ChampSim binary
that reads your ML model's output from a file and uses that as a prefetcher,
and (2) A ChampSim binary with no prefetching that is to be used as a baseline

```
./ml_prefetch_sim.py build
```

### Training

```
./ml_prefetch_sim.py train path_to_load_trace --model save_path --num-prefetch-warmup-instructions num_in_millions
```

To use the above, you need to modify the `model.py` file with your model. The
prefetch warm-up instructions specify how many to include in the training set.
The remainder of the instructions are the evaluation set.

### Generating the Prefetch File

```
./ml_prefetch_sim.py generate path_to_load_trace path_to_output_prefetch_file --model save_path --num-prefetch-warmup-instructions num_in_millions
```

To use the above, you need to modify the `model.py` file with your model. The
prefetch warm-up instructions specify how many to include in the training set.
The remainder of the instructions are the evaluation set.

### Running

To run the baseline ChampSim binaries on an execution trace:

```
./ml_prefetch_sim.py run path_to_champsim_trace_here
```

To additionally run the ChampSim binary with your prefetcher:

```
./ml_prefetch_sim.py run path_to_champsim_trace_here --prefetch path_to_prefetcher_file
```

To run the ChampSim binary with your prefetcher only:

```
./ml_prefetch_sim.py run path_to_trace_here --prefetch path_to_prefetcher_file --no-base
```

### Evaluation

To evaluate the performance of ML prefetcher (and compare it against the baseline
of no prefetcher, Best Offset, SISB, and SISB Best Offset), run:

```
./ml_prefetch_sim.py eval
```

## Competition Judging

To test how submissions generalize, our test set evaluation will have two components:

- Undisclosed execution samples for the training traces: You can submit a
  pre-trained model for each benchmark in the training set, and we will
  evaluate it on a different sample of the same benchmark

- Undisclosed benchmarks: We will train and test your model on unseen
  benchmarks using the training routines that you provide

## Changes made to ChampSim for the competition:

- Add LLC prefetcher (from\_file) to load ML model prefetch predictions into ChampSim
- Modify the LLC prefetcher to provide unique instruction IDs and cycle counts
- Remove same-page restriction in src/cache.cc for more irregular prefetching
  opportunity
- Add ml\_prefetch\_sim.py to handle all of the building, running, and evaluation.

---

<p align="center">
  <h1 align="center"> ChampSim </h1>
  <p> ChampSim is a trace-based simulator for a microarchitecture study. You can sign up to the public mailing list by sending an empty mail to champsim+subscribe@googlegroups.com. Traces for the 3rd Data Prefetching Championship (DPC-3) can be found from here (https://dpc3.compas.cs.stonybrook.edu/?SW_IS). A set of traces used for the 2nd Cache Replacement Championship (CRC-2) can be found from this link. (http://bit.ly/2t2nkUj) <p>
</p>

# Clone ChampSim repository
```
git clone https://github.com/ChampSim/ChampSim.git
```

# Compile

ChampSim takes five parameters: Branch predictor, L1D prefetcher, L2C prefetcher, LLC replacement policy, and the number of cores. 
For example, `./build_champsim.sh bimodal no no lru 1` builds a single-core processor with bimodal branch predictor, no L1/L2 data prefetchers, and the baseline LRU replacement policy for the LLC.
```
$ ./build_champsim.sh bimodal no no no no lru 1

$ ./build_champsim.sh ${BRANCH} ${L1I_PREFETCHER} ${L1D_PREFETCHER} ${L2C_PREFETCHER} ${LLC_PREFETCHER} ${LLC_REPLACEMENT} ${NUM_CORE}
```

# Download DPC-3 trace

Professor Daniel Jimenez at Texas A&M University kindly provided traces for DPC-3. Use the following script to download these traces (~20GB size and max simpoint only).
```
$ cd scripts

$ ./download_dpc3_traces.sh
```

# Run simulation

Execute `run_champsim.sh` with proper input arguments. The default `TRACE_DIR` in `run_champsim.sh` is set to `$PWD/dpc3_traces`. <br>

* Single-core simulation: Run simulation with `run_champsim.sh` script.

```
Usage: ./run_champsim.sh [BINARY] [N_WARM] [N_SIM] [TRACE] [OPTION]
$ ./run_champsim.sh bimodal-no-no-no-no-lru-1core 1 10 400.perlbench-41B.champsimtrace.xz

${BINARY}: ChampSim binary compiled by "build_champsim.sh" (bimodal-no-no-lru-1core)
${N_WARM}: number of instructions for warmup (1 million)
${N_SIM}:  number of instructinos for detailed simulation (10 million)
${TRACE}: trace name (400.perlbench-41B.champsimtrace.xz)
${OPTION}: extra option for "-low_bandwidth" (src/main.cc)
```
Simulation results will be stored under "results_${N_SIM}M" as a form of "${TRACE}-${BINARY}-${OPTION}.txt".<br> 

* Multi-core simulation: Run simulation with `run_4core.sh` script. <br>
```
Usage: ./run_4core.sh [BINARY] [N_WARM] [N_SIM] [N_MIX] [TRACE0] [TRACE1] [TRACE2] [TRACE3] [OPTION]
$ ./run_4core.sh bimodal-no-no-no-lru-4core 1 10 0 400.perlbench-41B.champsimtrace.xz \\
  401.bzip2-38B.champsimtrace.xz 403.gcc-17B.champsimtrace.xz 410.bwaves-945B.champsimtrace.xz
```
Note that we need to specify multiple trace files for `run_4core.sh`. `N_MIX` is used to represent a unique ID for mixed multi-programmed workloads. 


# Add your own branch predictor, data prefetchers, and replacement policy
**Copy an empty template**
```
$ cp branch/branch_predictor.cc branch/mybranch.bpred
$ cp prefetcher/l1d_prefetcher.cc prefetcher/mypref.l1d_pref
$ cp prefetcher/l2c_prefetcher.cc prefetcher/mypref.l2c_pref
$ cp prefetcher/llc_prefetcher.cc prefetcher/mypref.llc_pref
$ cp replacement/llc_replacement.cc replacement/myrepl.llc_repl
```

**Work on your algorithms with your favorite text editor**
```
$ vim branch/mybranch.bpred
$ vim prefetcher/mypref.l1d_pref
$ vim prefetcher/mypref.l2c_pref
$ vim prefetcher/mypref.llc_pref
$ vim replacement/myrepl.llc_repl
```

**Compile and test**
```
$ ./build_champsim.sh mybranch mypref mypref mypref myrepl 1
$ ./run_champsim.sh mybranch-mypref-mypref-mypref-myrepl-1core 1 10 bzip2_183B
```

# How to create traces

We have included only 4 sample traces, taken from SPEC CPU 2006. These 
traces are short (10 million instructions), and do not necessarily cover the range of behaviors your 
replacement algorithm will likely see in the full competition trace list (not
included).  We STRONGLY recommend creating your own traces, covering
a wide variety of program types and behaviors.

The included Pin Tool champsim_tracer.cpp can be used to generate new traces.
We used Pin 3.2 (pin-3.2-81205-gcc-linux), and it may require 
installing libdwarf.so, libelf.so, or other libraries, if you do not already 
have them. Please refer to the Pin documentation (https://software.intel.com/sites/landingpage/pintool/docs/81205/Pin/html/)
for working with Pin 3.2.

Get this version of Pin:
```
wget http://software.intel.com/sites/landingpage/pintool/downloads/pin-3.2-81205-gcc-linux.tar.gz
```

**Note on compatibility**: If you are using newer linux kernels/Ubuntu versions (eg. 20.04LTS), you might run into issues (such as [[1](https://github.com/ChampSim/ChampSim/issues/102)],[[2](https://stackoverflow.com/questions/55698095/intel-pin-tools-32-bit-processsectionheaders-560-assertion-failed)],[[3](https://stackoverflow.com/questions/43589174/pin-tool-segmentation-fault-for-ubuntu-17-04)]) with the PIN3.2. ChampSim tracer works fine with newer PIN tool versions that can be downloaded from [here](https://software.intel.com/content/www/us/en/develop/articles/pin-a-binary-instrumentation-tool-downloads.html). PIN3.17 is [confirmed](https://github.com/ChampSim/ChampSim/issues/102) to work with Ubuntu 20.04.1 LTS.

Once downloaded, open tracer/make_tracer.sh and change PIN_ROOT to Pin's location.
Run ./make_tracer.sh to generate champsim_tracer.so.

**Use the Pin tool like this**
```
pin -t obj-intel64/champsim_tracer.so -- <your program here>
```

The tracer has three options you can set:
```
-o
Specify the output file for your trace.
The default is default_trace.champsim

-s <number>
Specify the number of instructions to skip in the program before tracing begins.
The default value is 0.

-t <number>
The number of instructions to trace, after -s instructions have been skipped.
The default value is 1,000,000.
```
For example, you could trace 200,000 instructions of the program ls, after
skipping the first 100,000 instructions, with this command:
```
pin -t obj/champsim_tracer.so -o traces/ls_trace.champsim -s 100000 -t 200000 -- ls
```
Traces created with the champsim_tracer.so are approximately 64 bytes per instruction,
but they generally compress down to less than a byte per instruction using xz compression.

# Evaluate Simulation

ChampSim measures the IPC (Instruction Per Cycle) value as a performance metric. <br>
There are some other useful metrics printed out at the end of simulation. <br>

Good luck and be a champion! <br>

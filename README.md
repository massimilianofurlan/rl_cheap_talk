## Overview

This repository contains the code for *"D. Condorelli, M. Furlan (2023). Cheap Talking Algorithms"*. Replication instructions are listed in the [replication](replication.md) file.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

1. **Install Julia:** Download and install Julia: [julialang.org](https://julialang.org).

2. **Install Dependencies:** Run in the Julia REPL:

    ```
    using Pkg
    Pkg.add("ArgParse")
    Pkg.add("TOML")
    Pkg.add("JLD2")
    Pkg.add("PrettyTables")
    Pkg.add("ProgressMeter")
    Pkg.add("Random")
    Pkg.add("LoopVectorization")
    Pkg.add("StatsBase")
    ```
    
3. **Clone the Repository:** Clone this repository to your local machine.

    ```bash
    git clone https://github.com/massimilianofurlan/rl_cheap_talk.git
    ```

## Usage

1. **Navigate to the Project Folder:**

    ```bash
    cd rl_cheap_talk
    ```

2. **Basic Usage:** To run a batch of simulations use

    ```bash
    julia --threads auto main.jl [options]
    ```

    You can replace ```auto``` with the desired number of threads. Options are primarily used to select the configuration of the game, such as the number of states, the extent of the sender's bias, the utility functions and the distribution over states. To get the complete list of options use  
    
    ```bash
    julia main.jl --help
    ```
    
    To change the reinforcement learning hyperparameters, such as the learning rate and exploration rate, edit the [config](config.toml) file. The section ```[default]``` is loaded by default. A different section can be loaded using the ```--config``` option.

3. **Others:** To automatically run a batch of simulations for each bias level in {0.0, 0.01, ..., 0.49, 0.5} use 

    ```bash
    julia --threads auto scripts/1_script.jl --config CONFIG_SECTION --out_dir OUT_DIR
    ```

    Replace ```CONFIG_SECTION``` with the desired section among those in the [config](config.toml) file and ```OUT_DIR``` with the desired name for the output directory. Use ```--help``` to get the complete list of options. After running the script, Tikz plots can be generated using

    ```bash
    julia --threads auto scripts/1_plots.jl --in_dir OUT_DIR
    ```

    where ```OUT_DIT``` must match the output folder of the previus command.
    Generating plots requires an additional Julia dependency. To install it, run in the Julia REPL
    
    ```
    using Pkg
    Pkg.add("PGFPlotsX")
    ```
    Note that PGFPlotsX also requires having a LaTeX installation on your machine. 

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

For more information about the AGPL-3.0 license, please visit the [GNU website](https://www.gnu.org/licenses/agpl-3.0).


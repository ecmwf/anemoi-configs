# Anemoi Configs

This repository provides information, examples, and tutorials on config files for use with Anemoi.

[Apache License 2.0](LICENSE) In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.

You can find the announcements of all new configs [here](https://github.com/ecmwf/anemoi-configs/discussions/categories/configs).

## Table of Contents

- [Introduction](#introduction)
<!-- - [Getting Started](#getting-started) -->
- [Configuration Files](#configuration-files)
<!-- - [Tutorials](#tutorials) -->
- [Tools](#tools)
<!-- - [Contributing](#contributing) -->
- [License](#license)

## Introduction

Welcome to the Anemoi Configs repository.

This is still a work in progress and will expand to include more configs, and tools. Check back soon.

<!-- ## Getting Started

To get started with Anemoi and its configuration files, follow these steps:

1. Install Anemoi.
2. Clone this repository.
3. Follow the examples and tutorials provided. -->

## Configuration Files

In [configs](/configs) you can find a curated selection of Anemoi config files detailing
how to train a model, all the way from dataset creation to finetuning.

<!-- ## Tutorials

Step-by-step tutorials to guide you through creating and modifying configuration files.

![Tutorials](tutorials/)
 -->

## Tools

We provide some basic tools to help you get started with using these configs.

## Create Tool

### Description

The `create.sh` script is a tool designed to create a new environment from a specified configuration folder. It sets up a virtual environment, installs necessary packages, and copies configuration files to a specified output directory.

### Usage

To display the help message for the `create.sh` script, run the following command:

```bash
./tools/create.sh -h
```

```shell
Usage: ./tools/create.sh <config-path> [--use-uv] [--venv-path <path>] [--output-dir <path>]

    <config-path>          Relative path to the configuration folder
    --use-uv               Optionally use uv
    --venv-path <path>     Path to create the virtual environment (default: $HOME/anemoi_configs/<config-path>/venv/)
    --output-dir <path>    Path to copy the configuration (default: $HOME/anemoi_configs/<config-path>)
    -h, --help             Display this help message
```

#### Example

To create a new environment using the configuration folder `example-config`, use the following command:

```bash
./tools/create.sh example-config --use-uv --venv-path /custom/path/to/venv --output-dir /custom/path/to/output
```

This command will:

1. Create a virtual environment at `/custom/path/to/venv`.
2. Install the necessary packages listed in `example-config/environment.txt`.
3. Copy the configuration files from `example-config` to `/custom/path/to/output`.


<!-- 
## Contributing

We welcome contributions from the community. Please read our [contributing guidelines](CONTRIBUTING.md) to get started. -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

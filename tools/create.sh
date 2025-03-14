#!/bin/bash
# Tool to create a new environment from a configuration folder.

# Function to display help
function display_help() {
    echo "Usage: $0 <config-path> [--use-uv] [--venv-path <path>] [--output-dir <path>]"
    echo
    echo "   <config-path>          Relative path to the configuration folder"
    echo "   --use-uv               Optionally use uv"
    echo "   --venv-path <path>     Path to create the virtual environment (default: $HOME/anemoi_configs/<config-path>/venv/)"
    echo "   --output-dir <path>    Path to copy the configuration (default: $HOME/anemoi_configs/<config-path>)"
    echo "   -h, --help             Display this help message"
    exit 0
}

# Default values
USE_UV=false
VENV_PATH=""
OUTPUT_DIR=""

# Function to parse command line arguments
function parse_arguments() {
    if [ "$#" -lt 1 ]; then
        echo "Error: <config-path> is required"
        display_help
    fi

    CONFIG_PATH="$1"
    REAL_CONFIG_PATH="$(realpath "$(dirname "$0")/../configs/$CONFIG_PATH")"
    if [ ! -d "$REAL_CONFIG_PATH" ]; then
        echo "Error: $CONFIG_PATH is not a valid config path"
        echo "Must be one of the following:"
        ls "$(dirname "$0")/../configs"
        exit 1
    fi
    shift

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --use-uv) USE_UV=true ;;
            --venv-path) VENV_PATH="$2"; shift ;;
            --output-dir) OUTPUT_DIR="$2"; shift ;;
            -h|--help) display_help ;;
            *) echo "Unknown parameter passed: $1"; display_help ;;
        esac
        shift
    done

    # Set default values if not provided
    if [ -z "$VENV_PATH" ]; then
        VENV_PATH="$HOME/anemoi_configs/$CONFIG_PATH/venv"
    fi
    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="$HOME/anemoi_configs/$CONFIG_PATH"
    fi
}

# Function to create virtual environment
function create_virtual_environment() {
    mkdir -p $VENV_PATH
    if $USE_UV; then
        uv venv $VENV_PATH
    else
    if ! command -v python &> /dev/null; then
        echo "Python could not be found. Please install Python and try again."
        exit 1
    fi
        python -m venv $VENV_PATH
    fi
}

# Function to activate virtual environment
function activate_virtual_environment() {
    source "$VENV_PATH/bin/activate"
}

# Function to install packages
function install_packages() {
    if [ -f "$REAL_CONFIG_PATH/environment.txt" ]; then
        if $USE_UV; then
            uv pip install -r "$REAL_CONFIG_PATH/environment.txt"
        else
            pip install -r "$REAL_CONFIG_PATH/environment.txt"
        fi
    else
        echo "Error: environment.txt not found in $CONFIG_PATH"
        deactivate
        exit 1
    fi
}

# Function to copy configuration
function copy_configuration() {
    mkdir -p $OUTPUT_DIR
    cp -r "$REAL_CONFIG_PATH"/dataset $OUTPUT_DIR
    cp -r "$REAL_CONFIG_PATH"/training $OUTPUT_DIR
}

# Main script execution
parse_arguments "$@"
create_virtual_environment
activate_virtual_environment
install_packages
copy_configuration
deactivate

echo "Environment setup complete."
echo "Configuration copied to: $OUTPUT_DIR"
echo "Activate the virtual environment using: source $VENV_PATH/bin/activate"

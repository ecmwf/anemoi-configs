#!/bin/bash
# Tool to create a new environment from a configuration folder.

# Function to display help
function display_help() {
    echo "Usage: $0 <config-path> [--use-uv] [--venv-path <path>] [--output-path <path>] [--install-args <args>]"
    echo
    echo "   <config-path>          Relative path to the configuration folder"
    echo "   --use-uv               Optionally use uv"
    echo "   --venv-path <path>     Path to create the virtual environment (default: $HOME/anemoi_configs/<config-path>/.venv/)"
    echo "   --output-path <path>   Path to copy the configuration (default: $HOME/anemoi_configs/<config-path>)"
    echo "   --install-args <args>     Additional arguments to pass to the venv creation"
    echo "   -h, --help             Display this help message"
    exit 0
}

# Default values
USE_UV=false
VENV_PATH=""
OUTPUT_DIR=""
INSTALL_ARGS=""

# Function to parse command line arguments
function parse_arguments() {
    if [ "$#" -lt 1 ]; then
        echo "Error: <config-path> is required"
        display_help
    fi

    # Check for help flag before processing other arguments
    for arg in "$@"; do
        if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
            display_help
        fi
    done

    CONFIG_PATH="$1"
    
    REAL_CONFIG_PATH="$(realpath "$(dirname "$0")/../$CONFIG_PATH")"
    if [ ! -d "$REAL_CONFIG_PATH" ]; then
        echo "Error: $CONFIG_PATH is not a valid config path"
        echo "Must be one of the following:"
        find "$(dirname "$0")/../configs/" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;

        exit 1
    fi
    CONFIG_PATH=$(basename "$(dirname "$CONFIG_PATH")")/$(basename "$CONFIG_PATH")

    shift

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --use-uv) USE_UV=true ;;
            --venv-path) VENV_PATH="$2"; shift ;;
            --output-path) OUTPUT_DIR="$2"; shift ;;
            --install-args) INSTALL_ARGS="$2"; shift ;;
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
    mkdir -p $OUTPUT_DIR
    mkdir -p $(dirname $VENV_PATH)

    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "Virtual environment already exists at $VENV_PATH"
        return
    fi
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
            uv pip install -r "$REAL_CONFIG_PATH/environment.txt" $INSTALL_ARGS
        else
            pip install -r "$REAL_CONFIG_PATH/environment.txt" $INSTALL_ARGS
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
    cp -r "$REAL_CONFIG_PATH"/*.md $OUTPUT_DIR
    if [ ! -L "$OUTPUT_DIR/venv" ]; then
        ln -s $VENV_PATH "$OUTPUT_DIR/venv"
    fi
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

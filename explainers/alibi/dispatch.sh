#!/bin/bash

set -e

usage() {
    echo "Usage: docker run <image> <command>"
    echo "Available commands: help, cfproto"
}

if [ -z "$1" ]; then
    echo "ERROR: No command specified."
    usage
    exit 1
fi

COMMAND=$1
shift 

case "$COMMAND" in
    "help")
        usage
        ;;

    "cfproto")
	    celery -A explainers.alibi.cfproto worker -l info -n alibi_cfproto@%h -Q alibi_cfproto,celery
        ;;

    *)
        echo "ERROR: Unknown command '$COMMAND'."
        usage
        exit 1
        ;;
esac
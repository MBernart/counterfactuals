#!/bin/bash

set -e

usage() {
    echo "Usage: docker run <image> <command>"
    echo "Available commands: help, cfproto, cfrl"
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
	    celery -A explainers.alibi.cfproto worker -l info -n alibi_cfproto_worker@%h -Q alibi_cfproto,celery -P solo
        ;;

    "cfrl")
	    celery -A explainers.alibi.cfrl worker -l info -n alibi_cfrl_worker@%h -Q alibi_cfrl,celery -P solo
        ;;

    *)
        echo "ERROR: Unknown command '$COMMAND'."
        usage
        exit 1
        ;;
esac
#!/bin/bash

set -e

usage() {
    echo "Usage: docker run <image> <command>"
    echo "Available commands: help, growing_spheres, wachter"
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

    "growing_spheres")
	    celery -A explainers.native.growing_spheres worker -l info -n growing_spheres_worker@%h -Q growing_spheres,celery
        ;;
    "wachter")
        celery -A explainers.native.wachter worker -l info -n wachter_worker@%h -Q wachter,celery
        ;;

    *)
        echo "ERROR: Unknown command '$COMMAND'."
        usage
        exit 1
        ;;
esac

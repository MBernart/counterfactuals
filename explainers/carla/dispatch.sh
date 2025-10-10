#!/bin/bash

set -e

usage() {
    echo "Usage: docker run <image> <command>"
    echo "Available commands: help, actionable_recourse"
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

    "actionable_recourse")
	    celery -A explainers.carla.actionable_recourse worker -l info -n carla_actionable_recourse_worker@%h -Q carla_actionable_recourse,celery
        ;;
    "dice")
        celery -A explainers.carla.dice worker -l info -n carla_dice_worker@%h -Q carla_dice,celery
        ;;

    *)
        echo "ERROR: Unknown command '$COMMAND'."
        usage
        exit 1
        ;;
esac

#!/bin/bash

# Usage: cd to root folder of the project, execute scripts/local_install.sh

# TODO add mitsuba3 build commands here for ext folder?

source ext/mitsuba3/build/setpath.sh
python3 -m pip install .
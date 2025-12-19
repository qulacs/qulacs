#!/bin/bash

salloc -I3600 --qos=a64_interactive --mem-per-cpu=625 -p a64 -c 8 -t 02:00:00 srun -c 8 --pty --preserve-env $(getent passwd $LOGNAME | rev | cut -d':' -f 1 | rev) -i
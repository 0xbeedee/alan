#!/bin/bash

# count the lines of python code in the project (run from project root!)
find . -type f -name "*.py" -not -path "*/\.*" -not -path "*/lib/*" -not -path "*/etc/*" -not -path "*/share/*" -not -path "*/include/*" -not -path "*/test/*" | xargs wc -l | sort -nr 

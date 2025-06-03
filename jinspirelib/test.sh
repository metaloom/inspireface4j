#!/bin/bash

rm draw_detected.jpg
./build.sh
./build/sample/BoxSample
display draw_detected.jpg 
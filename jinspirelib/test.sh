#!/bin/bash

rm draw_detected.jpg
rm build/sample/BoxSample
./build.sh
./build/sample/BoxSample
display draw_detected.jpg 
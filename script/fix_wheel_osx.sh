#!/bin/sh

# Reference: https://github.com/matthew-brett/delocate/issues/72#issuecomment-623070388

set -ex

wheel_path=$1
wheel_filename=$(basename $wheel_path)
dest_dir=$2
temp_dir=$(mktemp -d)

cd $temp_dir
unzip $wheel_path
delocate-path -L qulacs.dylibs .
zip -r $dest_dir/$wheel_filename *
cd -
rm -rf $temp_dir

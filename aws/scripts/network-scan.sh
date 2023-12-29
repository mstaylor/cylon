#!/bin/bash

# Directory to store the snapshot
snapshot_dir="/cylon/sys/class/net"

# Create the snapshot directory
mkdir -p "$snapshot_dir"

# Loop over each network interface in /sys/class/net
for interface in /sys/class/net/*; do
    iface_name=$(basename "$interface")
    iface_dir="$snapshot_dir/$iface_name"

    # Create a directory for each interface
    mkdir -p "$iface_dir"

    # Copy the data from each file in the interface's directory
    for file in "$interface/"*; do
        if [ -f "$file" ]; then
            cp "$file" "$iface_dir/"
        fi
    done
done

echo "Snapshot saved in $snapshot_dir"
nm target/x86_64-D.TinctoriusAzureus/debug/dendrobates_tinctoreus_azureus | grep -i " T " | awk '{ print $1" "$3 }' > kernel.sym

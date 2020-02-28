#!/usr/bin/awk -f

BEGIN {
    addr = 0
    low_hit = 0
    high_hit = 0
    low_miss = 0
    high_miss = 0
    #print "DEBUG BEGIN"
}

/Calibration for 0x/ {
    addr = $3
    low_hit = 0
    high_hit = 0
    low_miss = 0
    high_miss = 0
    #print "DEBUG addr " addr
    #print "DEBUG " (addr != 0)
}

/:/ && addr != 0 {
    cycle = $1 + 0
    hit = $2 + 0
    miss = $3 + 0
    #print "DEBUG " cycle " " hit " " miss
    if (hit > 1) {
        high_hit = cycle
    } else if (high_hit == 0) {
        low_hit = cycle
    }
    if (miss > 1) {
        high_miss = cycle
    } else if (high_miss == 0) {
        low_miss = cycle
    }
    #print "DEBUG " low_miss ", " high_miss ", " low_hit ", " high_hit
}

addr != 0 && /Calibration done./ {
    print "" addr  ", " low_miss ", " high_miss ", " low_hit ", " high_hit
    addr = 0
}


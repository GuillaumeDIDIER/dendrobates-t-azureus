BEGIN {
    i = 0
}
{
    start = $0
    getline
    end = $0
    system("awk '$0 == \""start"\",$0 == \""end"\"' < log.txt | awk "\
              "'BEGIN { print_addr = ("i" == 0); if(print_addr) {print \"Addr, hmin"i", hmax"i", hmed"i", mmin"i", mmax"i", mmed"i"\" } else { print \"hmin"i", hmax"i", hmed"i", mmin"i", mmax"i", mmed"i"\" } }\n"\
              "/Calibration for/ { addr = $3 }\n /Hits/ { hmin = $3; hmax = $5; hmed = $7}\n /Miss/ {mmin = $3; mmax = $5; mmed = $7}\n /Calibration done/ { if(print_addr) { print addr\", \"hmin\", \"hmax\", \"hmed\", \"mmin\", \"mmax\", \"mmed } else { print hmin\", \"hmax\", \"hmed\", \"mmin\", \"mmax\", \"mmed } }'> "i".csv")
    i = i + 1
}

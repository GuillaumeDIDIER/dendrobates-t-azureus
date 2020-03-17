BEGIN {
    i = 0
}
{
    start = $0
    getline
    end = $0
    system("awk '$0 == \""start"\",$0 == \""end"\"' < log.txt | awk '/Calibration for/ { addr = $3 }\n /Hits/ { hmin = $3; hmax = $5; hmed = $7}\n /Miss/ {mmin = $3; mmax = $5; mmed = $7}\n /Calibration done/ {print addr\", \"hmin\", \"hmax\", \"hmed\", \"mmin\", \"mmax\", \"mmed}'> "i".csv")
    i = i + 1
}

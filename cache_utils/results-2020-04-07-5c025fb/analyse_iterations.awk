BEGIN {
    i = 0
}
{
    start = $0
    getline
    end = $0
    if (i > 0) {
        cut = "-f 3- -d,"
    } else {
        cut = "-b 5-"
    }
    system("awk '$0 == \""start"\",$0 == \""end"\"' < log.txt | grep \"CSV:\" | cut " cut " > "i".csv")
    i = i + 1
}

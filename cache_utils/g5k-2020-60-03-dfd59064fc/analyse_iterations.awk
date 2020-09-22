BEGIN {
    i = 0
    print(logname)
}
{
    start = $0
    getline
    end = $0
    if (i == 0) {
        # generate header
        system("bzcat < "logname".txt.bz2 | awk '$0 == \""start"\",$0 == \""end"\"' | grep \"RESULT:\" | head -n 1 | cut -b 8-  | awk '{print \"core,\" $0}'")
    }
    cut = "cut -b 8- | tail -n +2"

    system("bzcat < "logname".txt.bz2 | awk '$0 == \""start"\",$0 == \""end"\"' | grep \"RESULT:\" | " cut " | awk '{print  \""i",\" $0}'")
    i = i + 1
}

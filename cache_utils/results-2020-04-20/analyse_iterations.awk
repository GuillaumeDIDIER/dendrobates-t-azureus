BEGIN {
    i = 0
}
{
    start = $0
    getline
    end = $0
    if (i == 0) {
        # generate header
        system("awk '$0 == \""start"\",$0 == \""end"\"' < log.txt | grep \"RESULT:\" | head -n 1 | cut -b 8-  | awk '{print \"core,\" $0}'> results.csv")
    }
    cut = "cut -b 8- | tail -n +2"

    system("awk '$0 == \""start"\",$0 == \""end"\"' < log.txt | grep \"RESULT:\" | " cut " | awk '{print  \""i",\" $0}'>> results.csv")
    i = i + 1
}

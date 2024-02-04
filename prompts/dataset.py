import csv
with open('./fine-tune-libjpeg-mutate-llm.csv', "w", encoding="utf-8") as csvfilellm:
    with open('./fine-tune-libjpeg-mutate.csv', "r", newline='', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        a=0
        for row in spamreader:
            print(row[0])
            # print(row[0][0])
            a+=1
            if a>1:
                break
import sys

if __name__ =="__main__":
    DATA_FILE_PATH = sys.argv[1]
    # print(DATA_FILE_PATH)


# f = open("./words.txt")
    f = open(DATA_FILE_PATH)
    sol = {}
    id_num=0
    file = f.readline().rstrip()
    words=file.split(" ")

    for word in words:
        if word not in sol:
            sol[word]={
                'id':id_num,
                'count':1
            }
            id_num+=1
        else:
            sol[word]['count']+=1

    with open('Q1.txt','a') as the_file:
        for key,val in sol.items():
            the_file.write(key+' '+str(val['id'])+' '+str(val['count'])+'\n')
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
import csv

def main():
    dirs = [join("./data", d) for d in listdir("./data") \
            if isdir(join("./data", d)) and d != ".git"]
    for n in range(len(dirs)):
        files = [join(dirs[n], f) for f in listdir(dirs[n]) \
                 if isfile(join(dirs[n], f))]
        files_use = [f for f in listdir(dirs[n]) \
                     if isfile(join(dirs[n], f))]
        for m in range(len(files)):
            f = files[m]    
            temp = files_use[m].split('-')
            filename = temp[1]
            person = temp[0]

            # after this process, each list is 144*144*3 + 1
            try:
                im = Image.open(f)
                # width is 64, height is 64
                width, height = im.size
                pix = im.getdata()
    
                myfile = open("./files_padding/" + filename, 'a+')
                #print("open {0}".format(filename))
                with myfile:
                    writer = csv.writer(myfile)
                    mylist = [] 
                    # process the original picture, padding 80*3 zeros after each line
                    for i in range(height):
                        for j in range(width):
                            for m in range(len(pix[i*width+j])):
                                mylist.append(pix[i*width+j][m])
                        for n in range(80):
                            mylist.append(0)
                            mylist.append(0)
                            mylist.append(0)
                    # padding 80 lines, each line contains 144*3 zeros
                    for i in range(80):
                        for j in range(144):
                            mylist.append(0)
                            mylist.append(0)
                            mylist.append(0)
                    # append person id
                    mylist.append(person)
                    writer.writerow(mylist) 
            except OSError:
                print("1 mistake")
    
if __name__=='__main__':
    main()

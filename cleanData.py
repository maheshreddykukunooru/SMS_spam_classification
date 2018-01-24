import re

f = open('SMSSpamCollection','r')
fHam = open('cleanedHam','w')
fSpam = open('cleanedSpam','w')
textHam = ''
textSpam=''
i=0
for line in f:
    finalLine = line.rstrip()
    # print(finalLine)
    finalLine = re.sub(' +',' ',re.sub(r'[^A-Za-z]', ' ', finalLine)).lower()
    if finalLine.split(' ')[0]=='ham':
        textHam = textHam + finalLine[4:] + '\n'
    elif finalLine.split(' ')[0]=='spam':
        textSpam = textSpam + finalLine[5:] + '\n'

fHam.write(textHam)
fSpam.write(textSpam)

f.close()
fHam.close()
fSpam.close()

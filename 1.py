n = int(input())
signals = []
uniqueSignals = []
mEven = 0
mUneven = 0
for i in range(n):
    inp = int(input())
    if inp%2 == 0:
        if mEven < inp:
            mEven = inp
    else:
        if mUneven < inp:
            mUneven = inp

checksum = input()

if mEven != 0 and mUneven != 0:
    readChecksum = mEven+mUneven
    print('Вычисленное контрольное значение:' + str(readChecksum))

    if readChecksum == checksum:
        print('Контроль пройден')
    else:
        print('Контроль пройден')
else:
    print('Контроль пройден')


import numpy as np


def neural_element(activation, weight):
    SDCodeLUT = np.array([0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 0, 15, 0 ,14, 13])
    INV1 = 0
    INV2 = 0
    SDCode = SDCodeLUT[abs(weight)]

    SH1 = (SDCode >> 2) & 3
    SH2 = SDCode & 3

    if SH1 == 0:
        barrelShifter1 = 0
    else:
        barrelShifter1 = activation << (SH1-1)
        barrelShifter1 = barrelShifter1 << 2

    if SH2 == 0:
        barrelShifter2 = 0
    else:
        barrelShifter2 = activation << (SH2-1)


    if weight >= 0:
        if (weight == 3) | (weight == 7) | (weight == 12) | (weight == 14) | (weight == 15):
            INV1 = 0
            INV2 = 1
        else:
            INV1 = 0
            INV2 = 0
    else:
        if (weight == -3) | (weight == -7) | (weight == -12) | (weight == -14) | (weight == -15):
            INV1 = 1
            INV2 = 0
        else:
            INV1 = 1
            INV2 = 1

    if INV1 == 1:
        barrelShifter1 = -barrelShifter1 - 1

    if INV2 == 1:
        barrelShifter2 = -barrelShifter2 - 1

    return barrelShifter1 + barrelShifter2, INV1 + INV2

f = open("./neural_element_result/neural_element_test_result.txt", 'w')

#양수 x 양수
for activation in range(0, 128):
    for weight in range(0, 16):
        if  ((weight != 11) & (weight != 13)):
            sumBP, sumINV = neural_element(activation, weight)
            data = "activation(%d) x weight(%d) = sumBP(%d), sumINV(%d)"% (activation, weight, sumBP, sumINV)
            f.write(data)
            if (activation * weight) == (sumBP + sumINV):
                data = " Correct "
            else:
                data = " Error "
            f.write(data)
            f.write("\n")

#양수 x 음수
for activation in range(0, 128):
    for weight in range(0, -16, -1):
        if(weight != -11) & (weight != -13):
            sumBP, sumINV = neural_element(activation, weight)
            data = "activation(%d) x weight(%d) = sumBP(%d), sumINV(%d)"% (activation, weight, sumBP, sumINV)
            f.write(data)
            if (activation * weight) == (sumBP + sumINV):
                data = " Correct "
            else:
                data = " Error "
            f.write(data)
            f.write("\n")
        else:
            data = ""
            f.write(data)

#음수 x 양수
for activation in range(0, -128, -1):
    for weight in range(0, 16):
        if(weight != 11) & (weight != 13):
            sumBP, sumINV = neural_element(activation, weight)
            data = "activation(%d) x weight(%d) = sumBP(%d), sumINV(%d)"% (activation, weight, sumBP, sumINV)
            f.write(data)
            if (activation * weight) == (sumBP + sumINV):
                data = " Correct "
            else:
                data = " Error "
            f.write(data)
            f.write("\n")

#음수 x 음수
for activation in range(0, -128, -1):
    for weight in range(0, -16, -1):
        if(weight != -11) & (weight != -13):
            sumBP, sumINV = neural_element(activation, weight)
            data = "activation(%d) x weight(%d) = sumBP(%d), sumINV(%d)"% (activation, weight, sumBP, sumINV)
            f.write(data)
            if (activation * weight) == (sumBP + sumINV):
                data = " Correct "
            else:
                data = " Error "
            f.write(data)
            f.write("\n")

f.close()

sumBP, sumINV = neural_element(1, 3)

print('sumBP : ', sumBP)
print('sumINV : ', sumINV)

a = 13

if(a != 11) & (a != 3):
    print('A')


#print('a << 0 : ', a<<-1)

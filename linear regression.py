import numpy as np
import matplotlib.pyplot as plt
import math

def estimareFunctie(x,b0,b1):
    # y = b0 + b1*x
    return x *b1 + b0

def calculEroare(nr_ore_studiu,nota_obtinuta,b0,b1):
    return round(sum(list(map(lambda x,y: math.pow(y-(b0+b1*x),2),nr_ore_studiu,nota_obtinuta)))/2,2)

def gradientDescendent(nr_ore_studiu,nota_obtinuta):
    bb0 = 2
    bb1 = 3

    delta = 0.03
    alfa = 0.04

    for i in range(20):
        SSE = calculEroare(nr_ore_studiu, nota_obtinuta, bb0, bb1)
        print("SSE: " + str(SSE))

        derivata1 = sum(list(map(lambda x, y: -(y - (bb0 + bb1 * x)), nr_ore_studiu, nota_obtinuta)))
        derivata2 = sum(list(map(lambda x, y: -(y - (bb0 + bb1 * x)) * x, nr_ore_studiu, nota_obtinuta)))

        print("derivata1: " + str(derivata1))
        print("derivata2 " + str(derivata2))
        #
        # aprox_der1 = round((SSE * (bb0 + delta) - SSE * bb0) / delta, 2)
        # aprox_der2 = round((SSE * (bb1 + delta) - SSE * bb1) / delta, 2)

        bb0 = bb0 - alfa * derivata1
        bb1 = bb1 - alfa * derivata2

        print("b0:" + str(bb0))
        print("b1: " + str(bb1))

        SSE_nou = calculEroare(nr_ore_studiu, nota_obtinuta, bb0, bb1)
        print("SSEnou: " + str(SSE_nou))





nr_ore_studiu = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25 ,3.5 ,4, 4.25 ,4.5 ,4.75 ,5 ,5.5] #variabila independenta
nota_obtinuta = [4 ,3, 2.5, 1, 2, 3.5, 6, 4, 7, 1.5, 5, 2.5, 5.5, 3, 8, 7, 7.5, 6, 8.5, 9.5] #variabila dependenta
N = len(nota_obtinuta)
media_independ = round(sum(nr_ore_studiu)/N,2)
media_depend = round(sum(nota_obtinuta)/N,2)
print(media_independ)
print(media_depend)

x = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25 ,3.5 ,4, 4.25 ,4.5 ,4.75 ,5 ,5.5])
y = np.array([4 ,3, 2.5, 1, 2, 3.5, 6, 4, 7, 1.5, 5, 2.5, 5.5, 3, 8, 7, 7.5, 6, 8.5, 9.5])


DifX = list(map(lambda x: round(x - media_independ,2), nr_ore_studiu))
print(DifX)

DifY = list(map(lambda x: round(x - media_depend,2), nota_obtinuta))
print(DifY)

MulXY = list(map(lambda x,y: round(x*y,2),DifY,DifX))
print(MulXY)

MulXX = list(map(lambda x: round(x*x,2),DifX))
print(MulXX)

b1 = round(sum(MulXY)/(sum(MulXX)),2)
print("b1: " + str(b1))

b0 = round(media_depend - b1*media_independ,2)
print("b0: "+ str(b0))


sol_analitica = list(map(lambda x:round(estimareFunctie(x,b0,b1),2),nr_ore_studiu))
print("solutia:" + str(sol_analitica))

mse_calc = list(map(lambda x,y:(x-y)*(x-y),nota_obtinuta,sol_analitica))
MSE = round(sum(mse_calc)/N,2)
print("MSE: " + str(MSE))

gradientDescendent(nr_ore_studiu,nota_obtinuta)


plt.scatter(x, y)
xplot = np.arange(min(x), max(x), 0.01)
plt.plot(xplot, estimareFunctie(xplot,b0,b1))
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

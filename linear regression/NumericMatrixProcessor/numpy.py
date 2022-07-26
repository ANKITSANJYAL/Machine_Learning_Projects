from numpy import *

def matrix_input():
    
    number_of_rows = int(input("no of rows"))
    number_of_columns= int(input("no of col"))
    matrix_ip = zeros((number_of_rows,number_of_columns) , dtype = int)
    len_of_matrix = len(matrix_ip)
   
    for i in range(len_of_matrix):
        for j in range(len(matrix_ip[i])):
            x = input("enter the element")
            matrix_ip[i][j] = x
    print(matrix_ip)


oper = input("choose the operation to do: + for addition , * for multiplication , - for subtraction and / for division")
if oper == "+":
    which = input("1 for scalar addition and 2 for matrix addition")
    if which == 1:
        matrix1 = input("enter the matrix1")
        matrix_input()
        withwhat = input("enter the number to be added")
        result = numpy.array(matrix1) + withwhat
        print(result)
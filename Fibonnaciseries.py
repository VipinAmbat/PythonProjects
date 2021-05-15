# Python program to print fibonnaci series
first_value=0
second_value=1
count=0
sum=0
x=int(input("enter the no for the fibonnaci series to appear" ))
print("Fibonnaci series are as follows")
while count<x:

                count+=1 #count will get addeded by 1
                sum=first_value+second_value
                first_value=second_value
                second_value=sum
                print(sum)
            

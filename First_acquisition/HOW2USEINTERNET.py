from INTERNET import INTERNET

interNET = INTERNET(sample_rate = 250,control_freq = 10 ,folder_path='model42')

#lsl



#in loop 250HZ
data = [1,2]
result = interNET.predict(data)
# 2 idle
# 0 backward
# 1 forward
# 3 left
# 4 right
import numpy as np

def cal_roc_tpr(x,y,tpr):

#############X is the in-distribution array
############## y i s hte OOD array 

    counter1_max = np.size(x)
    counter2_max = np.size(y)
    terminal = counter1_max+counter2_max
    counter1 = counter1_max-1
    counter2 = counter2_max-1
    x= np.sort(x)
    y = np.sort(y)
    small_count = np.zeros([counter1_max])
    other_count = np.zeros([counter2_max])
    for i in range(terminal):
        if ((counter1>=0) & (counter2>=0)):
            specimen1 = x[counter1]
            specimen2 = y[counter2] 
            if specimen1<specimen2:
                counter2 -= 1
                small_count[counter1] += 1
                if counter2 >=0:
                    other_count[counter2] = other_count[counter2 +1]
            else:
                counter1 -= 1
                other_count[counter2] += 1
                if counter1 >=0:
                    small_count[counter1] = small_count[counter1 + 1]
          
    if counter1>=0:
       for i in range(counter1):
           small_count[i] = small_count[counter1]
    if counter2>=0:
       for i in range(counter2):
           other_count[i] = other_count[counter2] 
    detection_acc = np.zeros([counter1_max])
    small_count = np.sort(small_count)
    for i in range(counter1_max):
        detection_acc[i] = (i*1.0/counter1_max + 1.0*(counter2_max-small_count[i])/counter2_max) * 0.5 
    detection_accuracy = 1.0-np.min(detection_acc)
    aupr_in_prec = np.zeros([counter1_max]) # TP/(TP+FP), recall is just a hinerance
    for i in range(counter1_max):
        aupr_in_prec[i] = (counter1_max-i)*1.0/(counter2_max-small_count[i]+counter1_max-i)
    aupr_in = np.mean(aupr_in_prec)
    aupr_out_prec = np.zeros([counter2_max]) # TP/(TP+FP), but in OOD perspective
    other_count = np.sort(other_count)
    for i in range(counter2_max): # In here, we set threshold bigger as positive, so we do the opposite 
        aupr_out_prec[i] = 1.0*(i+1)/(i+1+other_count[i])
    aupr_out = np.mean(aupr_out_prec)
    return np.mean(small_count)*1.0/counter2_max,np.sort(small_count)[int((1.0-tpr)*counter1_max)]*1.0/counter2_max,detection_accuracy,aupr_in,aupr_out


#brief check
#a= np.array([1,3,5,7,9])

#b= np.array([0.8,2.2,4.6,5.3,6.8,7.2,8.5])+4.0
#print(np.size(b))
#c,d,e,f,g = cal_roc_tpr(a,b,0.95)
#print(c)
#print(d) 
#print(e)        
#print(f)
#print(g)


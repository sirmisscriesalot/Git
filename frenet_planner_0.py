import numpy as np
import matplotlib.pyplot as plt 
import cv2 
import math

#the color code for red seems to be #ED1C24
def Spline(input_array):
    h = []
    b = []
    v = []
    u = []
    z = []
    z.append(0)

    for i in range(len(input_array)-1):
        h.append(input_array[i+1][0]-input_array[i][0])
        b.append((input_array[i+1][1]-input_array[i][1])/h[i])
    
    for j in range(1,len(input_array)-1):
        v.append(2*(h[j-1]+h[j]))
        u.append(6*(b[j]-b[j-1]))

    req_mat = [[0 for ii in range(1,len(input_array)-1)] for jj in range(1,len(input_array)-1)]
    #not quite sure if there's a mistake in the paper I am using 
    #well lets see
    for iii in range(len(input_array)-2):
        for jjj in range(len(input_array)-2):
            if iii==jjj:
                req_mat[iii][jjj] = v[iii]
            elif jjj==iii+1:
                req_mat[iii][jjj] = h[jjj]
            elif iii==jjj+1:
                req_mat[iii][jjj] = h[iii]

    numpy_mat = np.array(req_mat)
    # print(numpy_mat)
    numpy_val = np.array(u)
    numpy_mat_inv = np.linalg.inv(numpy_mat)
    answer = np.matmul(numpy_mat_inv,numpy_val)
    answer_2 = answer.tolist()
    z = z + answer_2 + [0]
    # print(z)
    coefficients = []
    for m in range(len(input_array)-1):
        D = input_array[m][1]
        C = (-h[m]*z[m+1]/6) - (h[m]*z[m]/3) + ((input_array[m+1][1]-input_array[m][1])/h[m])
        B = z[m]/2
        A = (z[m+1]-z[m])/(6*h[m])
        coefficients.append([A,B,C,D])
    # print(coefficients)

    return coefficients

def poly5(initial_condition,final_condition):
    ix = initial_condition
    fx = final_condition
    #for now hardcode this 
    #write the functions later on
    M = []
    M.append([ix[3]**5,ix[3]**4,ix[3]**3,ix[3]**2,ix[3],1])
    M.append([5*(ix[3]**4),4*(ix[3]**3),3*(ix[3]**2),2*ix[3],1,0])
    M.append([20*(ix[3]**3),12*(ix[3]**2),6*(ix[3]),2,0,0])
    M.append([fx[3]**5,fx[3]**4,fx[3]**3,fx[3]**2,fx[3],1])
    M.append([5*(fx[3]**4),4*(fx[3]**3),3*(fx[3]**2),2*fx[3],1,0])
    M.append([20*(fx[3]**3),12*(fx[3]**2),6*(fx[3]),2,0,0])

    mat_M = np.array(M)
    mat_D = np.array([ix[0],ix[1],ix[2],fx[0],fx[1],fx[2]])
    mat_M_inv = np.linalg.inv(M)
    mat_coeff = np.matmul(mat_M_inv,mat_D)
    coeff = mat_coeff.tolist()

    
    print(coeff)
    return coeff
    
def collide(obs_img1,x_co,y_co):

    idx = 0
    for i in x_co:
        x_px = int(round(i))
        y_px = int(round(y_co[idx]))

        for j in range(y_px-30,y_px + 30):
            # if j<0:
            #     j = 0
            # if j>588:
            #     j = 588
            try:
                if obs_img1[x_px][j]==1:
                    return True
            except IndexError:
                return True

        idx += 1

    return False

def choose_path(obs,s,t,n):
#initially we check if the spline itself is traversable 
    x = np.arange(t[n][0],t[n+1][0],0.2)
    x_1 = x
    y = []
    for i in x:
        y.append(s[n][0]*((i-t[n][0])**3) + s[n][1]*((i-t[n][0])**2) + s[n][2]*(i-t[n][0]) + s[n][3])
    y_1 = y 
    v = 0.2 
    a = 0.2

    while collide(obs,x_1,y_1):
        x_1 = []
        y_1 = []
        x_poly = poly5([0,v,a,t[n][0]], [0,0,0,t[n+1][0]])
        idx = 0
        for i in x:
            d_1 = x_poly[0]*(i**5) + x_poly[1]*(i**4) + x_poly[2]*(i**3) + x_poly[3]*(i**2) + x_poly[4]*(i) + x_poly[5]
            angle = math.atan(-1/(3*s[n][0]*(i**2)+2*s[n][1]*i + s[n][2]))
            x_1.append(i + d_1 * math.cos(angle))
            y_1.append(y[idx] + d_1 * math.cos(angle))
            idx += 1


        v += 0.2
        a += 0.2
        
        plt.plot(x_1,y_1,"g")

        if v>2:
            break
    
    v1 = -0.2
    a1 = -0.2

    while collide(obs,x_1,y_1):
        x_1 = []
        y_1 = []
        x_poly = poly5([0,v1,a1,t[n][0]], [0,0,0,t[n+1][0]])
        idx = 0
        for i in x:
            d_1 = x_poly[0]*(i**5) + x_poly[1]*(i**4) + x_poly[2]*(i**3) + x_poly[3]*(i**2) + x_poly[4]*(i) + x_poly[5]
            angle = math.atan(-1/(3*s[n][0]*(i**2)+2*s[n][1]*i + s[n][2]))
            x_1.append(i + d_1 * math.cos(angle))
            y_1.append(y[idx] + d_1 * math.cos(angle))
            idx += 1
        

        v1 -= 0.2
        a1 -= 0.2

        plt.plot(x_1,y_1,"g")

        if v1<-2:
            break

    plt.plot(x_1,y_1,"r")

    
    
    



img = cv2.imread(r'C:\Users\himad\OneDrive\NokiaXpress\Public\Pictures\task2_test.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv,(0,0,0),(10,255,255))
mask2 = cv2.inRange(hsv,(36,0,0),(70,255,255))
target2 = cv2.bitwise_and(img,img,mask=mask2)
target1 = cv2.bitwise_not(target2,target2,mask=mask1)
gray = cv2.cvtColor(target2,cv2.COLOR_BGR2GRAY)
ret,bw_img = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
ret2,bw_obstacle = cv2.threshold(img_gray,240,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(bw_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(target2,contours[1:],-1,(0,0,255),3)

positions = []
positions_x = []
positions_y = []

for c in contours[1:]:
    x,y,w,h = cv2.boundingRect(c)
    positions.append([x,588-y])

positions.sort()
for i in positions:
    positions_x.append(i[0])
    positions_y.append(i[1])
plt.plot(positions_x,positions_y,"ro")

obs = []
py_list = bw_obstacle.tolist()
for l in range(len(py_list)):
    for m in range(len(py_list[l])):
        if py_list[l][m]==255:
            obs.append([m,588-l])

obs_img = [[0 for pos_i in range(len(py_list))] for pos_j in range(len(py_list[0]))]
for l in range(len(py_list)):
    for m in range(len(py_list[l])):
        if py_list[l][m]==255:
            obs_img[m][588-l] = 1
obs_x = []
obs_y = []

for n in obs:
    obs_x.append(n[0])
    obs_y.append(n[1])

plt.plot(obs_x,obs_y,"bo")

test_val = positions
work = Spline(positions)
for i in range(len(test_val)-1):
    x = np.linspace(test_val[i][0],test_val[i+1][0])
    plt.plot(test_val[i][0],test_val[i][1],'ro')
    plt.plot(x,work[i][0]*((x-test_val[i][0])**3) + work[i][1]*((x-test_val[i][0])**2) + work[i][2]*(x-test_val[i][0]) + work[i][3],'g')
    choose_path(obs_img,work,test_val,i)
plt.plot(test_val[-1][0],test_val[-1][1],'ro')



plt.show()
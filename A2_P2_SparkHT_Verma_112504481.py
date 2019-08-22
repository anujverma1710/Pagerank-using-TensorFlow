import numpy as np
import csv
from pprint import pprint
import sys
from pyspark import SparkConf, SparkContext
import math
from scipy import stats

def estimate_coef(n_tuple):
    x=[]
    y=[]
    result=[]
    temp = []
    for i in n_tuple[1]:
        temp.append(i)
        x.append(i[1])  #Frequency
        y.append(i[2])  #mortality rate

    x=np.array(x).astype(np.float)
    y=np.array(y).astype(np.float)
    x=(x-x.mean())/x.std()
    y=(y-y.mean())/y.std()

    xMatrix = np.asmatrix(np.c_[x, np.ones(np.size(x))])
    yMatrix = np.asmatrix(y)
    xMatrixTrans = np.transpose(xMatrix)
    xMatrixTransInverse = np.linalg.pinv(np.dot(xMatrixTrans, xMatrix))
    b = np.dot(xMatrixTransInverse,np.dot(xMatrixTrans,np.transpose(yMatrix)))
    b = np.asarray(b)

    n = np.size(xMatrix)
    t,p = calculate_p_value(n,x,y,b[1][0],b[0][0])
    result.append((n_tuple[0],(b[0][0],p)))
    print(result)
    return result


def calculate_p_value(n,x,y,b_0,b_1):
    rss = 0
    k = np.mean(x)
    for i, j in zip(x, y):
        rss = rss + np.square(j - b_0 - i * b_1)
    df = n -2
    s = rss / df
    z = 0
    for i in x:
        z = z + np.square(i - k)
    denominator = s / z
    dem = math.sqrt(denominator)
    t = b_1 / dem
    p = 2*(stats.t.sf(np.abs(t), df=df))
    return t,p


def calculateMultipleRegression(n_tuple):
    y,x,x1,result = [],[],[],[]
    for i in n_tuple[1]:
        x.append(i[1])             #frequency
        y.append(i[2])             #Mortality
        x1.append(i[3])  #i[3]  Income

    x=np.array(x).astype(np.float)
    x1=np.array(x1).astype(np.float)
    y=np.array(y).astype(np.float)

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    x1 = (x - x1.mean()) / x.std()
    k=np.mean(x)
    v=np.mean(x1)
    f=np.asmatrix(np.c_[x,x1])
    m = np.asmatrix(np.c_[f, np.ones(np.size(x))])
    n= np.asmatrix(y)
    x_tr=np.transpose(m)
    beta=np.dot(x_tr,m)
    beta1=np.linalg.pinv(beta)
    beta2=np.dot(beta1,x_tr)
    beta3=np.dot(beta2,np.transpose(n))
    final_b=np.asarray(beta3)
    rss=0
    for i,j,k in zip(x,y,x1):
        rss=rss+np.square(j-final_b[2][0]-i*final_b[0][0]-k*final_b[1][0])
    df=np.size(x)
    s=rss/df
    z=0
    m=0
    for i in x:
        z=z+np.square(i-k)
    for j in x1:
        m=m+np.square(j-v)

    denominator=s/(z+m)
    dem=math.sqrt(denominator)
    t1=final_b[0][0]/dem
    p=2*(stats.t.sf(np.abs(t1),df=df))
    result.append((n_tuple[0],(final_b[0][0],p)))
    return result

if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    wordDataByUniqueCounty =sys.argv[2]
    heartDiseaseDataByCounty=sys.argv[1]
    print(sys.argv[1])
    print(sys.argv[2])

    # wordDataByUniqueCounty="countyoutcomes.csv"
    # heartDiseaseDataByCounty="test.csv"
    rdd = sc.textFile(heartDiseaseDataByCounty).mapPartitions(lambda line: csv.reader(line))
    rdd1 = sc.textFile(wordDataByUniqueCounty).mapPartitions(lambda line: csv.reader(line))

    newRdd = rdd.map(lambda x:(x[0],(x[1],x[3]))).join(rdd1.map(lambda x:(x[0],(x[23],x[24])))).map(lambda y:(y[1][0][0],(y[0],np.float64(y[1][0][1]),np.float64(y[1][1][1]),np.float64(y[1][1][0]))))

    linearRegressionRdd = newRdd.groupByKey().flatMap(estimate_coef)

    multipleLinearRegressionRdd = newRdd.groupByKey().flatMap(calculateMultipleRegression)

    print("top 20 word positively correlated with heart disease mortality : ")
    pprint(linearRegressionRdd.takeOrdered(20, lambda x: -x[1][0]))
    print("top 20 word negatively correlated with heart disease mortality : ")
    pprint(linearRegressionRdd.takeOrdered(20, lambda x: x[1][0]))
    print("top 20 words positively related to hd mortality, controlling for income. : ")
    pprint(multipleLinearRegressionRdd.takeOrdered(20, lambda y :-y[1][0]))
    print("top 20 words negatively related to hd mortality, controlling for income. : ")
    pprint(multipleLinearRegressionRdd.takeOrdered(20, lambda y : y[1][0]))
    print("Bonferroni Correction for alpha = 0.05 and about 20k no. of distinct words : ")
    alpha = 0.05
    pprint(alpha/23693)  # 23693 : verified by calculation

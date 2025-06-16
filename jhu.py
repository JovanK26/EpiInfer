import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import argparse

countries=["AT_"]
parser = argparse.ArgumentParser()
parser.add_argument('--testdays', type=int, default=20, help='Optional testdays value (default: 10)')
args = parser.parse_args()
testdays=args.testdays #how many days in the future we are predicting
pop=[8913088] # population of locales
#population from each group
pop1=[0.887*8913088]
pop2=[0.076*8913088]
pop3=[0.019*8913088]
pop4=[0.011*8913088]
pop5=[0.004*8913088]
pop6=[0.003*8913088]
days=101 # total days
cnum=1 # number of locales
comps=4 # compartments: S,E,I,R
inc=7 #incubation period
gnum=6 # number of groups
perc=np.zeros((cnum,gnum+1)) # percentage of people from each group
for i in range(cnum):
	perc[i][1]=pop1[i]/pop[i] 
	perc[i][2]=pop2[i]/pop[i]
	perc[i][3]=pop3[i]/pop[i]
	perc[i][4]=pop4[i]/pop[i]
	perc[i][5]=pop5[i]/pop[i]
	perc[i][6]=pop6[i]/pop[i]

cdata=np.zeros((cnum,days,comps,gnum+1)) # cumulative data (4D array where cdata[i][j][k][l] gives the number of people from locale i on day j from compartment k and group l, where group 0 is everyone)
ndata=np.zeros((cnum,days,comps,gnum+1)) # new data (4D array where cdata[i][j][k][l] gives the number of people from locale i on day j from compartment k and group l, where group 0 is everyone)

# daily number of interactions of a person based on their group
nmeet1=3
nmeet2=8
nmeet3=15 
nmeet4=35
nmeet5=75 
nmeet6=100 

gamma=0.1 # recovery rate. It is chosen as the inverse of 10 days, as usually it takes around 10 days to recover from COVID
eta=0.01 # rate of going from recovered to susceptible. It is chosen as the inverse of 100 days, as usually COVID immunity lasts 3-4 months

bestp1=-1
bestp2=-1
bestloss=1000000000000000000

# load data from csv files
def loaddata2():
	for i,cnt in enumerate(countries):
		cfname="Data\\"+cnt+"datacumu.csv"
		nfname="Data\\"+cnt+"datanew.csv"
		f1=open(cfname,'r')
		f2=open(nfname,'r')
		for j,line in enumerate(f1):
			ls=line.split(',')
			for k,num in enumerate(ls):
				cdata[i][j][k][0]=float(num)
				cdata[i][j][k][1]=perc[i][1]*cdata[i][j][k][0]
				cdata[i][j][k][2]=perc[i][2]*cdata[i][j][k][0]
				cdata[i][j][k][3]=perc[i][3]*cdata[i][j][k][0]
				cdata[i][j][k][4]=perc[i][4]*cdata[i][j][k][0]
				cdata[i][j][k][5]=perc[i][5]*cdata[i][j][k][0]
				cdata[i][j][k][6]=perc[i][6]*cdata[i][j][k][0]
		for j,line in enumerate(f2):
			ls=line.split(',')
			for k,num in enumerate(ls):
				ndata[i][j][k][0]=float(num)
				ndata[i][j][k][1]=perc[i][1]*ndata[i][j][k][0]
				ndata[i][j][k][2]=perc[i][2]*ndata[i][j][k][0]
				ndata[i][j][k][3]=perc[i][3]*ndata[i][j][k][0]
				ndata[i][j][k][4]=perc[i][4]*ndata[i][j][k][0]
				ndata[i][j][k][5]=perc[i][5]*ndata[i][j][k][0]
				ndata[i][j][k][6]=perc[i][6]*ndata[i][j][k][0]
		f1.close()
		f2.close()


# EpiInfer-core
def sim(p1,p2):
	global days
	global testdays
	global bestloss
	global bestp1
	global bestp2
	simdata=np.zeros((cnum,days,comps,gnum+1)) # cumulative data
	nsimdata=np.zeros((cnum,days,comps,gnum+1)) # new data
	for i in range(cnum):
		for j in range(inc):
			for comp in range(comps):
				for grp in range(gnum+1):
					simdata[i][j][comp][grp]=cdata[i][j][comp][grp]
					nsimdata[i][j][comp][grp]=ndata[i][j][comp][grp]
	res=np.zeros((cnum,days-testdays))
	for i,cnt in enumerate(countries):
		for t in range(inc,days):
			infsum=0
			infsum1=0
			infsum2=0
			infsum3=0
			infsum4=0
			infsum5=0
			infsum6=0
			limit=days-testdays
			for j in range(1,inc+1):
				if t-j>=0 and t-j>=limit:
					infsum+=nsimdata[i][t-j][2][0]
					infsum1+=nsimdata[i][t-j][2][1]
					infsum2+=nsimdata[i][t-j][2][2]
					infsum3+=nsimdata[i][t-j][2][3]
					infsum4+=nsimdata[i][t-j][2][4]
					infsum5+=nsimdata[i][t-j][2][5]
					infsum6+=nsimdata[i][t-j][2][6]
				elif t-j>=0 and t-j<limit:
					infsum+=ndata[i][t-j][2][0] 
					infsum1+=ndata[i][t-j][2][1]
					infsum2+=ndata[i][t-j][2][2]
					infsum3+=ndata[i][t-j][2][3]
					infsum4+=ndata[i][t-j][2][4]
					infsum5+=ndata[i][t-j][2][5]
					infsum6+=ndata[i][t-j][2][6]
			if t-inc>=limit:
				asymp=pop[i]-simdata[i][t-inc][2][0] 
				asymp1=pop1[i]-simdata[i][t-inc][2][1]
				asymp2=pop2[i]-simdata[i][t-inc][2][2]
				asymp3=pop3[i]-simdata[i][t-inc][2][3]
				asymp4=pop4[i]-simdata[i][t-inc][2][4]
				asymp5=pop5[i]-simdata[i][t-inc][2][5]
				asymp6=pop6[i]-simdata[i][t-inc][2][6]
			else:
				asymp=pop[i]-cdata[i][t-inc][2][0] 
				asymp1=pop1[i]-cdata[i][t-inc][2][1]
				asymp2=pop2[i]-cdata[i][t-inc][2][2]
				asymp3=pop3[i]-cdata[i][t-inc][2][3]
				asymp4=pop4[i]-cdata[i][t-inc][2][4]
				asymp5=pop5[i]-cdata[i][t-inc][2][5]
				asymp6=pop6[i]-cdata[i][t-inc][2][6]
			p_exp=infsum/(p2*asymp) # eq 1
			ptran1=1-(1-p1)**(nmeet1*p_exp) # eq2
			ptran2=1-(1-p1)**(nmeet2*p_exp)
			ptran3=1-(1-p1)**(nmeet3*p_exp)
			ptran4=1-(1-p1)**(nmeet4*p_exp)
			ptran5=1-(1-p1)**(nmeet5*p_exp)
			ptran6=1-(1-p1)**(nmeet6*p_exp)
			asymp_notexp=asymp-infsum/p2-simdata[i][t-inc][3][0] # eq 3
			asymp_notexp1=asymp1-infsum1/p2-simdata[i][t-inc][3][1]
			asymp_notexp2=asymp2-infsum2/p2-simdata[i][t-inc][3][2]
			asymp_notexp3=asymp3-infsum3/p2-simdata[i][t-inc][3][3]
			asymp_notexp4=asymp3-infsum4/p2-simdata[i][t-inc][3][4]
			asymp_notexp5=asymp3-infsum5/p2-simdata[i][t-inc][3][5]
			asymp_notexp6=asymp3-infsum6/p2-simdata[i][t-inc][3][6]
			asymp_notexp1=max(asymp_notexp1,0)
			asymp_notexp2=max(asymp_notexp2,0)
			asymp_notexp3=max(asymp_notexp3,0)
			asymp_notexp4=max(asymp_notexp4,0)
			asymp_notexp5=max(asymp_notexp5,0)
			asymp_notexp6=max(asymp_notexp6,0)
			newexp1=asymp_notexp1*ptran1 # eq4
			newexp2=asymp_notexp2*ptran2
			newexp3=asymp_notexp3*ptran3
			newexp4=asymp_notexp4*ptran4
			newexp5=asymp_notexp5*ptran5
			newexp6=asymp_notexp6*ptran6
			newexp=newexp1+newexp2+newexp3+newexp4+newexp5+newexp6
			if t-inc>=inc:
				nsimdata[i][t-inc][1][0]=newexp
				nsimdata[i][t-inc][1][1]=newexp1
				nsimdata[i][t-inc][1][2]=newexp2
				nsimdata[i][t-inc][1][3]=newexp3
				nsimdata[i][t-inc][1][4]=newexp4
				nsimdata[i][t-inc][1][5]=newexp5
				nsimdata[i][t-inc][1][6]=newexp6
			nsimdata[i][t][2][0]=newexp*p2 # eq 5
			nsimdata[i][t][2][1]=newexp1*p2
			nsimdata[i][t][2][2]=newexp2*p2
			nsimdata[i][t][2][3]=newexp3*p2
			nsimdata[i][t][2][4]=newexp4*p2
			nsimdata[i][t][2][5]=newexp5*p2
			nsimdata[i][t][2][6]=newexp6*p2
			nsimdata[i][t][3][0]=simdata[i][t-1][2][0]*gamma
			nsimdata[i][t][3][1]=simdata[i][t-1][2][1]*gamma
			nsimdata[i][t][3][2]=simdata[i][t-1][2][2]*gamma
			nsimdata[i][t][3][3]=simdata[i][t-1][2][3]*gamma
			nsimdata[i][t][3][4]=simdata[i][t-1][2][4]*gamma
			nsimdata[i][t][3][5]=simdata[i][t-1][2][5]*gamma
			nsimdata[i][t][3][6]=simdata[i][t-1][2][6]*gamma
			nsimdata[i][t][0][0]=simdata[i][t-1][3][0]*eta # lose imunity
			nsimdata[i][t][0][1]=simdata[i][t-1][3][1]*eta # lose imunity
			nsimdata[i][t][0][2]=simdata[i][t-1][3][2]*eta # lose imunity
			nsimdata[i][t][0][3]=simdata[i][t-1][3][3]*eta # lose imunity
			nsimdata[i][t][0][4]=simdata[i][t-1][3][4]*eta # lose imunity
			nsimdata[i][t][0][5]=simdata[i][t-1][3][5]*eta # lose imunity
			nsimdata[i][t][0][6]=simdata[i][t-1][3][6]*eta # lose imunity
			simdata[i][t][3][0]=simdata[i][t-1][3][0]-nsimdata[i][t][0][0]+nsimdata[i][t][3][0] # total rec
			simdata[i][t][3][1]=simdata[i][t-1][3][1]-nsimdata[i][t][0][1]+nsimdata[i][t][3][1] # total rec
			simdata[i][t][3][2]=simdata[i][t-1][3][2]-nsimdata[i][t][0][2]+nsimdata[i][t][3][2] # total rec
			simdata[i][t][3][3]=simdata[i][t-1][3][3]-nsimdata[i][t][0][3]+nsimdata[i][t][3][3] # total rec
			simdata[i][t][3][4]=simdata[i][t-1][3][4]-nsimdata[i][t][0][4]+nsimdata[i][t][3][4] # total rec
			simdata[i][t][3][5]=simdata[i][t-1][3][5]-nsimdata[i][t][0][5]+nsimdata[i][t][3][5] # total rec
			simdata[i][t][3][6]=simdata[i][t-1][3][6]-nsimdata[i][t][0][6]+nsimdata[i][t][3][6] # total rec
			simdata[i][t][2][0]=simdata[i][t-1][2][0]+nsimdata[i][t][2][0]-nsimdata[i][t][3][0]
			simdata[i][t][2][1]=simdata[i][t-1][2][1]+nsimdata[i][t][2][1]-nsimdata[i][t][3][1]
			simdata[i][t][2][2]=simdata[i][t-1][2][2]+nsimdata[i][t][2][2]-nsimdata[i][t][3][2]
			simdata[i][t][2][3]=simdata[i][t-1][2][3]+nsimdata[i][t][2][3]-nsimdata[i][t][3][3]
			simdata[i][t][2][4]=simdata[i][t-1][2][4]+nsimdata[i][t][2][4]-nsimdata[i][t][3][4]
			simdata[i][t][2][5]=simdata[i][t-1][2][5]+nsimdata[i][t][2][5]-nsimdata[i][t][3][5]
			simdata[i][t][2][6]=simdata[i][t-1][2][6]+nsimdata[i][t][2][6]-nsimdata[i][t][3][6]
	#calculate RMSE on last day of training data		
	for i in range(cnum):
		for t in range(days-testdays-1,days-testdays): 
			res[i][t]=(simdata[i][t][2][0]-cdata[i][t][2][0])
	return res,simdata,nsimdata

#Algorithm 2
def srch():
	global bestloss
	global bestp1
	global bestp2
	for pp2 in range(1,11): #grid search for p2
		p2=pp2/10
		left=0
		right=10000
		while left<=right: # binary search for p1
			mid=(left+right)/2
			p1=mid/10000
			ls,simdata,nsimdata=sim(p1,p2)
			sm=sum(sum(ls))
			sz=ls.size
			curloss=math.sqrt(sum(sum(ls**2))/sz)
			if curloss<bestloss:
				bestp1=p1
				bestp2=p2
				bestloss=curloss
			if sm<0:
				left=mid+1
			elif sm>0:
				right=mid-1
			else:
				break
	return bestp1,bestp2

#calculating relative RMSE
def calcmse2(simdata,realdata):
	res=np.zeros((1,days))
	for i in range(1):
		for t in range(days-1,days):
			res[i][t]=(simdata[i][t][2][0]-realdata[i][t][2][0])/realdata[i][t][2][0] 
	sm=sum(sum(res))
	sz=1
	mse=math.sqrt(sum(sum(res**2))/sz)
	return mse

def main():
	global cnum
	global days
	global bestp1
	global bestp2
	global bestloss
	loaddata2()
	preddata=np.zeros((cnum,days,comps,gnum+1)) # cumulative data
	msels=[]
	daylist=range(10,51) 
	for i in daylist:
		bestp1=-1
		bestp2=-1
		bestloss=1000000000000000000
		days=i+testdays+1
		p1,p2=srch()
		ls,simdata,nsimdata=sim(p1,p2)
		preddata[0,i+testdays,2,0]=simdata[0][i+testdays][2][0]
		msels.append(calcmse2(simdata,cdata))

	print(f"Start EpiInfer output for testdays={testdays}")
	print(sum(msels)/len(msels))
	print(f"End EpiInfer output for testdays={testdays}")
	


main()


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
days=101 # total number of days of the simulation
cnum=1 # number of locales
comps=4 # compartments: S,E,I,R
inc=7 #incubation period between being exposed and infected
gnum=6 # number of groups
pop=np.zeros((cnum,gnum+1),dtype=np.float64)
pop[0][0]=8913088 # population of the locale
#population from each group having a distinct meeting distribution
pop[0][1]=0.887*8913088
pop[0][2]=0.076*8913088
pop[0][3]=0.019*8913088
pop[0][4]=0.011*8913088
pop[0][5]=0.004*8913088
pop[0][6]=0.003*8913088
perc=np.zeros((cnum,gnum+1),dtype=np.float64) # percentage of people from each group
for i in range(cnum):
	for g in range(1,gnum+1):
		perc[i][g]=pop[i][g]/pop[i][0] 


cdata=np.zeros((cnum,days,comps,gnum+1)) # cumulative data (4D array where cdata[i][j][k][l] gives the number of people from locale i on day j from compartment k and group l, where group 0 is everyone)
ndata=np.zeros((cnum,days,comps,gnum+1)) # new data (4D array where cdata[i][j][k][l] gives the number of people from locale i on day j from compartment k and group l, where group 0 is everyone)

# daily number of interactions of a person based on their group
nmeet=[3,8,15,35,75,100]


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
				for g in range(1,gnum+1):
					cdata[i][j][k][g]=perc[i][g]*cdata[i][j][k][0]
				
		for j,line in enumerate(f2):
			ls=line.split(',')
			for k,num in enumerate(ls):
				ndata[i][j][k][0]=float(num)
				for g in range(1,gnum+1):
					ndata[i][j][k][g]=perc[i][g]*ndata[i][j][k][0]
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
			infsum=np.zeros(gnum+1,dtype=np.float64)
			asymp=np.zeros(gnum+1,dtype=np.float64)
			ptran=np.zeros(gnum+1,dtype=np.float64)
			asymp_notexp=np.zeros(gnum+1,dtype=np.float64)
			newexp=np.zeros(gnum+1,dtype=np.float64)
			limit=days-testdays
			for j in range(1,inc+1):
				if t-j>=0 and t-j>=limit:
					for g in range(gnum+1):
						infsum[g]+=nsimdata[i][t-j][2][g]
				elif t-j>=0 and t-j<limit:
					for g in range(gnum+1):
						infsum[g]+=ndata[i][t-j][2][g] 
			if t-inc>=limit:
				for g in range(gnum+1):
					asymp[g]=pop[i][g]-simdata[i][t-inc][2][g] 
			else:
				for g in range(gnum+1):
					asymp[g]=pop[i][g]-cdata[i][t-inc][2][g] 
			p_exp=infsum[0]/(p2*asymp[0]) # eq 1
			for g in range(1,gnum+1):
				ptran[g]=1-(1-p1)**(nmeet[g-1]*p_exp) # eq2
			for g in range(gnum+1):
				asymp_notexp[g]=asymp[g]-infsum[g]/p2-simdata[i][t-inc][3][g] # eq 3
			for g in range(1,gnum+1):
				asymp_notexp[g]=max(asymp_notexp[g],0)
			for g in range(1,gnum+1):
				newexp[g]=asymp_notexp[g]*ptran[g] # eq4
				newexp[0]+=newexp[g]
			if t-inc>=inc:
				for g in range(gnum+1):	
					nsimdata[i][t-inc][1][g]=newexp[g]
			for g in range(gnum+1):
				nsimdata[i][t][2][g]=newexp[g]*p2 # eq 5
			for g in range(gnum+1):
				nsimdata[i][t][3][g]=simdata[i][t-1][2][g]*gamma
			for g in range(gnum+1):
				nsimdata[i][t][0][g]=simdata[i][t-1][3][g]*eta # lose immunity
			for g in range(gnum+1): 
				simdata[i][t][3][g]=simdata[i][t-1][3][g]-nsimdata[i][t][0][g]+nsimdata[i][t][3][g] # total recovered
			for g in range(gnum+1): 
				simdata[i][t][2][g]=simdata[i][t-1][2][g]+nsimdata[i][t][2][g]-nsimdata[i][t][3][g]
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
	print(preddata[0,10+testdays:50+testdays+1,2,0])
	print(f"End EpiInfer output for testdays={testdays}")
	


main()

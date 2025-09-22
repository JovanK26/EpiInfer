import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import argparse

countries=["Austria"]
parser = argparse.ArgumentParser()
parser.add_argument('--testdays', type=int, default=10, help='Optional testdays value (default: 10)')
args = parser.parse_args()
testdays=args.testdays #how many days in the future we are predicting
pop=[8677088] # array of the population of each locale
#population from each group having a distinct meeting distribution
pop1=[0.2*8677088]
pop2=[0.6*8677088]
pop3=[0.2*8677088]
days=101 # total number of days of the simulation
cnum=1 # number of locales
comps=4 # compartments: S,E,I,R
inc=7 #incubation period between being exposed and infected
gnum=3 # number of groups
cdata=np.zeros((cnum,days,comps,gnum+1)) # cumulative data (4D arraywhere cdata[i][j][k][l] gives the number of people from locale i on day j from compartment k and group l, where group 0 is everyone)
ndata=np.zeros((cnum,days,comps,gnum+1)) # new data (4D array where cdata[i][j][k][l] gives the number of people from locale i on day j from compartment k and group l, where group 0 is everyone)
# daily number of interactions of a person based on their group
nmeet1=(13,10)
nmeet2=(15,18)
nmeet3=(10,3) 
gamma=0.1 # recovery rate. It is chosen based on the parameter value in the EpiPolicy simulator
eta=0.1 # rate of going from recovered to susceptible. It is chosen based on the parameter value in the EpiPolicy simulator

bestp1=-1
bestp2=-1
bestloss=1000000000000000000

# load data from csv files
def loaddata2():
	for i,cnt in enumerate(countries):
		add=""
		if i>4:
			add="New"
		cfname="Data\\cumu"+cnt+"Grp"+add+".csv"
		nfname="Data\\new"+cnt+"Grp"+add+".csv"
		cfname1="Data\\cumu"+cnt+"Young"+add+".csv"
		nfname1="Data\\new"+cnt+"Young"+add+".csv"
		cfname2="Data\\cumu"+cnt+"Adults"+add+".csv"
		nfname2="Data\\new"+cnt+"Adults"+add+".csv"
		cfname3="Data\\cumu"+cnt+"Seniors"+add+".csv"
		nfname3="Data\\new"+cnt+"Seniors"+add+".csv"
		f1=open(cfname,'r')
		f2=open(nfname,'r')
		f1y=open(cfname1,'r')
		f2y=open(nfname1,'r')
		f1a=open(cfname2,'r')
		f2a=open(nfname2,'r')
		f1s=open(cfname3,'r')
		f2s=open(nfname3,'r')
		for j,line in enumerate(f1):
			ls=line.split(',')
			for k,num in enumerate(ls):
				cdata[i][j][k][0]=float(num)
		for j,line in enumerate(f2):
			ls=line.split(',')
			for k,num in enumerate(ls):
				ndata[i][j][k][0]=float(num)
		for j,line in enumerate(f1y):
			ls=line.split(',')
			for k,num in enumerate(ls):
				cdata[i][j][k][1]=float(num)
		for j,line in enumerate(f2y):
			ls=line.split(',')
			for k,num in enumerate(ls):
				ndata[i][j][k][1]=float(num)
		for j,line in enumerate(f1a):
			ls=line.split(',')
			for k,num in enumerate(ls):
				cdata[i][j][k][2]=float(num)
		for j,line in enumerate(f2a):
			ls=line.split(',')
			for k,num in enumerate(ls):
				ndata[i][j][k][2]=float(num)
		for j,line in enumerate(f1s):
			ls=line.split(',')
			for k,num in enumerate(ls):
				cdata[i][j][k][3]=float(num)
		for j,line in enumerate(f2s):
			ls=line.split(',')
			for k,num in enumerate(ls):
				ndata[i][j][k][3]=float(num)
		f1.close()
		f2.close()
		f1y.close()
		f2y.close()
		f1a.close()
		f2a.close()
		f1s.close()
		f2s.close()


# EpiInfer-core 
def sim(p1,p2):
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
			limit=days-testdays
			for j in range(1,inc+1):
				if t-j>=0:
					infsum+=nsimdata[i][t-j][2][0]
					infsum1+=nsimdata[i][t-j][2][1]
					infsum2+=nsimdata[i][t-j][2][2]
					infsum3+=nsimdata[i][t-j][2][3]

			asymp=pop[i]-simdata[i][t-inc][2][0]  
			asymp1=pop1[i]-simdata[i][t-inc][2][1]
			asymp2=pop2[i]-simdata[i][t-inc][2][2]
			asymp3=pop3[i]-simdata[i][t-inc][2][3]
			
			p_exp=infsum/(p2*asymp) # eq 1
			index=0
			ptran1=1-(1-p1)**(nmeet1[index]*p_exp) # eq 2
			ptran2=1-(1-p1)**(nmeet2[index]*p_exp)
			ptran3=1-(1-p1)**(nmeet3[index]*p_exp)
			asymp_notexp=asymp-(infsum/p2)-simdata[i][t-inc][3][0] # eq 3
			asymp_notexp1=asymp1-(infsum1/p2)-simdata[i][t-inc][3][1]
			asymp_notexp2=asymp2-(infsum2/p2)-simdata[i][t-inc][3][2]
			asymp_notexp3=asymp3-(infsum3/p2)-simdata[i][t-inc][3][3]
			asymp_notexp1=max(asymp_notexp1,0)
			asymp_notexp2=max(asymp_notexp2,0)
			asymp_notexp3=max(asymp_notexp3,0)
			newexp1=asymp_notexp1*ptran1 #eq 4
			newexp2=asymp_notexp2*ptran2
			newexp3=asymp_notexp3*ptran3
			newexp=newexp1+newexp2+newexp3
			if t-inc>=inc:
				nsimdata[i][t-inc][1][0]=newexp
				nsimdata[i][t-inc][1][1]=newexp1
				nsimdata[i][t-inc][1][2]=newexp2
				nsimdata[i][t-inc][1][3]=newexp3
			nsimdata[i][t][2][0]=newexp*p2 # eq 5
			nsimdata[i][t][2][1]=newexp1*p2
			nsimdata[i][t][2][2]=newexp2*p2
			nsimdata[i][t][2][3]=newexp3*p2
			nsimdata[i][t][3][0]=simdata[i][t-1][2][0]*gamma 
			nsimdata[i][t][3][1]=simdata[i][t-1][2][1]*gamma
			nsimdata[i][t][3][2]=simdata[i][t-1][2][2]*gamma
			nsimdata[i][t][3][3]=simdata[i][t-1][2][3]*gamma
			nsimdata[i][t][0][0]=simdata[i][t-1][3][0]*eta # lose imunity 
			nsimdata[i][t][0][1]=simdata[i][t-1][3][1]*eta 
			nsimdata[i][t][0][2]=simdata[i][t-1][3][2]*eta 
			nsimdata[i][t][0][3]=simdata[i][t-1][3][3]*eta 
			simdata[i][t][3][0]=(simdata[i][t-1][3][0]-nsimdata[i][t][0][0])+nsimdata[i][t][3][0] # total rec
			simdata[i][t][3][1]=simdata[i][t-1][3][1]-nsimdata[i][t][0][1]+nsimdata[i][t][3][1] 
			simdata[i][t][3][2]=simdata[i][t-1][3][2]-nsimdata[i][t][0][2]+nsimdata[i][t][3][2] 
			simdata[i][t][3][3]=simdata[i][t-1][3][3]-nsimdata[i][t][0][3]+nsimdata[i][t][3][3] 
			simdata[i][t][2][0]=simdata[i][t-1][2][0]+nsimdata[i][t][2][0]-nsimdata[i][t][3][0]
			simdata[i][t][2][1]=simdata[i][t-1][2][1]+nsimdata[i][t][2][1]-nsimdata[i][t][3][1]
			simdata[i][t][2][2]=simdata[i][t-1][2][2]+nsimdata[i][t][2][2]-nsimdata[i][t][3][2]
			simdata[i][t][2][3]=simdata[i][t-1][2][3]+nsimdata[i][t][2][3]-nsimdata[i][t][3][3]
	#calculate RMSE on last day of training data		
	for i in range(cnum):
		for t in range(days-testdays-1,days-testdays):
			res[i][t]=(simdata[i][t][2][0]-cdata[i][t][2][0])/pop[i]
	return res,simdata,nsimdata

# Algorithm 2 
def srch(): 
	global bestloss
	global bestp1
	global bestp2
	for pp2 in range(1,11): # grid search for p2
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


#calculate RMSE as described in the paper
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
		preddata[0][i+testdays][2][0]=simdata[0][i+testdays][2][0]
		msels.append(calcmse2(simdata,cdata))

	print(f"Start EpiInfer EpiPolicy output for testdays={testdays}")
	print("Error List: ", msels)
	print("Average Error: ", sum(msels)/len(msels))
	print("Daily Predictions: ", preddata[0,10+testdays:50+testdays+1,2,0])
	print(f"End EpiInfer EpiPolicy output for testdays={testdays}")
	


main()



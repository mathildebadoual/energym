#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:30:29 2018

@author: shpark
"""

import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline #import interp1d
from spm.envs.ParamFile_LGC import *

def spm_plant_obs_mats(p):

	# Electorchemical Model params
	alpha_n = p['D_s_n0'] / (p['R_s_n'] * p['delta_r_n'])**2
	alpha_p = p['D_s_p0'] / (p['R_s_p'] * p['delta_r_p'])**2

	# Block matrices
	M1_n = np.zeros((p['Nr']-1,p['Nr']-1))
	M1_p = np.zeros((p['Nr']-1,p['Nr']-1))

	for idx in range(0,p['Nr']-1):
		# Lower diagonal
		if(idx != 0):
			M1_n[idx,idx-1] = (idx) / float(idx+1) * alpha_n
			M1_p[idx,idx-1] = (idx) / float(idx+1) * alpha_p

		# Main diagonal
		M1_n[idx,idx] = -2*alpha_n
		M1_p[idx,idx] = -2*alpha_p

		# Upper diagonal
		if(idx != p['Nr']-2):
			M1_n[idx,idx+1] = (idx+2)/ float(idx+1) * alpha_n
			M1_p[idx,idx+1] = (idx+2)/ float(idx+1) * alpha_p


	M2_n = np.zeros((p['Nr']-1,2))
	M2_p = np.zeros((p['Nr']-1,2))

	M2_n[-1,-1] = p['Nr'] / float(p['Nr']-1) * alpha_n
	M2_p[-1,-1] = p['Nr'] / float(p['Nr']-1) * alpha_p

	N1 = np.zeros((2,p['Nr']-1))

	# 2nd order BCs
	N1[0,0] = 4
	N1[0,1] = -1
	N1[1,-1] = -4
	N1[1,-2] = 1

	N2 = np.diag([-3, 3])

	N3_n = np.array([[0], [-(2*p['delta_r_n'] * p['R_s_n'])/(p['D_s_n0'] * p['Faraday'] * p['a_s_n'] * p['Area'] * p['L_n'])]]);
	N3_p = np.array([[0], [(2*p['delta_r_p'] * p['R_s_p'])/(p['D_s_p0'] * p['Faraday'] * p['a_s_p'] * p['Area'] * p['L_p'])]]);


	# A,B matrices for each electrode
	A_n = M1_n - np.dot(M2_n,np.dot(np.linalg.inv(N2),N1))
	"A_n = M1_n - M2_n*N2_n^(-1)N1_n"
	A_p = M1_p - np.dot(M2_p,np.dot(np.linalg.inv(N2),N1))

	B_n = np.dot(M2_n,np.dot(np.linalg.inv(N2),N3_n))
	"M2_n*N2^(-1)*N3"
	B_p = np.dot(M2_p,np.dot(np.linalg.inv(N2),N3_p))

	# C,D matrices for each electrode
	C_n = -np.dot(np.array([[0,1]]),np.dot(np.linalg.inv(N2),N1))
	C_p = -np.dot(np.array([[0,1]]),np.dot(np.linalg.inv(N2),N1))

	D_n = np.dot(np.array([[0,1]]),np.dot(np.linalg.inv(N2),N3_n))
	D_p = np.dot(np.array([[0,1]]),np.dot(np.linalg.inv(N2),N3_p))

	return A_n, A_p, B_n, B_p, C_n, C_p, D_n, D_p


def init_cs_NMC(p,V0):
	# This init_cs function is for NMC
	# Used for LG chem

	max_iters = 5000
	x = np.zeros(max_iters)
	f = np.nan * np.ones(max_iters)
	tol = 1e-5

	# Stoichiometry pts

	x0 = 0.032
	x100 = 0.932
	y0 = 0.863
	y100 = 0.237

	# Theta grid
	n_points = 1000000
	theta_n = np.linspace(x0,x100,n_points)
	theta_p = np.linspace(y0,y100,n_points)
	CellSOC = np.linspace(1,0,n_points)

	OCPn = refPotentialAnode_casadi(theta_n)
	OCPp = refPotentialCathode_casadi(theta_p)
	OCV = np.zeros(n_points)
	OCV = OCPp - OCPn

	minDistance = np.min(abs(V0-OCV))
	indexOfMin = np.argmin(abs(V0-OCV))

	theta_n0 = theta_n[indexOfMin]
	theta_p0 = theta_p[indexOfMin]

	checkV = refPotentialCathode_casadi(theta_p0) - refPotentialAnode_casadi(theta_n0)

	if abs(checkV - V0) > tol and V0 >4.0:
		print('Check init_cs_NMC function, initial conditions not found')
		sys.exit(1)

	if abs(checkV - V0) > 1e-3 and V0 < 3.0:
		print('Check init_cs_NMC function, initial conditions not found')
		sys.exit(1)


	csn0 = theta_n0 * p['c_s_n_max']
	csp0 = theta_p0 * p['c_s_p_max']

	return csn0, csp0

def init_cs(p,V0):

	# Bi-section algorithm parameters
	max_iters = 5000
	x = np.zeros(max_iters)
	f = np.nan * np.ones(max_iters)
	tol = 1e-5

	# Interpolation

	mat_contents = sio.loadmat('NCA_SOC_OCV_MAP.mat')
	flip_volt = mat_contents['flip_volt']
	soc1 = mat_contents['soc1']
	flipVolt, index = np.unique(flip_volt, return_index = True)
	soc_index = soc1[0,index]
	#set_interp = interp1d(flipVolt,soc_index, kind='cubic',bounds_error=False) #cubic = spline
	set_interp = CubicSpline(flipVolt,soc_index) #cubic = spline
	soc00 = set_interp(V0)

	csn0 = 34265*(soc00) + 44.5
	csp0 = 46053-(soc00)*35934.6

	x_low = 46053-(1)*35934.6
	x_high = 46053-(0)*35934.6
	x[0] = csp0
	for idx in range(max_iters):
	    theta_p = x[idx] / p['c_s_p_max']
	    theta_n = (p['n_Li_s']-p['epsilon_s_p']*p['L_p']*p['Area']*x[idx])/(p['c_s_n_max']*p['epsilon_s_n']*p['L_n']*p['Area']);
	    OCPn = refPotentialAnode_casadi(theta_n)
	    OCPp = refPotentialCathode_casadi(theta_p)
	    f[idx] = OCPp - OCPn - V0

	    if np.abs(f[idx]) <= tol :
	        break
	    elif f[idx] <= 0 :
	        x_high = x[idx]
	    else:
	        x_low = x[idx]

	    # Bisection
	    x[idx+1] = (x_high + x_low) / 2

	    if idx == max_iters :
	        print('PLEASE CHECK INITIAL VOLTAGE & CONDITION')

	csp0 = x[idx]
	return csn0, csp0



def nonlinear_SPM_Voltage(p, c_ss_n,c_ss_p,cen_bar,ces_bar,cep_bar,I):
	# Stochiometric Concentration Ratio
	theta_n = c_ss_n / p['c_s_n_max']
	theta_p = c_ss_p / p['c_s_p_max']


	# Equilibrium Potential
	Unref = refPotentialAnode_casadi(theta_n)
	Upref = refPotentialCathode_casadi(theta_p)

	# Exchange Current Density
	c_e = np.zeros(p['Nx'])
	c_e[range(p['Nxn'])] = cen_bar
	c_e[range(p['Nxn'],p['Nxn']+p['Nxs'])] = ces_bar
	c_e[range(p['Nxn']+p['Nxs'],p['Nx'])] = cep_bar
	i_0n, i_0p = exch_cur_dens(p,c_ss_n,c_ss_p,c_e)
	RTaF = (p['R']*p['T_amb']) / (p['alph']*p['Faraday'])

	voltage = RTaF * np.arcsinh(-I / (2*p['a_s_p']*p['Area']*p['L_p']*i_0p[-1])) \
			 - RTaF * np.arcsinh(I / (2*p['a_s_n']*p['Area']*p['L_n']*i_0n[0])) \
			 + Upref - Unref \
			 - (p['R_f_n'] / (p['a_s_n']*p['L_n']*p['Area']) + p['R_f_p'] / (p['a_s_p']*p['L_p']*p['Area'])) * I

	return voltage



def exch_cur_dens(p, c_ss_n, c_ss_p, c_e):
	c_e_n = c_e[range(p['Nxn'])]
	c_e_p = c_e[range(p['Nxn']+p['Nxs'],p['Nx'])]

	# Compute exchange current density
	i_0n = p['k_n0'] * ((p['c_s_n_max'] - c_ss_n) * c_ss_n * c_e_n)**(0.5)
	i_0p = p['k_p0'] * ((p['c_s_p_max'] - c_ss_p) * c_ss_p * c_e_p)**(0.5)

	return i_0n, i_0p

# #comment these out and put in paramfile_LGC saehongs code
# def refPotentialAnode(p,theta):
#  c_n=np.array([-0.084294960339275,
#    0.920754744005144,
#   -0.500066623566425,
#    0.062731837918546,
#    0.782151587417570,
#   -0.373761901864611,
#    0.019988184317997,
#    0.543282314780430,
#   -0.295609630222051,
#    0.040970248093866,
#    0.231152288743602,
#   -0.217847875913234,
#    0.068744203951316,
#    0.353848415118256,
#   -0.114753994434564,
#   -0.028613032233089,
#    0.260671608316041,
#   -0.212058177468640,
#   -0.025506157489854,
#    0.211741908826122,
#   -0.241880220004548,
#    0.188872027034948,
#    0.867520021192469,
#   -0.225038983698359,
#   -0.111904175370177,
#    0.537399173641857,
#   -0.020780743382893,
#    0.108353745941168,
#    0.537735904911254,
#   -0.020226723056513,
#    0.171375773597772,
#    0.729717193288193,
#   -0.323902793428930,
#    0.064143152162965,
#    1.289849595601526,
#    0.704961322280748,
#    0.023028206444624,
#    0.481699223765299,
#   -0.076233450161839,
#   -0.182559256738691,
#    0.830851470359638,
#   -0.226362977193547,
#   -0.040952011143767,
#    1.626936110900125,
#    0.295695270567609,
#   -1.000228763094078,
#    0.007914258576845,
#   -0.016476666187381,
#   -0.341740372496750,
#    0.001274961492701,
#   -0.004879090290810,
#   -0.930906698538900,
#    0.001549868904555,
#   -0.010583717929547,
#    2.554274538083029,
#   -0.012402969675540,
#   -0.029257893810540,
#   -0.512533408582419,
#    0.066122834568301,
#   -0.077930639597751,
#   -0.499673574757569,
#    0.044470609922510,
#   -0.134483437256594,
#    1.904111886758372,
#   -0.035336812622768,
#   -0.306171040837701,
#   -1.122974595772499,
#    0.028740372472439,
#   -0.079271479637875,
#   -0.093855421675871,
#    0.930843806570863,
#   -0.516652668839875,
#   -0.846383609865041,
#    0.012151749801329,
#   -0.029511731110250,
#   -0.561782895480513,
#    0.098392530745244,
#   -0.109853910868333,
#   -0.818206413176353,
#    0.026850808833446,
#   -0.051805538572186,
#   -0.525543070925015,
#    0.188590232596615,
#   -0.192054642003214,
#   -0.046580230674248,
#    0.002863828671823,
#   -0.000914487593373,
#    2.650656293235332,
#   -0.008182255230700,
#   -0.117937922743741,
#   -0.295664205008775,
#    0.137690106957231,
#   -0.310460986123659,
#   -0.835065551163236,
#    0.711574616090746,
#   -0.997353098073145,
#    0.415746756470558,
#    0.423984781966332,
#    3.189835673119072,
#    0.413779708001205,
#    0.426343693564050,
#    3.190867502582611])
#
#  Uref = c_n[0]*np.exp(-((theta - c_n[1])**2/c_n[2]**2))+ \
#     c_n[3]*np.exp(-((theta - c_n[4])**2/c_n[5]**2))+ \
#     c_n[6]*np.exp(-((theta - c_n[7])**2/c_n[8]**2))+ \
#     c_n[9]*np.exp(-((theta - c_n[10])**2/c_n[11]**2))+ \
#     c_n[12]*np.exp(-((theta - c_n[13])**2/c_n[14]**2))+ \
#     c_n[15]*np.exp(-((theta - c_n[16])**2/c_n[17]**2))+ \
#     c_n[18]*np.exp(-((theta - c_n[19])**2/c_n[20]**2))+ \
#     c_n[21]*np.exp(-((theta - c_n[22])**2/c_n[23]**2))+ \
#     c_n[24]*np.exp(-((theta - c_n[25])**2/c_n[26]**2))+ \
#     c_n[27]*np.exp(-((theta - c_n[28])**2/c_n[29]**2))+ \
#     c_n[30]*np.exp(-((theta - c_n[31])**2/c_n[32]**2))+ \
#     c_n[33]*np.exp(-((theta - c_n[34])**2/c_n[35]**2))+ \
#     c_n[36]*np.exp(-((theta - c_n[37])**2/c_n[38]**2))+ \
#     c_n[39]*np.exp(-((theta - c_n[40])**2/c_n[41]**2))+ \
#     c_n[42]*np.exp(-((theta - c_n[43])**2/c_n[44]**2))+ \
#     c_n[45]*np.exp(-((theta - c_n[46])**2/c_n[47]**2))+ \
#     c_n[48]*np.exp(-((theta - c_n[49])**2/c_n[50]**2))+ \
#     c_n[51]*np.exp(-((theta - c_n[52])**2/c_n[53]**2))+ \
#     c_n[54]*np.exp(-((theta - c_n[55])**2/c_n[56]**2))+ \
#     c_n[57]*np.exp(-((theta - c_n[58])**2/c_n[59]**2))+ \
#     c_n[60]*np.exp(-((theta - c_n[61])**2/c_n[62]**2))+ \
#     c_n[63]*np.exp(-((theta - c_n[64])**2/c_n[65]**2))+ \
#     c_n[66]*np.exp(-((theta - c_n[67])**2/c_n[68]**2))+ \
#     c_n[69]*np.exp(-((theta - c_n[70])**2/c_n[71]**2))+ \
#     c_n[72]*np.exp(-((theta - c_n[73])**2/c_n[74]**2))+ \
#     c_n[75]*np.exp(-((theta - c_n[76])**2/c_n[77]**2))+ \
#     c_n[78]*np.exp(-((theta - c_n[79])**2/c_n[80]**2))+ \
#     c_n[81]*np.exp(-((theta - c_n[82])**2/c_n[83]**2))+ \
#     c_n[84]*np.exp(-((theta - c_n[85])**2/c_n[86]**2))+ \
#     c_n[87]*np.exp(-((theta - c_n[88])**2/c_n[89]**2))+ \
#     c_n[90]*np.exp(-((theta - c_n[91])**2/c_n[92]**2))+ \
#     c_n[93]*np.exp(-((theta - c_n[94])**2/c_n[95]**2))+ \
#     c_n[96]*np.exp(-((theta - c_n[97])**2/c_n[98]**2))+ \
#     c_n[99]*np.exp(-((theta - c_n[100])**2/c_n[101]**2))
#
#  return Uref
#
#
# def refPotentialCathode(p,theta):
#  c_p= np.array([ -40.045585568588542,
#  -62.042811084183654,
#   52.447046217508564,
#  -11.281882678497276,
#   63.276043910291172,
#   21.159687366489823,
#   37.390508845897301,
#   22.261671639629835,
#    8.574181451931103,
#   10.133001156239731,
#   -3.313604725236584,
#    1.977856101799057,
#   -3.046181118828750,
#   -0.087883198680019,
#   -0.836818408057937,
#   -0.072435003409969,
#   -0.069320106210594,
#    4.456159792325947])
#
#  w=c_p[-1]
#
#  Uref=c_p[0] + c_p[1]*np.cos(theta*w) + c_p[2]*np.sin(theta*w) + \
#   c_p[3]*np.cos(2*theta*w) + c_p[4]*np.sin(2*theta*w) + c_p[5]*np.cos(3*theta*w) + c_p[6]*np.sin(3*theta*w) + \
#   c_p[7]*np.cos(4*theta*w) + c_p[8]*np.sin(4*theta*w) + c_p[9]*np.cos(5*theta*w) + c_p[10]*np.sin(5*theta*w) + \
#   c_p[11]*np.cos(6*theta*w) + c_p[12]*np.sin(6*theta*w) + c_p[13]*np.cos(7*theta*w) + c_p[14]*np.sin(7*theta*w) + \
#   c_p[15]*np.cos(8*theta*w) + c_p[16]*np.sin(8*theta*w)
#
#  return Uref

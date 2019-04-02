#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr. 1 2019

NCA-18650 cell

@author: shpark

Parameters from NCA-18650

Reference:
Park et al., "Optimal Experimental Design for Parameterization of an Electrochemical Lithium-ion Battery Model"
Journal of The Electrochemical Society, 165(7), 2018
"""

import numpy as np

p={}


#==============================================================================
# Geometric params
#==============================================================================

# Thickness of each layer
p['L_n'] = 79.0e-6       # Thickness of negative electrode [m]
p['L_s'] = 80.0e-6       # Thickness of separator [m]
p['L_p'] = 61.5e-6     # Thickness of positive electrode [m]

L_ccn = 25e-6;    # Thickness of negative current collector [m]
L_ccp = 25e-6;    # Thickness of negative current collector [m]


# Particle Radii
p['R_s_n'] = 2.0249e-05 # Radius of solid particles in negative electrode [m]
p['R_s_p'] = 1.6973e-05 # Radius of solid particles in positive electrode [m]

# Volume fractions
p['epsilon_s_n'] = 0.543889597565723 # Volume fraction in solid for neg. electrode
p['epsilon_s_p'] = 0.666364981170368 # Volume fraction in solid for pos. electrode

p['epsilon_e_n'] = 0.347495486967184   # Volume fraction in electrolyte for neg. electrode
p['epsilon_e_s'] = 0.5	  # Volume fraction in electrolyte for separator
p['epsilon_e_p'] = 0.330000000000000   # Volume fraction in electrolyte for pos. electrode

p['epsilon_f_n'] = 1 - p['epsilon_s_n'] - p['epsilon_e_n']  # Volume fraction of filler in neg. electrode
p['epsilon_f_p'] = 1 - p['epsilon_s_p'] - p['epsilon_e_p']  # Volume fraction of filler in pos. electrode


# Specific interfacial surface area
p['a_s_n'] = 3*p['epsilon_s_n'] / p['R_s_n']  # Negative electrode [m^2/m^3]
p['a_s_p'] = 3*p['epsilon_s_p'] / p['R_s_p']  # Positive electrode [m^2/m^3]


#==============================================================================
# Transport params
#==============================================================================

p['D_s_n0'] = 2.63029669224544e-14 # Diffusion coeff for solid in neg. electrode, [m^2/s]
p['D_s_p0'] = 6.81035680483463e-14 # Diffusion coeff for solid in pos. electrode, [m^2/s]


# Conductivity of solid
p['sig_n'] = 100    # Conductivity of solid in neg. electrode, [1/Ohms*m]
p['sig_p'] = 100    # Conductivity of solid in pos. electrode, [1/Ohms*m]

#==============================================================================
# Kinetic params
#==============================================================================
p['R_f_n'] = 0       # Resistivity of SEI layer, [Ohms*m^2]
p['R_f_p'] = 0  # Resistivity of SEI layer, [Ohms*m^2]
#p.R_c = 2.5e-03;%5.1874e-05/p.Area; % Contact Resistance/Current Collector Resistance, [Ohms-m^2]

# Nominal Reaction rates
p['k_n0'] = 7.50e-03  # Reaction rate in neg. electrode, [(A/m^2)*(mol^3/mol)^(1+alpha)]
p['k_p0'] = 2.30e-03  # Reaction rate in pos. electrode, [(A/m^2)*(mol^3/mol)^(1+alpha)]


#==============================================================================
# Thermodynamic params
#==============================================================================

# Thermal dynamics
p['C_p'] = 2000    # Heat capacity, [J/kg-K]
p['R_th'] = 2      # Thermal resistance, [K/W]
p['mth'] = 0.834   # Mass of cell [Kg]

# Activation Energies
# Taken from Zhang et al (2014) [Harbin]
# http://dx.doi.org/10.1016/j.jpowsour.2014.07.110
# All units are [J/mol]
p['E_kn'] = 37.48e+3
p['E_kp'] = 39.57e+3
p['E_Dsn'] = 42.77e+3
p['E_Dsp'] = 18.55e+3
p['E_De'] = 37.04e+3
p['E_kappa_e'] = 34.70e+3


# Ambient Temperature
p['T_amb'] = 298.15 # [K]
p['T_ref'] = 298.15 # [K] for ElectrolyteACT

#==============================================================================
# Miscellaneous
#==============================================================================
p['R'] = 8.314472;      # Gas constant, [J/mol-K]
p['Faraday'] = 96485.3329  # Faraday constant [Coulombs/mol]
p['Area'] = 1.425      # Electrode current collector area [m^2]
p['alph'] = 0.5         # Charge transfer coefficients
p['t_plus'] = 0.45		# Transference number
p['brug'] = 1.8		# Bruggeman porosity
#==============================================================================
# Concentrations
#==============================================================================

p['c_s_n_max'] = 3.71e+04 # Max concentration in anode, [mol/m^3]
p['c_s_p_max'] = 5.10e+04 # Max concentration in cathode, [mol/m^3]
p['n_Li_s'] = 0.1406 # Total moles of lithium in solid phase [mol]
p['c_e0'] = 1.0e3    # Electrolyte concentration [mol/m^3]

#==============================================================================
# Discretization params
#==============================================================================
p['PadeOrder'] = 3


p['Nr'] = 20
p['delta_r_n'] = 1/float(p['Nr'])
p['delta_r_p'] = 1/float(p['Nr'])

p['Nxn'] = 10;
p['Nxs'] = 5;
p['Nxp'] = 10;
p['Nx'] = p['Nxn']+p['Nxs']+p['Nxp']

p['delta_x_n'] = 1 / float(p['Nxn'])
p['delta_x_s'] = 1 / float(p['Nxs'])
p['delta_x_p'] = 1 / float(p['Nxp'])


def refPotentialAnode_casadi(theta):
 

    c_n=np.array([-0.084294960339275,
       0.920754744005144,
      -0.500066623566425,
       0.062731837918546,
       0.782151587417570,
      -0.373761901864611,
       0.019988184317997,
       0.543282314780430,
      -0.295609630222051,
       0.040970248093866,
       0.231152288743602,
      -0.217847875913234,
       0.068744203951316,
       0.353848415118256,
      -0.114753994434564,
      -0.028613032233089,
       0.260671608316041,
      -0.212058177468640,
      -0.025506157489854,
       0.211741908826122,
      -0.241880220004548,
       0.188872027034948,
       0.867520021192469,
      -0.225038983698359,
      -0.111904175370177,
       0.537399173641857,
      -0.020780743382893,
       0.108353745941168,
       0.537735904911254,
      -0.020226723056513,
       0.171375773597772,
       0.729717193288193,
      -0.323902793428930,
       0.064143152162965,
       1.289849595601526,
       0.704961322280748,
       0.023028206444624,
       0.481699223765299,
      -0.076233450161839,
      -0.182559256738691,
       0.830851470359638,
      -0.226362977193547,
      -0.040952011143767,
       1.626936110900125,
       0.295695270567609,
      -1.000228763094078,
       0.007914258576845,
      -0.016476666187381,
      -0.341740372496750,
       0.001274961492701,
      -0.004879090290810,
      -0.930906698538900,
       0.001549868904555,
      -0.010583717929547,
       2.554274538083029,
      -0.012402969675540,
      -0.029257893810540,
      -0.512533408582419,
       0.066122834568301,
      -0.077930639597751,
      -0.499673574757569,
       0.044470609922510,
      -0.134483437256594,
       1.904111886758372,
      -0.035336812622768,
      -0.306171040837701,
      -1.122974595772499,
       0.028740372472439,
      -0.079271479637875,
      -0.093855421675871,
       0.930843806570863,
      -0.516652668839875,
      -0.846383609865041,
       0.012151749801329,
      -0.029511731110250,
      -0.561782895480513,
       0.098392530745244,
      -0.109853910868333,
      -0.818206413176353,
       0.026850808833446,
      -0.051805538572186,
      -0.525543070925015,
       0.188590232596615,
      -0.192054642003214,
      -0.046580230674248,
       0.002863828671823,
      -0.000914487593373,
       2.650656293235332,
      -0.008182255230700,
      -0.117937922743741,
      -0.295664205008775,
       0.137690106957231,
      -0.310460986123659,
      -0.835065551163236,
       0.711574616090746,
      -0.997353098073145,
       0.415746756470558,
       0.423984781966332,
       3.189835673119072,
       0.413779708001205,
       0.426343693564050,
       3.190867502582611])
     
    Uref=c_n[0]*np.exp(-((theta - c_n[1])**2/c_n[2]**2))+ \
         c_n[3]*np.exp(-((theta - c_n[4])**2/c_n[5]**2))+ \
         c_n[6]*np.exp(-((theta - c_n[7])**2/c_n[8]**2))+ \
         c_n[9]*np.exp(-((theta - c_n[10])**2/c_n[11]**2))+ \
         c_n[12]*np.exp(-((theta - c_n[13])**2/c_n[14]**2))+ \
         c_n[15]*np.exp(-((theta - c_n[16])**2/c_n[17]**2))+ \
         c_n[18]*np.exp(-((theta - c_n[19])**2/c_n[20]**2))+ \
         c_n[21]*np.exp(-((theta - c_n[22])**2/c_n[23]**2))+ \
         c_n[24]*np.exp(-((theta - c_n[25])**2/c_n[26]**2))+ \
         c_n[27]*np.exp(-((theta - c_n[28])**2/c_n[29]**2))+ \
         c_n[30]*np.exp(-((theta - c_n[31])**2/c_n[32]**2))+ \
         c_n[33]*np.exp(-((theta - c_n[34])**2/c_n[35]**2))+ \
         c_n[36]*np.exp(-((theta - c_n[37])**2/c_n[38]**2))+ \
         c_n[39]*np.exp(-((theta - c_n[40])**2/c_n[41]**2))+ \
         c_n[42]*np.exp(-((theta - c_n[43])**2/c_n[44]**2))+ \
         c_n[45]*np.exp(-((theta - c_n[46])**2/c_n[47]**2))+ \
         c_n[48]*np.exp(-((theta - c_n[49])**2/c_n[50]**2))+ \
         c_n[51]*np.exp(-((theta - c_n[52])**2/c_n[53]**2))+ \
         c_n[54]*np.exp(-((theta - c_n[55])**2/c_n[56]**2))+ \
         c_n[57]*np.exp(-((theta - c_n[58])**2/c_n[59]**2))+ \
         c_n[60]*np.exp(-((theta - c_n[61])**2/c_n[62]**2))+ \
         c_n[63]*np.exp(-((theta - c_n[64])**2/c_n[65]**2))+ \
         c_n[66]*np.exp(-((theta - c_n[67])**2/c_n[68]**2))+ \
         c_n[69]*np.exp(-((theta - c_n[70])**2/c_n[71]**2))+ \
         c_n[72]*np.exp(-((theta - c_n[73])**2/c_n[74]**2))+ \
         c_n[75]*np.exp(-((theta - c_n[76])**2/c_n[77]**2))+ \
         c_n[78]*np.exp(-((theta - c_n[79])**2/c_n[80]**2))+ \
         c_n[81]*np.exp(-((theta - c_n[82])**2/c_n[83]**2))+ \
         c_n[84]*np.exp(-((theta - c_n[85])**2/c_n[86]**2))+ \
         c_n[85]*np.exp(-((theta - c_n[88])**2/c_n[89]**2))+ \
         c_n[90]*np.exp(-((theta - c_n[91])**2/c_n[92]**2))+ \
         c_n[93]*np.exp(-((theta - c_n[94])**2/c_n[95]**2))+ \
         c_n[96]*np.exp(-((theta - c_n[97])**2/c_n[98]**2))+ \
         c_n[99]*np.exp(-((theta - c_n[100])**2/c_n[101]**2))

    return Uref

def refPotentialCathode_casadi(theta):


    c_p=np.array([ -40.045585568588542, \
                   -62.042811084183654, \
                    52.447046217508564, \
                   -11.281882678497276, \
                    63.276043910291172, \
                    21.159687366489823, \
                    37.390508845897301, \
                    22.261671639629835, \
                     8.574181451931103, \
                    10.133001156239731, \
                    -3.313604725236584, \
                     1.977856101799057, \
                    -3.046181118828750, \
                    -0.087883198680019, \
                    -0.836818408057937, \
                    -0.072435003409969, \
                    -0.069320106210594, \
                     4.456159792325947])

    w= c_p[-1]

    Uref=c_p[0] + c_p[1]*np.cos(theta*w) + c_p[2]*np.sin(theta*w) + \
      c_p[3]*np.cos(2*theta*w) + c_p[4]*np.sin(2*theta*w) + c_p[5]*np.cos(3*theta*w) + c_p[6]*np.sin(3*theta*w) +  \
      c_p[7]*np.cos(4*theta*w) + c_p[8]*np.sin(4*theta*w) + c_p[9]*np.cos(5*theta*w) + c_p[10]*np.sin(5*theta*w) +  \
      c_p[11]*np.cos(6*theta*w) + c_p[12]*np.sin(6*theta*w) + c_p[13]*np.cos(7*theta*w) + c_p[14]*np.sin(7*theta*w) + \
      c_p[15]*np.cos(8*theta*w) + c_p[16]*np.sin(8*theta*w)


    return Uref

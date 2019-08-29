#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import functools

import numpy as np
import os
import h5py

import common
import utilities_statistics as us

##################################
# Constants
M_sun=1.9891e30
c_light=2.9979e8
G=6.6726e-11
M_atom=1.66053873e-27
H_atom_mass=1.00794
sigma_Thomson=6.6525e-29
J2erg=1e7
eps_eff=0.1
m2cm=1.0e2
kg2g=1.0e3
yr2s=3.1558e7
a=0.67
Lsun=3.839e33

# Choose if jets are powered only by hot gas accretion or by all accretion
jet_all=True

def LBH(mass,acc,f_hot,jet_all):
	mass/=h_sim
	acc/=h_sim
	scale=np.log10(4.0*np.pi)+np.log10(c_light)+np.log10(G)+np.log10(M_sun)+np.log10(M_atom)+np.log10(H_atom_mass)-np.log10(sigma_Thomson)
	scale=10.0**scale
	LEdd=scale*mass*J2erg
	M_dot_Edd=LEdd/(eps_eff*(c_light*m2cm)**2)
	Macc_temp=acc*(M_sun/yr2s)*(kg2g)
	m_dot=Macc_temp/M_dot_Edd
	m_dot_hh=m_dot*f_hot
	
	M_temp=eps_eff*Macc_temp*(c_light*m2cm)**2
	Lbol=np.zeros(len(m_dot))
	Lbol[m_dot<0.01]=np.where(m_dot[m_dot<0.01]>7.5e-6,44*m_dot[m_dot<0.01],6.3e-5)*M_temp[m_dot<0.01]
	Lbol[m_dot>=0.01]=M_temp[m_dot>=0.01]
	Lbol=np.where(Lbol>4*LEdd,4*(1+np.log10(m_dot/4))*LEdd,Lbol)
	
	if jet_all:
		Lmech=np.ones(len(m_dot))
		Lmech[m_dot<0.01]=2.0e45*(mass[m_dot<0.01]/1.0e9)*(100*m_dot[m_dot<0.01])*a**2
		Lmech[m_dot>=0.01]=2.5e43*(mass[m_dot>=0.01]/1.0e9)**1.1*(100*m_dot[m_dot>=0.01])**1.2*a**2
	else:
		Lmech=np.ones(len(m_dot_hh))
		Lmech[m_dot_hh<0.01]=2.0e45*(mass[m_dot_hh<0.01]/1.0e9)*(100*m_dot_hh[m_dot_hh<0.01])*a**2
		Lmech[m_dot_hh>=0.01]=2.5e43*(mass[m_dot_hh>=0.01]/1.0e9)**1.1*(100*m_dot_hh[m_dot_hh>=0.01])**1.2*a**2
	
	l=np.log10(Lbol/(1e12*Lsun))
	lhx=-1.54+0.76*l-0.012*l**2+0.0015*l**3
	Lhx=(1e12*Lsun)*10**lhx
	lsx=-1.65+0.76*l-0.012*l**2+0.0015*l**3
	Lsx=(1e12*Lsun)*10**lsx
	lvb=-0.8+1.067*l-0.017*l**2+0.0023*l**3
	Mb=-11.33-2.5*(lvb+np.log10(Lsun/1e28))
	
	if jet_all:
		return(Lbol,Lmech,m_dot,Lhx,Lsx,Mb)
	else:
		return(Lbol,Lmech,m_dot,m_dot_hh,Lhx,Lsx,Mb)


def prepare_data(hdf5_data,index,model_dir,snapshot,subvol):
	bin_it=functools.partial(us.wmedians,xbins=xmf)
	
	# Unpack data
	(h_sim,_,id_gal,macc_hh,macc_sb,mbh)=hdf5_data
	
	# Calculation of the relevant AGN quantities
	macc=macc_hh+macc_sb
	mass_sel=(mbh>mbh_low)&(macc>0)
	mbh=mbh[mass_sel]
	macc_hh,macc_sb,macc=[macc_hh[mass_sel],macc_sb[mass_sel],macc[mass_sel]]
	macc_hh=macc_hh/macc
	
	if jet_all:
		lbh,lmech,maccr,Lhx,Lsx,Mb=[np.zeros(len(macc))]*6
		lbh[mass_sel],lmech[mass_sel],maccr[mass_sel],Lhx[mass_sel],Lsx[mass_sel],Mb[mass_sel]=LBH(mbh,macc,macc_hh,jet_all)
	else:
		lbh,lmech,maccr,maccr_hh,Lhx,Lsx,Mb=[np.zeros(len(macc))]*7
		lbh[mass_sel],lmech[mass_sel],maccr[mass_sel],maccr_hh[mass_sel],Lhx[mass_sel],Lsx[mass_sel],Mb[mass_sel]=LBH(mbh,macc,macc_hh,jet_all)
	
	# Writing of the hdf5 files with the relevant AGN quantities
	file_to_write=os.path.join(model_dir,str(snapshot),str(subvol),'AGN.hdf5')
	print('Will write extinction to %s' % file_to_write)
	hf=h5py.File(file_to_write,'w')
	
	hf.create_dataset('galaxies/lqso',data=lbh)
	hf.create_dataset('galaxies/lmech',data=lmech)
	hf.create_dataset('galaxies/bh_accretion_ratio',data=maccr)
	hf.create_dataset('galaxies/lqso_hx',data=Lhx)
	hf.create_dataset('galaxies/lqso_sx',data=Lsx)
	hf.create_dataset('galaxies/lqso_mb',data=Mb)
	if not jet_all:
		hf.create_dataset('galaxies/bh_accretion_ratio_hh',data=maccr_hh)
	hf.create_dataset('galaxies/id_galaxy',data=idgal)
	hf.close()

def main(model_dir,output_dir,redshift_table,subvols,obs_dir):
	zlist=(0,0.1,0.2,0.5,1.0,2.0,4.0,6.0)
	
	plt=common.load_matplotlib()
	fields={'galaxies':('id_galaxy','bh_accretion_rate_hh','bh_accretion_rate_sb','m_bh',)}
	
	for index, snapshot in enumerate(redshift_table[zlist]):
		for subv in subvols:
			hdf5_data=common.read_data(model_dir,snapshot,fields,[subv])
			prepare_data(hdf5_data,index,model_dir,snapshot,subv)

if __name__ == '__main__':
	main(*common.parse_args())

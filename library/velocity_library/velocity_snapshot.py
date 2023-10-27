import numpy as np
import readgadget
import MAS_library as MASL
import units_library as UL
import velocity_library as VL
import sys,os

########### routines ###########
# velocity_Gadget(snapshot_fname, dims, particle_type, axis, cpus)
################################

U = UL.units();  rho_crit = U.rho_crit

# dictionary for files name
name_dict = {'0' :'GAS',  '01':'GCDM',  '02':'GNU',    '04':'Gstars',
             '1' :'CDM',                '12':'CDMNU',  '14':'CDMStars',
             '2' :'NU',                                '24':'NUStars',
             '4' :'Stars',
             '-1':'matter'}


###############################################################################
# This routine computes the power spectrum of the velocity field using Gadget
# either a single species or of all
# snapshot_fname -----------> name of the Gadget snapshot
# ptype --------------------> scalar: 0-GAS, 1-CDM, 2-NU, 4-Stars, -1:ALL
# dims ---------------------> Total number of cells is dims^3 to compute the Pk
# axis ---------------------> axis along which move particles in redshift-space
# cpus ---------------------> Number of cpus to compute power spectrum
# folder_out ---------------> folder where to put outputs
def velocity_Gadget(snapshot_fname,ptype,dims,axis,cpus,MAS='CIC',
                    folder_out=None):
    
    # find folder to place output files. Default is current directory
    if folder_out is None:  folder_out = os.getcwd()

    # read relevant parameters on the header
    print('Computing velocity power spectrum...')
    head     = readgadget.header(snapshot_fname)
    BoxSize  = head.boxsize/1e3 #Mpc/h
    Masses   = head.massarr*1e10 #Msun/h
    Nall     = head.nall; Ntotal = np.sum(Nall,dtype=np.int64)
    Omega_m  = head.omega_m
    Omega_l  = head.omega_l
    redshift = head.redshift
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l) #km/s/(Mpc/h)

    # find output file name
    fout = folder_out+'/Pk_velocity_'+name_dict[str(ptype)]+'.dat'

    # read positions of the particles
    pos = readgadget.read_block(snapshot_fname,"POS ",ptype=[ptype])/1e3 #Mpc/h
    print('%.3f < X [Mpc/h] < %.3f'%(np.min(pos[:,0]),np.max(pos[:,0])))
    print('%.3f < Y [Mpc/h] < %.3f'%(np.min(pos[:,1]),np.max(pos[:,1])))
    print('%.3f < Z [Mpc/h] < %.3f'%(np.min(pos[:,2]),np.max(pos[:,2])))

    # read velocities of the particles
    vel = readgadget.read_block(snapshot_fname,"VEL ",ptype=[ptype]) #km/s

    # define velcity field array
    vx = np.zeros((dims,dims,dims), dtype=np.float32)
    vy = np.zeros((dims,dims,dims), dtype=np.float32)
    vz = np.zeros((dims,dims,dims), dtype=np.float32)

    # construct the velocity field
    #TODO: check if this field is divided by volume
    print('Constructing velocity field...')
    MASL.MA(pos, vx, BoxSize, MAS, vel[:,0])
    MASL.MA(pos, vy, BoxSize, MAS, vel[:,1])
    MASL.MA(pos, vz, BoxSize, MAS, vel[:,2])
    del pos, vel

    # compute the power spectrum
    Pk = VL.Pk_v(vx, vy, vz, BoxSize, axis=axis, MAS=MAS, threads=cpus); del vx, vy, vz
    np.savetxt(fout, np.transpose([Pk.k3D, Pk.Pk_v, Pk.Std_v, Pk.Pk_dv, Pk.Std_dv,
                                   Pk.Nmodes3D]), header="k\t <vv> \t \sigma_<vv>\t  <dvdv> \t \sigma_<dvdv> \t Nmodes", comments='#')
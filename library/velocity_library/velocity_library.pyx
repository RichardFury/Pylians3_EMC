import numpy as np
import time, sys, os
import pyfftw
import scipy.integrate as si
cimport numpy as np
cimport cython
from libc.math cimport sqrt,pow,sin,log10,abs,atan2
from libc.stdlib cimport malloc,free
from cpython cimport bool

################################ ROUTINES ####################################
# Pk_v(vel, BoxSize, axis=3, MAS='CIC', threads=1)
#   k[k]    Pk[k, ell]]    Nmodes[k] ===> <vv> = Pk[:,0]; <dv dv> = Pk[:,1];
#                                         <ww> = Pk[:,2];
##############################################################################

# This function determines the fundamental (kF) and Nyquist (kN) frequencies
# It also finds the maximum frequency sampled in the box, the maximum
# frequency along the parallel and perpendicular directions in units of kF
def frequencies(BoxSize,dims):
    kF = 2.0*np.pi/BoxSize;  middle = dims//2;  kN = middle*kF
    kmax_par = middle
    kmax_per = int(np.sqrt(middle**2 + middle**2))
    kmax     = int(np.sqrt(middle**2 + middle**2 + middle**2))
    return kF,kN,kmax_par,kmax_per,kmax

# This function finds the MAS correction index and return the array used
def MAS_function(MAS):
    MAS_index = 0;  #MAS_corr = np.ones(3,dtype=np.float64)
    if MAS=='NGP':  MAS_index = 1
    if MAS=='CIC':  MAS_index = 2
    if MAS=='TSC':  MAS_index = 3
    if MAS=='PCS':  MAS_index = 4
    return MAS_index#,MAS_corr

################################ FFT Plan ####################################
##############################################################################
# This function implement the MAS correction to modes amplitude
#@cython.cdivision(False)
#@cython.boundscheck(False)
cpdef inline double MAS_correction(double x, int MAS_index):
    return (1.0 if (x==0.0) else pow(x/sin(x),MAS_index))

# This function checks that all independent modes have been counted
def check_number_modes(Nmodes,dims):
    # (0,0,0) own antivector, while (n,n,n) has (-n,-n,-n) for dims odd
    if dims%2==1:  own_modes = 1 
    # (0,0,0),(0,0,n),(0,n,0),(n,0,0),(n,n,0),(n,0,n),(0,n,n),(n,n,n)
    else:          own_modes = 8 
    repeated_modes = (dims**3 - own_modes)//2  
    indep_modes    = repeated_modes + own_modes

    if int(np.sum(Nmodes))!=indep_modes:
        print('WARNING: Not all modes counted')
        print('Counted  %d independent modes'%(int(np.sum(Nmodes))))
        print('Expected %d independent modes'%indep_modes)
        sys.exit() 

# This function performs the 3D FFT of a field in single precision
def FFT3Dr_f(np.ndarray[np.float32_t,ndim=3] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')
    a_out = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in double precision
def FFT3Dr_d(np.ndarray[np.float64_t,ndim=3] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims),    dtype='float64')
    a_out = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex128')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in single precision
def IFFT3Dr_f(np.complex64_t[:,:,::1] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex64')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float32')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD', threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out

# This function performs the 3D FFT of a field in double precision
def IFFT3Dr_d(np.complex128_t[:,:,::1] a, int threads):

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims,dims,dims//2+1),dtype='complex128')
    a_out = pyfftw.empty_aligned((dims,dims,dims),    dtype='float64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in,a_out,axes=(0,1,2),
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_BACKWARD',threads=threads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a;  fftw_plan(a_in,a_out);  return a_out


##############################################################################
##############################################################################
# THis routine computes the power spectrum of velocity field, its
#      divergence field and vorticity field.
# vel ------------> 3D velocity field (dims, dims, dims, 3) numpy array
# BoxSize --------> size of the cubic velocity field
# axis -----------> axis along which place the line of sight
# MAS ------------> mass assignment scheme used to compute velocity field
#                   needed to correct modes amplitude
# threads --------> number of threads (OMP) used to make the FFTW
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
class Pk_v:
    def __init__(self,Vx,Vy,Vz,BoxSize,axis=2,MAS='CIC',threads=1, verbose=True):
        start = time.time()
        cdef int kxx,kyy,kzz,kx,ky,kz,dims,middle,k_index,kmax,MAS_index
        cdef double kmod,prefact,real,imag,theta2,vel2,omega2
        cdef double MAS_corr[3]
        ####### change this for double precision ######
        cdef float MAS_factor
        cdef np.complex64_t[:,:,::1] Vx_k,Vy_k,Vz_k
        ###############################################
        cdef np.float64_t[::1] k,Nmodes
        cdef np.float64_t[::1] Pk_v, Pk_dv, Pk_w, Std_v, Std_dv, Std_w

        # find dimensions of delta: we assuming is a (dims,dims,dims) array
        # determine the different frequencies, the MAS_index and the MAS_corr
        if verbose:  print('Computing the power spectrum of the velocity field...')
        dims = len(Vx);  middle = dims//2
        kF,kN,kmax_par,kmax_per,kmax = frequencies(BoxSize,dims)
        MAS_index = MAS_function(MAS)

        ## compute FFT of the field (change this for double precision) ##
        #delta_k = FFT3Dr_f(delta,threads)
        Vx_k = FFT3Dr_f(Vx,threads)
        Vy_k = FFT3Dr_f(Vy,threads)
        Vz_k = FFT3Dr_f(Vz,threads)
        #################################

        # define arrays containing the k, Pk_v, Pk_dv, Pk_w and Nmodes. We need 
        # bins since the mode (middle, middle, middle) has an index = kmax
        k       = np.zeros(kmax+1,dtype=np.float64)
        Pk_v    = np.zeros(kmax+1,dtype=np.float64)
        Pk_dv   = np.zeros(kmax+1,dtype=np.float64)
        Pk_w    = np.zeros(kmax+1,dtype=np.float64)
        Nmodes  = np.zeros(kmax+1,dtype=np.float64)
        Std_v   = np.zeros(kmax+1,dtype=np.float64)
        Std_dv  = np.zeros(kmax+1,dtype=np.float64)
        Std_w   = np.zeros(kmax+1,dtype=np.float64)

        start2 = time.time(); prefact = np.pi/dims
        for kxx in range(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)
            MAS_corr[0] = MAS_correction(prefact*kx,MAS_index)

            for kyy in range(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)
                MAS_corr[1] = MAS_correction(prefact*ky,MAS_index)

                for kzz in range(middle+1): #kzz=[0,1,...,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)
                    MAS_corr[2] = MAS_correction(prefact*kz,MAS_index)

                    # kz=0 and kz=middle planes are special
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue

                    # compute |k| of the mode and its integer part
                    kmod = sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = int(kmod)

                    # correct modes amplitude for MAS
                    MAS_factor = MAS_corr[0]*MAS_corr[1]*MAS_corr[2]
                    Vx_k[kxx,kyy,kzz] = Vx_k[kxx,kyy,kzz]*MAS_factor
                    Vy_k[kxx,kyy,kzz] = Vy_k[kxx,kyy,kzz]*MAS_factor
                    Vz_k[kxx,kyy,kzz] = Vz_k[kxx,kyy,kzz]*MAS_factor
                    #delta_k[kxx,kyy,kzz] *= MAS_factor

                    # compute the theta for each mode: theta = ik*V(k)
                    real = -(kx*Vx_k[kxx,kyy,kzz].imag + \
                             ky*Vy_k[kxx,kyy,kzz].imag + \
                             kz*Vz_k[kxx,kyy,kzz].imag)

                    imag = kx*Vx_k[kxx,kyy,kzz].real + \
                           ky*Vy_k[kxx,kyy,kzz].real + \
                           kz*Vz_k[kxx,kyy,kzz].real

                    theta2 = real*real + imag*imag

                    # compute the velocity for each mode:
                    vel2 = Vx_k[kxx,kyy,kzz].real*Vx_k[kxx,kyy,kzz].real + \
                           Vy_k[kxx,kyy,kzz].real*Vy_k[kxx,kyy,kzz].real + \
                           Vz_k[kxx,kyy,kzz].real*Vz_k[kxx,kyy,kzz].real + \
                           Vx_k[kxx,kyy,kzz].imag*Vx_k[kxx,kyy,kzz].imag + \
                           Vy_k[kxx,kyy,kzz].imag*Vy_k[kxx,kyy,kzz].imag + \
                           Vz_k[kxx,kyy,kzz].imag*Vz_k[kxx,kyy,kzz].imag

                    #TODO compute the vorticity for each mode: w = ik x V(k)

                    # add the mode to the k, Pk and Nmodes arrays
                    k[k_index]         += kmod
                    Pk_v[k_index]      += vel2
                    Pk_dv[k_index]     += theta2
                    #Pk_w[k_index]      += omega2
                    Nmodes[k_index]    += 1.0
        # Calculate the statistical error
        for kxx in range(dims):
            kx = (kxx-dims if (kxx>middle) else kxx)

            for kyy in range(dims):
                ky = (kyy-dims if (kyy>middle) else kyy)

                for kzz in range(middle+1): #kzz=[0,1,...,middle] --> kz>0
                    kz = (kzz-dims if (kzz>middle) else kzz)

                    # kz=0 and kz=middle planes are special
                    if kz==0 or (kz==middle and dims%2==0):
                        if kx<0: continue
                        elif kx==0 or (kx==middle and dims%2==0):
                            if ky<0.0: continue

                    # compute |k| of the mode and its integer part
                    kmod = sqrt(kx*kx + ky*ky + kz*kz)
                    k_index = int(kmod)

                    # compute the theta for each mode: theta = ik*V(k)
                    real = -(kx*Vx_k[kxx,kyy,kzz].imag + \
                             ky*Vy_k[kxx,kyy,kzz].imag + \
                             kz*Vz_k[kxx,kyy,kzz].imag)

                    imag = kx*Vx_k[kxx,kyy,kzz].real + \
                           ky*Vy_k[kxx,kyy,kzz].real + \
                           kz*Vz_k[kxx,kyy,kzz].real

                    theta2 = real*real + imag*imag

                    # compute the velocity for each mode:
                    vel2 = Vx_k[kxx,kyy,kzz].real*Vx_k[kxx,kyy,kzz].real + \
                           Vy_k[kxx,kyy,kzz].real*Vy_k[kxx,kyy,kzz].real + \
                           Vz_k[kxx,kyy,kzz].real*Vz_k[kxx,kyy,kzz].real + \
                           Vx_k[kxx,kyy,kzz].imag*Vx_k[kxx,kyy,kzz].imag + \
                           Vy_k[kxx,kyy,kzz].imag*Vy_k[kxx,kyy,kzz].imag + \
                           Vz_k[kxx,kyy,kzz].imag*Vz_k[kxx,kyy,kzz].imag

                    #TODO compute the vorticity for each mode: w = ik x V(k)

                    # Statistical error
                    Std_v[k_index]     += (vel2-(Pk_v[k_index]/Nmodes[k_index]))**2
                    Std_dv[k_index]    += (theta2-(Pk_dv[k_index]/Nmodes[k_index]))**2
                    #Std_w[k_index]     += (omega2-(Pk_w[k_index]/Nmodes[k_index]))**2
        if verbose:  print('Time compute modulus = %.2f'%(time.time()-start2))

        # check modes, discard fundamental frequency bin and give units
        # we need to multiply the multipoles by (2*ell + 1)
        check_number_modes(Nmodes,dims)
        #TODO: check the correctness of the power spectrum factor.
        k       = k[1:];    Nmodes = Nmodes[1:]; Pk_v = Pk_v[1:]; Pk_dv = Pk_dv[1:]
        Std_v  = Std_v[1:];   Std_dv = Std_dv[1:]
        #   Pk_w = Pk_w[1:];    Std_w = Std_w[1:]
        for i in range(len(k)):
            k[i]        = (k[i]/Nmodes[i])*kF
            Pk_v[i]     = (Pk_v[i]/Nmodes[i])*(BoxSize/dims**2)**3
            Pk_dv[i]    = (Pk_dv[i]/Nmodes[i])*(BoxSize/dims**2)**3*kF**2
            #Pk_w[i]    = (Pk_w[i]/Nmodes[i])*(BoxSize/dims**2)**3*kF**2
            Std_v[i]    = sqrt(Std_v[i]/Nmodes[i])*(BoxSize/dims**2)**3
            Std_dv[i]   = sqrt(Std_dv[i]/Nmodes[i])*(BoxSize/dims**2)**3*kF**2
            #Std_w[i]   = sqrt(Std_w[i]/Nmodes[i])*(BoxSize/dims**2)**3*kF**2
        #TODO: include the power spectrum of vorticity field.
        self.k3D = np.asarray(k);  self.Nmodes3D = np.asarray(Nmodes)
        self.Pk_v = np.asarray(Pk_v);  self.Pk_dv = np.asarray(Pk_dv)
        #self.Pk_w = np.asarray(Pk_w)
        self.Std_v = np.asarray(Std_v);  self.Std_dv = np.asarray(Std_dv)
        #self.Std_w = np.asarray(Std_w)

        if verbose:  print('Time taken = %.2f seconds'%(time.time()-start))
################################################################################
################################################################################

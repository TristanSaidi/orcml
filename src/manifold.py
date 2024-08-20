## Code from https://github.com/aghickok/curvature ##

import numpy as np
from scipy.special import gamma
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
#from mpl_toolkits import mplot3d

##############################################################################
# Sphere sampling
##############################################################################

class Sphere:
    
    def Rdist(n, x1, x2):
        # x1, x2: two points on unit n-sphere
        # output: geodesic distance between x1 and x2
        dotprod = sum([x1[i]*x2[i] for i in range(n+1)])
        Rdist = np.arccos(dotprod)
        return Rdist
        
    def Rdist_array(n, X):
        # n: sphere dimension
        # X: point cloud (rows are observations)
        # output: distance matrix
        N = X.shape[0]
        Rdist = np.zeros((N, N))
        for i in range(N):
            for j in range(i):
                x1 = X[i, :]
                x2 = X[j, :]
                Rdist[i, j] = Sphere.Rdist(n, x1, x2)
                Rdist[j, i] = Rdist[i, j]
        return Rdist
    
    def sample(N, n, noise = 0, R = 1):
         # To sample a point x, let x_i ~ N(0, 1) and then rescale to have norm R. then add isotropic Gaussian noise to x with variance noise^2
        X = []
        for i in range(N):
            x = np.random.normal(size = n+1)
            x /= np.linalg.norm(x)
            x *= R
            X.append(x)
        return np.array(X)
    
    def S2_ball_volume(r):
        # volume of geodesic ball of radius r in unit 2-sphere
        return 4*math.pi*(math.sin(r/2)**2)
    
    def unit_volume(n):
        # returns volume of Euclidean unit n-sphere
        m = n+1
        Sn = (2*(math.pi)**(m/2))/gamma(m/2)
        return Sn
    
    def S2_area(R):
        return 4 * math.pi * R**2


##############################################################################
# Euclidean sampling
##############################################################################

class Euclidean:
    
    def sample(N, n, R, Rsmall = None):
        # If Rsmall = None, sample N points in an n-ball of radius R
        # Otherwise, sample points in an n-ball of radius R until you get N points within an n-ball of radius Rsmall < R
        X = []
        if Rsmall is None:
            for i in range(N):
                x = np.random.normal(size = n)
                u = (R**n)*np.random.random()
                r = u**(1/n)
                x *= r/np.linalg.norm(x)
                X.append(x)
        else:
            Nsmall = 0
            while Nsmall < N:
                x = np.random.normal(size = n)
                u = (R**n)*np.random.random()
                r = u**(1/n)
                x *= r/np.linalg.norm(x) # now the norm of x is r
                X.append(x)
                if r < Rsmall: Nsmall += 1
                
        return np.array(X)

    def density(n, R):
        # density in a ball of radius R in R^n
        vn = (math.pi**(n/2))/gamma(n/2 + 1) # volume of Euclidean unit n-ball
        vol = vn*R**(n)
        return 1/vol

    def distance_array(X):
        N = X.shape[0]
        D = np.zeros((N, N))
        for i in range(N):
            x = X[i, :]
            for j in range(i):
                y = X[j, :]
                D[i, j] = np.linalg.norm(x - y)
                D[j, i] = D[i, j]
        return D
    
    def R2_area(r):
        return math.pi*r**2
    
##############################################################################
# Torus sampling
##############################################################################

class Torus:
    
    def exact_curvatures(thetas, r, R):
        curvatures = [Torus.S_exact(theta, r, R) for theta in thetas]
        return curvatures

    def sample(N, r, R, double=False, supersample=False, supersample_factor=2.5):
        # N: number of points, r: minor radius, R: major radius
        # double: if True, return a second torus with half of the points rotated and offset
        # supersample: if True, sample N*supersample_factor points. Likely to be used for accurate geodesic distance computation

        if supersample:
            N_total = int(N*supersample_factor)
            subsample_indices = np.random.choice(N_total, N, replace=False)
        else:
            N_total = N

        psis = [np.random.random()*2*math.pi for i in range(N_total)]
        j = 0
        thetas = []
        while j < N_total:
            theta = np.random.random()*2*math.pi
            #eta = np.random.random()*2*(r/R) + 1 - (r/R)
            #if eta < 1 + (r/R)*math.cos(theta):
            eta = np.random.random()/math.pi
            if eta < (1 + (r/R)*math.cos(theta))/(2*math.pi):
                thetas.append(theta)
                j += 1
    
        def embed_torus(theta, psi):
            x = (R + r*math.cos(theta))*math.cos(psi)
            y = (R + r*math.cos(theta))*math.sin(psi)
            z = r*math.sin(theta)
            return [x, y, z]
    
        X = np.array([embed_torus(thetas[i], psis[i]) for i in range(N_total)])

        if double:
            # randomly pick half of points to rotate and offset
            indices = np.random.choice(N_total, N_total//2, replace=False)
            for i in indices:
                # rotate by pi/2 about x-axis
                x = X[i, 0]
                y = X[i, 1]
                z = X[i, 2]
                X[i, 0] = x
                X[i, 1] = z
                X[i, 2] = -y
                # offset
                X[i, 0] += R
            # get one-hot encoding of which points were rotated
            rotated = np.zeros(N_total)
            rotated[indices] = 1
        else:
            rotated = None
        
        if supersample:
            X_supersample = X.copy()
            X = X[subsample_indices]
            thetas = [thetas[i] for i in subsample_indices]
            if rotated is not None:
                rotated = rotated[subsample_indices]
        else:
            X_supersample = None
            subsample_indices = None
        return X, np.array(thetas), rotated, X_supersample, subsample_indices
    
    def S_exact(theta, r, R):
        # Analytic scalar curvature
        S = (2*math.cos(theta))/(r*(R + r*math.cos(theta)))
        return S
    
    def theta_index(theta, thetas):
        # Returns index in thetas of the angle closest to theta
        err = [abs(theta_ - theta) for theta_ in thetas]
        return np.argmin(err)
    
    def area(r, R):
        return (2 * math.pi * r) * (2 * math.pi * R)
    
##############################################################################
# Poincare disk sampling
##############################################################################

class PoincareDisk:
    
    def sample(N, K = -1, Rh = 1):
        # N: number of points, K: Gaussian curvature
        # Rh: hyperbolic radius of the disk
        assert K < 0, "K must be negative"
        R = 1/math.sqrt(-K)
        thetas = 2*math.pi*np.random.random(N)
        us = np.random.random(N)
        C1 = 2/math.sqrt(-K)
        C2 = np.sinh(Rh*math.sqrt(-K)/2)
        rs = [C1*np.arcsinh(C2*math.sqrt(u)) for u in us]
        ts = [R*np.tanh(r/(2*R)) for r in rs]
        X = np.array([[ts[i]*math.cos(thetas[i]), ts[i]*math.sin(thetas[i])] for i in range(N)])
        return X
    
    def sample_polar(N, K = -1):
        # N: number of points
        # Gaussian curvature is K = -1
        thetas = 2*math.pi*np.random.random(N)
        us = np.random.random(N)
        C1 = 2/math.sqrt(-K)
        C2 = np.sinh(math.sqrt(-K)/2)
        rs = [C1*np.arcsinh(C2*math.sqrt(u)) for u in us]
        X = np.array([[rs[i], thetas[i]] for i in range(N)])
        return X
    
    def cartesian_to_polar(X, K = -1):
        R = 1/math.sqrt(-K)
        N = X.shape[0]
        X = []
        for i in range(N):
            x = X[i, 0]
            y = X[i, 1]
            r = 2*R*np.arctanh(math.sqrt(x**2 + y**2)/R)
            theta = np.arccos(x/(R*np.tanh(r/(2*R))))
            X.append([r, theta])
        return X
    
    def norm(x, K = -1):
        assert K < 0, "K must be negative"
        return PoincareDisk.Rdist(np.array([0, 0]), x, K = K)
        
    def polar_to_cartesian(X, K = -1):
        R = 1/math.sqrt(-K)
        N = X.shape[0]
        rs = X[:, 0]
        thetas = X[:, 1]
        ts = [R*np.tanh(r/(2*R)) for r in rs]
        X = np.array([[ts[i]*math.cos(thetas[i]), ts[i]*math.sin(thetas[i])] for i in range(N)])
        return X
    
    def Rdist(u, v, K = -1):
        assert K < 0, "K must be negative"
        R = 1/math.sqrt(-K)
        z = u/R
        w = v/R
        #wconj = np.array([w[0], -w[1]]) # conjugate of w, thought of as a complex number
        z_wconj = np.array([z[0]*w[0] + z[1]*w[1], w[0]*z[1] - z[0]*w[1]]) # product of z and w_conj, thought of as complex numbers
        dist = 2*R*np.arctanh(np.linalg.norm(z - w)/np.linalg.norm(np.array([1, 0]) - z_wconj))
        return dist
    
    def Rdist_polar(u, v):
        # u, v: tuples. polar coordinates (r, theta)
        # Gaussian curvature is K = -1
        r1 = u[0]
        theta1 = u[1]
        r2 = v[0]
        theta2 = v[1]
        return np.arccosh(np.cosh(r1)*np.cosh(r2) - np.sinh(r1)*np.sinh(r2)*np.cos(theta2 - theta1))
    
    def Rdist_array(X, K = -1, polar = False):
        # K is the Gaussian curvature of the hyperbolic plane that X is sampled from
        if polar: assert K == -1
        N = X.shape[0]
        Rdist = np.zeros((N, N))
        for i in tqdm(range(N), desc = "Computing distance matrix for Poincare disc"):
            for j in range(i):
                x1 = X[i, :]
                x2 = X[j, :]
                if polar:
                    Rdist[i, j] = PoincareDisk.Rdist_polar(x1, x2)
                else:
                    Rdist[i, j] = PoincareDisk.Rdist(x1, x2, K)
                Rdist[j, i] = Rdist[i, j]
        return Rdist
    
    def area(Rh, K = -1):
        # Rh: hyperbolic radius, K: curvature
        assert K < 0
        return (-4*math.pi/K)*(np.sinh(Rh*math.sqrt(-K)/2)**2)


##############################################################################
# Hyperboloid sampling
##############################################################################

# TO DO- organize/clean up

class Hyperboloid:
    
    def det_g(a, c, u):
        return (a**4)*(u**2) + a**2*(u**2 + 1)*c**2
    
    def sample(N, a = 2, c = 1, B = 2, within_halfB = True, double=False, supersample=False, supersample_factor=2.5):
        # if within_halfB = False, then sample N points from the hyperboloid with u in [-B, B]
        # if within_halfB = True, then sample points uniformly from u in [-B, B] until there are at least N points with u in [-.5B, .5B]
        if supersample:
            N_total = int(N*supersample_factor)
            subsample_indices = np.random.choice(N_total, N, replace=False)
        else:
            N_total = N
        sqrt_max_det_g = math.sqrt(Hyperboloid.det_g(a, c, B))
        us = []
        thetas = []
        i = 0
        while i < N_total:
            theta = 2*math.pi*np.random.random()
            u = 2*B*np.random.random() - B
            eta = sqrt_max_det_g*np.random.random()
            sqrt_det_g = math.sqrt(Hyperboloid.det_g(a, c, u))
            if eta < sqrt_det_g:
                if (within_halfB and -.5*B <= u <= .5*B) or (not within_halfB):
                    i += 1
                    us.append(u)
                    thetas.append(theta)
       
        xs = [a*math.cos(thetas[i])*math.sqrt(u**2 + 1) for i, u in enumerate(us)]
        ys = [a*math.sin(thetas[i])*math.sqrt(u**2 + 1) for i, u in enumerate(us)]
        zs = [c*u for i, u in enumerate(us)]

        X = np.array([[x, ys[i], zs[i]] for i, x in enumerate(xs)])
        if double:
            # scale half of points to create a second hyperboloid
            indices = np.random.choice(N_total, N_total//2, replace=False)
            for i in indices:
                X[i, 0] *= 0.6
                X[i, 1] *= 0.6
            # get one-hot encoding of which points were scaled
            scaled = np.zeros(N_total)
            scaled[indices] = 1
        else:
            scaled = None
        if supersample:
            X_supersample = X.copy()
            X = X[subsample_indices]
            scaled = scaled[subsample_indices]
        else:
            X_supersample = None
            subsample_indices = None
        return X, scaled, X_supersample, subsample_indices

    def area(a, c, B):
        alpha = math.sqrt(c**2 + a**2)/(c**2)
        cBalpha = c*B*alpha
        return 2*math.pi*a*(math.sqrt(cBalpha**2 + 1)*cBalpha + np.arcsinh(cBalpha))/alpha
    
    def S(z):
        # actual scalar curvature at z when a = b = 2 and c = 1
        return -2/((5*z**2 + 1)**2)
    

##############################################################################
# Cassini sampling
##############################################################################

class Cassini:
    
    def sample(N, e=1.01, a=1):
        def r(theta):
            return a*np.sqrt(np.cos(2*theta) + np.sqrt(e**4 - 1 + np.cos(2*theta)**2))
        angles = 2*np.pi*np.random.rand(N)
        Xpolar = [[theta, r(theta)] for theta in angles]
        X = np.array([[x[1]*np.cos(x[0]), x[1]*np.sin(x[0])] for x in Xpolar])
        return X, None
    
##############################################################################
# Paraboloid sampling
##############################################################################

class Paraboloid:

    def sample(N, r=1, z_max=2, offset=[0.0, 0.0, 0.0]):
        # sample polar coordinates
        thetas = 2*np.pi*np.random.rand(N)
        r = r*np.random.rand(N)

        # convert to Cartesian
        X = np.array([[r[i]*np.cos(thetas[i]) + offset[0], r[i]*np.sin(thetas[i]) + offset[1], z_max * r[i]**2 + offset[2]] for i in range(N)])
        return X, None
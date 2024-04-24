import numpy as np
from numpy.polynomial.legendre import leggauss
from .quadrature import lglnodes,equispaced

class DeC:
    def __init__(self, M_sub, n_iter, nodes_type):
        """ Deffered Correction time intergration solver

        Args:
            M_sub (_type_): Number of subtimesteps (subtimenodes -1)
            n_iter (_type_): Number of iterations
            nodes_type (_type_): Type of node distribution
        """
        self.n_subNodes = M_sub+1
        self.M_sub = M_sub
        self.n_iter = n_iter
        self.nodes_type = nodes_type
        self.compute_theta_DeC()
        self.name = f"DeC_{self.nodes_type}"
    
    def compute_theta_DeC(self):
        nodes, w = get_nodes(self.n_subNodes,self.nodes_type)
        int_nodes, int_w = get_nodes(self.n_subNodes,"gaussLobatto")
        # generate theta coefficients 
        self.theta = np.zeros((self.n_subNodes,self.n_subNodes))
        self.beta = np.zeros(self.n_subNodes)
        for m in range(self.n_subNodes):
            self.beta[m] = nodes[m]
            nodes_m = int_nodes*(nodes[m])
            w_m = int_w*(nodes[m])
            for r in range(self.n_subNodes):
                self.theta[r,m] = sum(lagrange_basis(nodes,nodes_m,r)*w_m)
        return self.theta, self.beta
    
    
    def compute_RK_from_DeC(self):
        bar_beta=self.beta[1:]  # M_sub
        bar_theta=self.theta[:,1:].transpose() # M_sub x (M_sub +1)
        theta0= bar_theta[:,0]  # M_sub x 1
        bar_theta= bar_theta[:,1:] #M_sub x M_sub
        self.ARK=np.zeros((self.M_sub*(self.n_iter-1)+1,self.M_sub*(self.n_iter-1)+1))  # (M_sub x K_corr +1)^2
        self.bRK=np.zeros(self.M_sub*(self.n_iter-1)+1)
        self.cRK=np.zeros(self.M_sub*(self.n_iter-1)+1)

        self.cRK[1:self.M_sub+1]=bar_beta
        self.ARK[1:self.M_sub+1,0]=bar_beta
        for k in range(1,self.n_iter-1):
            r0=1+self.M_sub*k
            r1=1+self.M_sub*(k+1)
            c0=1+self.M_sub*(k-1)
            c1=1+self.M_sub*(k)
            self.cRK[r0:r1]=bar_beta
            self.ARK[r0:r1,0]=theta0
            self.ARK[r0:r1,c0:c1]=bar_theta
        self.bRK[0]=theta0[-1]
        self.bRK[-self.M_sub:]=bar_theta[self.M_sub-1,:]
        return self.ARK,self.bRK,self.cRK
    
    def compute_RK_from_DeCImplicit(self):
        self.compute_RK_from_DeC()
        self.ARKIM=np.zeros((self.M_sub*(self.n_iter-1)+2,self.M_sub*(self.n_iter-1)+2))  # (M_sub x K_corr +1)^2
        self.bRKIM=np.zeros(self.M_sub*(self.n_iter-1)+2)
        self.cRKIM=np.zeros(self.M_sub*(self.n_iter-1)+2)
        bar_beta=self.beta[1:]  # M_sub
        bar_theta=self.theta[:,1:].transpose() # M_sub x (M_sub +1)
        bar_theta= bar_theta[:,1:] #M_sub x M_sub
        theta0= bar_theta[:,0]  # M_sub x 1
        self.ARKIM[:-1,:-1]=self.ARK  # (M_sub x K_corr +1)^2
        self.bRKIM[:-1]=self.bRK
        self.cRKIM[:-1]=self.cRK

        self.cRK[1:self.M_sub+1]=bar_beta
        self.ARK[1:self.M_sub+1,0]=bar_beta
        k=0
        r0=1+self.M_sub*k
        r1=1+self.M_sub*(k+1)
        c0=0
        c1=1+self.M_sub*(k)
        c2=1+self.M_sub*(k+1)
        self.ARKIM[r0:r1,c1:c2] = np.diag(bar_beta)
        self.ARKIM[r0:r1,c0:c1] = self.ARKIM[r0:r1,c0:c1] - bar_beta.reshape((r1-r0,c1-c0))
        for k in range(1,self.n_iter-1):
            r0=1+self.M_sub*k
            r1=1+self.M_sub*(k+1)
            c0=1+self.M_sub*(k-1)
            c1=1+self.M_sub*(k)
            c2=1+self.M_sub*(k+1)
            self.ARKIM[r0:r1,c1:c2] = np.diag(bar_beta) 
            self.ARKIM[r0:r1,c0:c1] = self.ARKIM[r0:r1,c0:c1] - np.diag(bar_beta) 
        self.ARKIM[-1,:-1] = self.bRK
        self.ARKIM[-1,-2] = self.ARKIM[-1,-2] - bar_beta[-1]
        self.ARKIM[-1,-1] = bar_beta[-1]
        self.cRKIM[-1] = bar_beta[-1]
        self.bRKIM = self.ARKIM[-1,:]
        return self.ARKIM,self.bRKIM,self.cRKIM
    
    def dec(self, func, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        rhs= np.zeros((dim,self.M_sub+1))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            t_sub = tspan[it-1]+ delta_t*self.beta
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            rhs[:,0] = func(U[:,it-1],t_sub[0])
            for r in range(1,self.M_sub+1):
                rhs[:,r] = rhs[:,0]
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                if k>1:
                    for r in range(1,self.M_sub+1):
                        rhs[:,r]=func(u_p[:,r],t_sub[r])
                if k < self.n_iter:
                    for m in range(1,self.M_sub+1):
                        u_a[:,m]= U[:,it-1]+delta_t*sum([self.theta[r,m]*rhs[:,r] for r in range(self.M_sub+1)])
                else:
                    u_a[:,self.M_sub]= U[:,it-1]+delta_t*sum([self.theta[r,self.M_sub]*rhs[:,r] for r in range(self.M_sub+1)])
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U

    def decImplicit(self, func,jac_stiff, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        u_help= np.zeros(dim)
        rhs= np.zeros((dim,self.M_sub+1))
        invJac=np.zeros((self.M_sub+1,dim,dim))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])            
            t_sub = tspan[it-1]+ delta_t*self.beta
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            SS=jac_stiff(u_p[:,0])
            for m in range(1,self.M_sub+1):
                invJac[m,:,:]=np.linalg.inv(np.eye(dim) - delta_t*self.beta[m]*SS)
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                for r in range(self.M_sub+1):
                    rhs[:,r]=func(u_p[:,r],t_sub[r])
                for m in range(1,self.M_sub+1):
                    u_a[:,m]= u_p[:,m]+delta_t*np.matmul(invJac[m,:,:],\
                    (-(u_p[:,m]-u_p[:,0])/delta_t\
                     +sum([self.theta[r,m]*rhs[:,r] for r in range(self.M_sub+1)])))
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U



    def decMPatankar(self, prod_dest, rhs, tspan, y_0):
        N_time=len(tspan)
        dim=len(y_0)
        U=np.zeros((dim, N_time))
        u_p=np.zeros((dim, self.M_sub+1))
        u_a=np.zeros((dim, self.M_sub+1))
        prod_p = np.zeros((dim,dim,self.M_sub+1))
        dest_p = np.zeros((dim,dim,self.M_sub+1))
        rhs_p= np.zeros((dim,self.M_sub+1))
        U[:,0]=y_0
        for it in range(1, N_time):
            delta_t=(tspan[it]-tspan[it-1])
            for m in range(self.M_sub+1):
                u_a[:,m]=U[:,it-1]
                u_p[:,m]=U[:,it-1]
            for k in range(1,self.n_iter+1):
                u_p=np.copy(u_a)
                for r in range(self.M_sub+1):
                    prod_p[:,:,r], dest_p[:,:,r]=prod_dest(u_p[:,r])
                    rhs_p[:,r]=rhs(u_p[:,r])
                for m in range(1,self.M_sub+1):
                    u_a[:,m]= patankar_type_dec(prod_p,dest_p,rhs_p,delta_t,m,self.M_sub,self.theta,u_p,dim)
            U[:,it]=u_a[:,self.M_sub]
        return tspan, U
    


def lagrange_basis(nodes,x,k):
    y=np.zeros(x.size)
    for ix, xi in enumerate(x):
        tmp=[(xi-nodes[j])/(nodes[k]-nodes[j])  for j in range(len(nodes)) if j!=k]
        y[ix]=np.prod(tmp)
    return y

def get_nodes(order,nodes_type):
    if nodes_type=="equispaced":
        nodes,w = equispaced(order)
    elif nodes_type == "gaussLegendre":
        nodes,w = leggauss(order)
    elif nodes_type == "gaussLobatto":
        nodes, w = lglnodes(order-1,10**-15)
    nodes=nodes*0.5+0.5
    w = w*0.5
    return nodes, w
        



def patankar_type_dec(prod_p,dest_p,rhs_p,delta_t,m,M_sub,Theta,u_p,dim):
    mass= np.eye(dim)
    RHS= u_p[:,0]
    for i in range(dim):
        for r in range(M_sub+1):
            RHS[i]=RHS[i]+delta_t*Theta[r,m]*rhs_p[i,r]
            if Theta[r,m]>0:
                for j in range(dim):
                    mass[i,j]=mass[i,j]-delta_t*Theta[r,m]*(prod_p[i,j,r]/u_p[j,m])
                    mass[i,i]=mass[i,i]+ delta_t*Theta[r,m]*(dest_p[i,j,r]/u_p[i,m])
            elif Theta[r,m]<0:
                for j in range(dim):
                    mass[i,i]=mass[i,i]- delta_t*Theta[r,m]*(prod_p[i,j,r]/u_p[i,m])
                    mass[i,j]=mass[i,j]+ delta_t*Theta[r,m]*(dest_p[i,j,r]/u_p[j,m])
    return np.linalg.solve(mass,RHS)

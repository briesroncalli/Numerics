using FFTW, Plots, LinearAlgebra, DelimitedFiles

function nonlinear(nuvec, deltat, rk4N, kvec, FPlan, BPlan)
    """
    solves u_t + 2uu_x = 0 (Burger's eqn)
    uses RK4Vector
    """
    deltatrk4 = deltat / rk4N #resizing time step for RK4
    return rk4vector(nuvec, deltatrk4, rk4N, kvec, FPlan, BPlan)
end

function rk4vector(nuvec, deltat, rk4N, kvec, FPlan, BPlan)

    for n=1:rk4N
        k1 = F(nuvec, kvec, FPlan, BPlan)
        k2 = F(nuvec .+ 0.5 * deltat .* k1, kvec, FPlan, BPlan)
        k3 = F(nuvec .+ 0.5 * deltat .* k2, kvec, FPlan, BPlan)
        k4 = F(nuvec .+ deltat .* k3, kvec, FPlan, BPlan)
        nuvec .= nuvec .+ (deltat/6) .* (k1 .+ 2 .*(k2 + k3).+ k4)
    end
      
    return nuvec
end

function F(nuvec, kvec, FPlan, BPlan) #right hand side of differential eqn
    """"
    n_t = -(nu)_x
    u_t = -uu_x
    """
    n = nuvec[:, 1]
    u = nuvec[:, 2]
    u_x = deriv(u, kvec, FPlan, BPlan)
    u_t = -u .* u_x

    nuproduct = n .* u
    nuproduct_x = deriv(nuproduct, kvec, FPlan, BPlan)
    n_t = -nuproduct_x
    
    return [n_t u_t]
end

function deriv(u, kvec, FPlan, BPlan)
    """
    takes derivative of u via Fourier transform
    """
    uhat = FPlan * u
    uhat = uhat .* kvec
    return real(BPlan * uhat)
end

function linear(nuvec, deltat, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    """
    nhat_t + h0 * tanh(k) * i * uhat = 0 
    uhat_t + g * i * k * nhat = 0
    """
    nhat = FPlan * nuvec[:,1]
    uhat = FPlan * nuvec[:,2]

    nuhatvec = [nhat uhat]
    nuhatvecswap = [uhat nhat]

    dtkgtanhkvec = kgtanhkvec .* deltat

    for j=1:Nd2p1
        solvedhats[:,j] = nuhatvec[j,:] .* cos.(dtkgtanhkvec[j]) .- nuhatvecswap[j,:] .* position2[:,j] .* sin.(dtkgtanhkvec[j])
    end

    solvedn = BPlan * solvedhats[1,:]
    solvedu = BPlan * solvedhats[2,:]
    return [solvedn solvedu]
end

function initializer(h0, g, N, L)
    Nd2 = div(N, 2) 
    Nd2p1 = Nd2 + 1

    realkvec = [0:Nd2;] * (2.0 * pi / L) #vector of k values (multiplied by 2.0 * pi / L for deriv?)
    kvec = realkvec * im #made imaginary; left out earlier for evals

    gtanhkvec = sqrt.( ( g .* tanh.( h0 * realkvec) ) ./ (realkvec) )
    kgtanhkvec = realkvec .* gtanhkvec
    kgtanhkvec[1] = 0.0

    etaterm = sqrt.( tanh.(h0 .* realkvec) ./ (g .* realkvec) ) *im
    uterm = kgtanhkvec .* coth.(h0 .* realkvec) *im #should be kgtanhkvec??

    position2 = [etaterm uterm]
    position2 = position2'
    position2[:,1] = [1;0]

    solvedhats = zeros(Complex{Float64}, (2,Nd2p1))

    FPlan  = plan_rfft(u)
    BPlan  = plan_irfft(zeros(Complex{Float64}, Nd2p1), N)

    return kgtanhkvec, position2, solvedhats, FPlan, BPlan, Nd2p1, kvec
end

function CQ1(u)
    return sum(u) / length(u)
end

function CQ2(eta)
    return real(sum(eta)) / length(eta)
end

function CQ3(eta, u)
    eta_u = eta .* u
    return sum(eta_u) / length(eta_u)
end

function CQ4(eta, u, h0, kappasqd, g, FPlan, BPlan)
    uhat = FPlan * u
    uhatkappasqd = kappasqd .* uhat
    uconv = real(BPlan * uhatkappasqd)
    t1 = g * eta.^2
    t2 = h0 * u .* uconv 
    t3 = eta .* u .^2
    integrand = t1 .+ t2 .+ t3
    println(0.5 * sum(integrand) / length(integrand))
    return 0.5 * sum(integrand) / length(integrand)
end

function solnchecker(initialvec, finalvec, h0, g, kappasqd, N, FPlan, BPlan)
    Np1 = N + 1

    eta_i = initialvec[1:N]
    u_i = initialvec[Np1:end]

    eta_f = finalvec[1:N]
    u_f = finalvec[Np1:end]

    CQ1_error = abs(CQ1(u_i) - CQ1(u_f))
    CQ2_error = abs(CQ2(eta_i) - CQ2(eta_f))
    CQ3_error = abs(CQ3(eta_i, u_i) - CQ3(eta_f, u_f))
    CQ4_error = abs(CQ4(eta_i, u_i, h0, kappasqd, g, FPlan, BPlan) - CQ4(eta_f, u_f, h0, kappasqd, g, FPlan, BPlan))   

    println("The CQ1 error is ", CQ1_error)
    println("The CQ2 error is ", CQ2_error)
    println("The CQ3 error is ", CQ3_error)
    println("The CQ4 error is ", CQ4_error)
end

function OSsixth(nuvec, deltat, Q, h0, g, N, L)
    w3 = 0.784513610477560
    w2 = 0.235573213359357
    w1 = -1.17767998417887
    w0 = 1.0 - 2.0 * (w1 + w2 + w3)
    deltatd2 = deltat / 2.0

    kgtanhkvec, position2, solvedhats, FPlan, BPlan, Nd2p1, kvec = initializer(h0, g, N, L)

    len = length(nuvec[:,1])
    
    factor = 1
    mark = 1

    #soln_matrix = zeros(Float64, (Q+1, 2*len))
    soln_matrix = zeros(Float64, (div(Q,factor)+1, 2*len))

    soln_matrix[1, :] = hcat(nuvec[:,1], nuvec[:,2])

    nuvec = linear(nuvec, w3*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    for i=1:Q-1
        nuvec = nonlinear(nuvec, w3*deltat, rk4N, kvec, FPlan, BPlan)
        nuvec = linear(nuvec, (w3+w2)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
        nuvec = nonlinear(nuvec, w2*deltat, rk4N, kvec, FPlan, BPlan)
        nuvec = linear(nuvec, (w2+w1)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
        nuvec = nonlinear(nuvec, w1*deltat, rk4N, kvec, FPlan, BPlan)
        nuvec = linear(nuvec, (w1+w0)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
        nuvec = nonlinear(nuvec, w0*deltat, rk4N, kvec, FPlan, BPlan)
        nuvec = linear(nuvec, (w0+w1)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
        nuvec = nonlinear(nuvec, w1*deltat, rk4N, kvec, FPlan, BPlan)
        nuvec = linear(nuvec, (w1+w2)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
        nuvec = nonlinear(nuvec, w2*deltat, rk4N, kvec, FPlan, BPlan)
        nuvec = linear(nuvec, (w2+w3)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
        nuvec = nonlinear(nuvec, w3*deltat, rk4N, kvec, FPlan, BPlan)
        nuvec = linear(nuvec, w3*deltat, FPlan, BPlan, kgtanhkvec,  position2, solvedhats, Nd2p1)
        
        
        #soln_matrix[i+1,:] = hcat(nuvec[:,1], nuvec[:,2])
        if mod(i, Q/8) == 0
            println(mark, "/8 of the way done")
            mark += 1
        end
        if mod(i, factor) == 0
            soln_matrix[div(i,factor)+1,:] = hcat(nuvec[:,1], nuvec[:,2])
        end
    end
    
    nuvec = nonlinear(nuvec, w3*deltat, rk4N, kvec, FPlan, BPlan)
    nuvec = linear(nuvec, (w3+w2)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    nuvec = nonlinear(nuvec, w2*deltat, rk4N, kvec, FPlan, BPlan)
    nuvec = linear(nuvec, (w2+w1)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    nuvec = nonlinear(nuvec, w1*deltat, rk4N, kvec, FPlan, BPlan)
    nuvec = linear(nuvec, (w1+w0)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    nuvec = nonlinear(nuvec, w0*deltat, rk4N, kvec, FPlan, BPlan)
    nuvec = linear(nuvec, (w0+w1)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    nuvec = nonlinear(nuvec, w1*deltat, rk4N, kvec, FPlan, BPlan)
    nuvec = linear(nuvec, (w1+w2)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    nuvec = nonlinear(nuvec, w2*deltat, rk4N, kvec, FPlan, BPlan)
    nuvec = linear(nuvec, (w2+w3)*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    nuvec = nonlinear(nuvec, w3*deltat, rk4N, kvec, FPlan, BPlan)
    nuvec = linear(nuvec, w3*deltatd2, FPlan, BPlan, kgtanhkvec, position2, solvedhats, Nd2p1)
    
    soln_matrix[end,:] = hcat(nuvec[:,1], nuvec[:,2])
    
    realkvec = kvec *-im
    #println(realkvec)
    kappasqd = tanh.(h0 * realkvec) ./ (h0 * realkvec)
    kappasqd[1] = 1.0
    solnchecker(soln_matrix[1,:], soln_matrix[end,:], h0, g, kappasqd, N, FPlan, BPlan)

    return soln_matrix
    #return nuvec
end

h0 = 100 # depth of tank when water flat (look up depth of faraday wave tanks in cm)
g = 981 # gravity (in cm/s^2)
k = 0.05 # wave amplitude
tf = 8 # final time of approximation
rk4N = 1 # number of steps used for RK4 routine
N = 1024 # spatial resolution #1024*2
Q = 200*5*2 # temporal resolution #200*5
deltat = tf / Q # size of step taken forward in time (in sec) #1/250
L = 1000 # period of IC


# add delta*u_xx where delta gets scaled to add dissipation 
# add to u and not h & then h and not u & maybe both?

# pass super high freq filter to prevent explosion (the less you have to cut out the better); will take out energy & thus make CQ worse

x = [0:N-1;] * L/N #discretization of spatial period
u = zeros(Float64, N) #u is velocity
n = zeros(Float64, N) #n is eta; is surface displacement

for j=1:N
    x[j] = (j-1.0) * L/N #x values of spatial period discretization
    u[j] = 0*k * cos(x[j] * 2*pi / L) #defining u on x discretization
    n[j] = k * cos(x[j] * 2*pi / L) #defining n on eta on x discretization
end
"""
#this gives explodey tv static
for j=1:N
    x[j] = (j - 1.0) * L/N #x values of spatial period discretization
    u[j] = k*0  #defining u on x discretization
    n[j] = -k*sech(x[j]-L/2)  #defining n on eta on x discretization
end
"""
nuvec = [n u] #puts eta and u in array for use in functions

fullsoln = OSsixth(nuvec, deltat, Q, h0, g, N, L)

writedlm("ASMPSolutionBen.out", fullsoln)
fullsoln
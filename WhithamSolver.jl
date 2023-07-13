using FFTW, Elliptic, Plots, DelimitedFiles

function OSsixth(u, deltat, Q, kvec, ktanh, RK4N, ForwardPlan, BackwardPlan)
    w3 = 0.784513610477560
    w2 = 0.235573213359357
    w1 = -1.17767998417887
    w0 = 1.0 - 2.0 * (w1 + w2 + w3)
    deltatd2 = deltat / 2.0

    len = length(u)
    u_matrix = zeros(Float64, (div(Q,2)+1, len))

    u_matrix[1,:] .= u

    u = linear(u, w3*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    for i=1:Q-1
        u = nonlinear(u, w3*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
        u = linear(u, (w3+w2)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
        u = nonlinear(u, w2*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
        u = linear(u, (w2+w1)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
        u = nonlinear(u, w1*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
        u = linear(u, (w1+w0)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
        u = nonlinear(u, w0*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
        u = linear(u, (w0+w1)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
        u = nonlinear(u, w1*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
        u = linear(u, (w1+w2)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
        u = nonlinear(u, w2*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
        u = linear(u, (w2+w3)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
        u = nonlinear(u, w3*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
        u = linear(u, w3*deltat, ktanh, ForwardPlan, BackwardPlan)
        
        if mod(i, 2) == 0
            u_matrix[div(i,2)+1,:] .= u
        end
    end

    u = nonlinear(u, w3*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
    u = linear(u, (w3+w2)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    u = nonlinear(u, w2*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
    u = linear(u, (w2+w1)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    u = nonlinear(u,w1*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
    u = linear(u, (w1+w0)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    u = nonlinear(u, w0*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
    u = linear(u, (w0+w1)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    u = nonlinear(u, w1*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
    u = linear(u, (w1+w2)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    u = nonlinear(u, w2*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
    u = linear(u, (w2+w3)*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    u = nonlinear(u, w3*deltat, RK4N, kvec, ForwardPlan, BackwardPlan)
    u = linear(u, w3*deltatd2, ktanh, ForwardPlan, BackwardPlan)
    
    u_matrix[end,:] .= u
    return u_matrix
end

function nonlinear(u0, deltat, rk4N, kvec, FPlan, BPlan)
    """
    solves u_t + 2uu_x = 0 (Burger's eqn)
    uses RK4Vector
    """
    deltatrk4 = deltat / rk4N
    return rk4vector(u0, deltatrk4, rk4N, kvec, FPlan, BPlan)
end

function linear(u, deltat,  ktanh, FPlan, BPlan)
    """
    uses FFT & IFFT from FFTW
    """
    u_hat = FPlan * u
    u_hat = u_hat .* exp.(ktanh * deltat) 
    return real(BPlan * u_hat) 
end

function deriv(u, kvec, FPlan, BPlan)
    uhat = FPlan * u
    uhat = uhat .* kvec
    return real(BPlan * uhat)
end
  
function F(u, kvec, FPlan, BPlan) #right hand side of differential eqn
    u_x = deriv(u, kvec, FPlan, BPlan)
    return -2 .* u .* u_x
end

function rk4vector(u, deltat, rk4N, kvec, FPlan, BPlan)
    for n=1:rk4N
        k1 = F(u, kvec, FPlan, BPlan)
        k2 = F(u + 0.5 * deltat .* k1, kvec, FPlan, BPlan)
        k3 = F(u .+ 0.5 * deltat .* k2, kvec, FPlan, BPlan)
        k4 = F(u .+ deltat .* k3, kvec, FPlan, BPlan)
        u += (deltat/6) .* (k1 .+ 2 .*(k2 + k3).+ k4)
    end
      
    return u
end

function CQ1(u)
    return real(sum(u)) / length(u)
end

function CQ2(u)
    return real(sum(u.^2)) / length(u)
end

function solnchecker(u_initial, u_final)
    CQ1_error = abs(CQ1(u_initial) - CQ1(u_final))
    CQ2_error = abs(CQ2(u_initial) - CQ2(u_final))
    #CQ3_error = abs(CQ3(u_initial, kvec, FPlan, BPlan) - CQ3(u_final, kvec, FPlan, BPlan))

    println("The CQ1 error is ", CQ1_error)
    println("The CQ2 error is ", CQ2_error)
    #println("The CQ3 error is ", CQ3_error)
end

Q = 76 #USE EVEN
RK4N = 4

file = readdlm("s628.out", Float64)
sz = size(file)
len = sz[1]

RowChoice = 15

L = file[RowChoice, 1] #spatial period
c = file[RowChoice, 2] #wave speed
H = file[RowChoice, 3] #wave height
deltat = L / c / Q
u = file[RowChoice, 4:end]
N = length(u)
Nd2 = div(N, 2)
Nd2p1 = Nd2 + 1

ForwardPlan  = plan_rfft(u)
BackwardPlan  = plan_irfft(zeros(Complex{Float64}, Nd2p1), N)
realkvec = collect(0:Nd2) * (2.0 * pi / L)
tanhkvec = sqrt.( tanh.(realkvec) ./ realkvec )
tanhkvec[1] = 1.0
kvec = im*realkvec
ktanh = -kvec .* tanhkvec

soln = OSsixth(u, deltat, Q, kvec, ktanh, RK4N, ForwardPlan, BackwardPlan)

writedlm("WhithamSolution.out", soln)
solnchecker(u, soln[end,:])

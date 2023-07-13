 using BenchmarkTools
 using Plots

#we have y1'=y2 and y2'=-f(t)y1 from theta''+f(t)theta=0 and y1=theta; y2=theta'

function F(x, y) #right hand side of differential eqn
  Fmatrix = [y[2]; -om0^2 * (1+eps * cos(2 * om * x)) * y[1]] #insert each differential eqn into matrix
  return Fmatrix
end

function k1(x0, yn, h) #computes k1s
  return h .* F(x0, yn)
end
    
function k2(x0, yn, h) #computes k2s
  return h .* F(x0 .+ h/2, yn + 0.5 .* k1(x0, yn, h))
end

function k3(x0, yn, h) #computes k3s
  return h .* F(x0 .+ h/2, yn .+ 0.5 .* k2(x0, yn, h))
end

function k4(x0, yn, h) #computes k4s
  return h .* F(x0 .+ h, yn .+ k3(x0, yn, h))
end

function yn(x, y, h) #computes next y values
  return y .+ (1/6) .* (k1(x, y, h) .+ 2 .* k2(x, y, h) .+ 2 .* k3(x, y, h) .+ k4(x, y, h)) #updates y value; the y_n+1
end

function rk4vector(x0, xf, y0, N)
    h = (xf-x0) / N #computes step size (Note: N changes later so has to be computed here)
    a = size(y0)
    len = a[1]
    lenp1 = len + 1

    Np1 = N + 1 # added this because I included the ICs in array so needed to add 1 to compensate

    M = zeros(Np1, len+1) #defines array of zeros w/ correct dimensions
    
    M[1,1] = x0

    for n in 2:Np1 #fills x column w/ values of x at each step
        M[n, 1] = M[n-1,1] + h
    end

    for n in 1:len #fills first row y0 values
        M[1, n+1] = y0[n]
    end

    for n in 1:N #loop for filling in M
        M[n+1, 2:lenp1] = yn(M[n, 1], M[n, 2:lenp1], h) #calls rk4 on each row of M
    end
    
    return M
end

eps = 0.0004
om0 = 1
om = 1

a = rk4vector(0, 100*pi, [1, 0], 100000)


#f= exp.(0.0005*eps.*a[:,1].^2)
plot(a[:,1], a[:,2], xlabel = "Time", ylabel = "theta")
#plot!(a[:,1],f)
#plot(a[:,2], [a[:,3]], xlabel = "Theta", ylabel = "Theta'")
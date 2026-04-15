from matplotlib import pyplot as plt
import numpy as np
import rk2_pendulum

def main():
    lyapunov_extraction()
    pass

def lyapunov_extraction(): #quantatively finds the lyapunov exponent (plotting difference, best fit)
   #defining initial conditions
   fDriving = 1.2
   fDamping = .5
   omegaDriving = 2/3
   pendulumLength = 9.8
   simulationTime = 60
   theta_i = 2
   theta_n = 2.0001
   
   #run runge-kutta once
   pendulum1 = rk2_pendulum.Pendulum(fDriving, fDamping, omegaDriving)
   pendulum1.set_length(pendulumLength)
   pendulum1.run_rk2(simulationTime, th_initial=theta_i) # run our rk2 simulation, specifying our initial theta
   
   theta_1 = pendulum1.get_theta_array()

   #run runge-kutta second time
   pendulum2 = rk2_pendulum.Pendulum(fDriving, fDamping, omegaDriving)
   pendulum2.set_length(pendulumLength)
   pendulum2.run_rk2(simulationTime, th_initial=theta_n) # run our rk2 simulation, changing initial theta slightly

   theta_2 = pendulum2.get_theta_array()

   #obtain difference in angle (per t)
   theta_diff = []
   log_theta_diff = []
   for i in range(0,len(theta_1)):
      diff = np.abs(theta_1[i] - theta_2[i])
      theta_diff.append(diff)
      log_diff = np.log(diff)
      log_theta_diff.append(log_diff)
   
   #make best fit, calculate gradient --> this is the exponent
   time_array = pendulum1.get_time_array()
   
   start = int(0.05 * len(time_array))
   end   = int(0.6  * len(time_array))

   t_fit   = time_array[start:end]
   log_fit = log_theta_diff[start:end]

   # numpy polyfit — degree 1 = straight line
   coeffs = np.polyfit(t_fit, log_fit, 1)
   lyapunov = coeffs[0]   # slope = λ
   #intercept = coeffs[1]

   #plot angle difference against time
   plt.xlim(0, simulationTime)
   plt.plot(time_array, log_theta_diff, label="ln|Δθ|", color="steelblue", linewidth=0.8)
   plt.plot(t_fit, np.polyval(coeffs, t_fit),
            'r--', linewidth=2, label=f"Best fit  λ = {lyapunov:.4f}")
   plt.plot(time_array,log_theta_diff)
   plt.title(f"Lyapunov exponent λ, $F_d$=1.2, q=0.5, $θ_i$=2.0000, $θ_n$=2.0001")
   plt.xlabel("Time (s)")
   plt.ylabel("ln|Δθ|")
   plt.legend()
   plt.show()



#main
if __name__ == "__main__":
  main()
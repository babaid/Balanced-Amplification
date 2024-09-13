### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d7ff42de-1f3b-473b-b734-36ecf866b60c
using Pkg; Pkg.activate(".");

# ╔═╡ e31d26ee-7970-11ed-082b-c3726a3719b6
using LinearAlgebra, PlutoUI, CairoMakie, DifferentialEquations;

# ╔═╡ 176295f7-36d0-4012-a1fe-5e9d1b149a36
md"""
# Theory

The paper discusses idea of balanced amplification using a linear firing rate model.
The model is described through following equation:

$T\frac{d\vec{\textbf{r}}}{dt} = - \vec{\textbf{r}} + \textbf{W} \vec{\textbf{r}} + \textbf{I} = -(\textbf{1} -\textbf{W})\vec{\textbf{r}} + \textbf{I}$

With $$\vec{\textbf{r}}\in \mathbb{R}^N$$ representing the firing rates of N neurons, $$\textbf{W}\in\mathbb{R}^{N\times N}$$ a connectivity matrix, and $$\textbf{I}$$ being the feedforward input from 'external' neurons. 


The paper proceeds by analysing the simples example: $\vec{\textbf{r}} = \begin{bmatrix} r_E \\ r_I\end{bmatrix}$ and
$\textbf{W} = \begin {bmatrix} w & -k_I w\\w &- k_I w\\ \end{bmatrix}$
Here $$r_E$$ is the excitatory and $$r_I$$ the inhibitory input.

After clarifying these concepts the paper proceeds to plot the solutions of these differential equations for different values of k, l and the respective time constant, showing that for certain values the input gets amplified regardless of strong inhibition.

This notebook derives these parts of the results shown in the paper, to be precise, figures 1 and 2.

In the next cell the w for balanced amplification can be seen/used.
"""



# ╔═╡ e8f819a1-5af2-4172-947b-f9fdba81a77f
wb = 4+ 2/7

# ╔═╡ 84f095e9-bc9b-47da-a2c5-60852d7715cc
md"""
### Parameters
Changing these parameters accordingly to the descriptions in the paper leads to the theoretical plots shown.

Some values mentioned in the paper are $w\in [0.75, 4\frac{2}{7}, 3/4, 0.9, 2.5, 90]$, $k=1.1$. 

The boundary conditions are the following: 
$\vec{\textbf{r}}(0) = \begin{bmatrix} r_E(0) \\ r_I(0)\end{bmatrix} = \begin{bmatrix} 1 \\ 0\end{bmatrix}$.

Synaptic weight (w)   : $(@bind w PlutoUI.Slider(0.:0.01:4.28, show_value=true)) \
Inhibition factor (kᵢ) : $(@bind kᵢ PlutoUI.Slider(1.1:0.01:10, show_value=true)) \
Time constant (τ) :   $(@bind τ PlutoUI.Slider(1:0.01:10., show_value=true))\
"""

# ╔═╡ 2577b054-a601-4ebe-be26-ed84556d9898
#sigmoid function
σ(x) = 1/(1+exp(-x))

# ╔═╡ 480ddc97-b28b-4dd3-bdfb-ffea970860c6
#dirac delta
δ(t) = if t == 0 0. else 0. end

# ╔═╡ 302def95-25ca-4e12-af97-668a47f287ab
#theta
θ(t) = if t<0 0 else 1. end

# ╔═╡ cad0f835-747a-49fd-9dce-16facd560495
#external inputs maybe this is useless
begin
	Iₑ(t) =  δ(t);
	Iᵢ(t) = 0.;
end

# ╔═╡ 1a9c7cc4-e7ef-4690-82df-7185b7889573
#Callbacks to set inputs. δ functions somehow get ignored if we add them manually
begin
	#a condition when the callsbacks gonna be called
	condition(u,t,integrator) = t==0
	#affect the inetgrator
	function affect!(integrator)
		integrator.u[1] += .5
		integrator.u[2] += .5
	end
	#callback
	cb = DiscreteCallback(condition,affect!)
end

# ╔═╡ c2fa525d-2dff-4cce-a362-f22aa9889250
#ODE System for two populations of neurons
function balancedRNN!(du, u, p, t)
	du[1] = ((p[1] - p[2]*p[1] - 1)*u[1]+p[1]*(p[2]+1)*u[2]+(Iᵢ(t)+Iₑ(t))/2)/p[3]
	du[2] = (-u[2] + (Iₑ(t)-Iᵢ(t))/2)/p[3]
end

# ╔═╡ f0fdff5a-448f-4b69-a1bc-f773c9245ffd
#solution for the ODE with two different parameters, using RK4
begin
	#meh
	u0 = [.0, .0]
	tspan =(-1., 10)

	#noice
	p1 = [0, kᵢ, τ]
	prob1 = ODEProblem(balancedRNN!, u0, tspan, p1)
	solvbal = solve(prob1, Tsit5(), dt=0.01, dtmax=0.001, callback=cb, tstops=[0.0])

	p2 = [wb, kᵢ, τ]
	prob2 = ODEProblem(balancedRNN!, u0, tspan, p2)
	solvamp = solve(prob2, Tsit5(), dtmax=0.001, callback=cb, tstops=[0.0])
end;

# ╔═╡ 91de3b96-0bb8-4cad-859f-6bb2193af08e
#Plotting stuff and hope that god does not hate me
begin
	xvals1 = Float32.(solvamp.t[:])
	xvals2 = Float32.(solvbal.t[:])
	
	y1  = Float32.(vec(solvamp[1,:, : ].+solvamp[2, :, :]))
	y2  = Float32.(vec(solvbal[1,:, : ].+solvbal[2, :, :]))
	
	
	fig1 = Figure()
	ax1 = Axis(fig1[1,1 ], xlabel = "Time [τ]", ylabel = "rₑ")
	xlims!(ax1, (-1, 10))
	ylims!(ax1, (0, 2))
	ax1.xticks = -1:10
	l1 = lines!(ax1, xvals1, y1, color=:red, linewidth=5)
	l2 = lines!(ax1, xvals2, y2, color=:blue, linewidth=5)
	Legend(fig1[1, 1], [l1, l2], ["Amplification: 4x", "Amplification: 1x"], halign=:right, valign=:top, margin=[10, 10, 10, 10], tellwidth=false, tellheight=false)	
end

# ╔═╡ 9b07d41d-2ff8-44da-98c3-74a0b7942d08
fig1

# ╔═╡ 12589aa8-a8e0-4e6a-9b6c-b4c457d4579e
#save("fig2Bcairo.pdf", fig1)

# ╔═╡ 1c406eb5-d0ad-41c7-b9ad-0362701f43a4
#Integral of red plot should be 4x integral blue thing
begin
	s1 = sum(y1)*0.001;
	s2 = sum(y2)*0.001
	isit = isapprox(s1, 4*s2, rtol=0.01)
	print(if isit "Integrals are good, amplification OK" else "Integrals are not good, amplification WRONG" end)

end

# ╔═╡ 36891f66-6093-4fae-9f28-959d009805a5
md"""
## Hebbian case
"""

# ╔═╡ 38fad181-042a-4702-ab47-027e4be3b59c
begin
	times1 = 0.:0.001:15
	times2 = 15:0.001:30
	
	ys1 = -exp.(-times1) .+ 1.
	ys2 = ys1[end]*exp.(-times1)
	ys4 = exp.((0.75-1)*times1)/(0.75-1) .- 1/(0.75-1)
	ys5 = ys4[end]*exp.((0.75-1)*times1)
	
	fig2 = Figure()
	ax2 = Axis(fig2[1, 1])
	ax2.xticks = 0:5:30
	ylims!(ax2, (0., 5.))
	
	lnf2 = lines!(ax2, cat(times1, times2, dims=1), cat(ys1, ys2, dims=1),linewidth=5, color=:blue)
	lnf3 = lines!(ax2, cat(times1, times2, dims=1), cat(ys4, ys5, dims=1), linewidth=5, color=:red)
	
	Legend(fig2[1, 1], [lnf2, lnf3], ["Amplification: 1x", "Amplification: 4x"], halign=:right, valign=:top, margin=[10, 10, 10, 10], tellwidth=false, tellheight=false)
	
	fig2
end

# ╔═╡ bc167dfa-740f-4361-a85d-5b14acd36513
#integrals
begin
	s11 = (sum(ys1)+sum(ys2))*0.001
	s22 = (sum(ys4)+sum(ys5))*0.001
	isap = isapprox(4*s11, s22, rtol=0.1)
	print(if isap "Integrals OK, amplification OK" else "integrals and amp WRONG" end)
end

# ╔═╡ cbe42594-51e5-420f-a029-362b5cf0aa24


# ╔═╡ Cell order:
# ╠═d7ff42de-1f3b-473b-b734-36ecf866b60c
# ╠═e31d26ee-7970-11ed-082b-c3726a3719b6
# ╟─176295f7-36d0-4012-a1fe-5e9d1b149a36
# ╟─e8f819a1-5af2-4172-947b-f9fdba81a77f
# ╟─84f095e9-bc9b-47da-a2c5-60852d7715cc
# ╠═2577b054-a601-4ebe-be26-ed84556d9898
# ╠═480ddc97-b28b-4dd3-bdfb-ffea970860c6
# ╠═302def95-25ca-4e12-af97-668a47f287ab
# ╠═cad0f835-747a-49fd-9dce-16facd560495
# ╠═1a9c7cc4-e7ef-4690-82df-7185b7889573
# ╠═c2fa525d-2dff-4cce-a362-f22aa9889250
# ╠═f0fdff5a-448f-4b69-a1bc-f773c9245ffd
# ╟─91de3b96-0bb8-4cad-859f-6bb2193af08e
# ╟─9b07d41d-2ff8-44da-98c3-74a0b7942d08
# ╠═12589aa8-a8e0-4e6a-9b6c-b4c457d4579e
# ╠═1c406eb5-d0ad-41c7-b9ad-0362701f43a4
# ╟─36891f66-6093-4fae-9f28-959d009805a5
# ╟─38fad181-042a-4702-ab47-027e4be3b59c
# ╟─bc167dfa-740f-4361-a85d-5b14acd36513
# ╠═cbe42594-51e5-420f-a029-362b5cf0aa24

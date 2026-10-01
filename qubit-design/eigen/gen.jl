# Full-chip transmon + meander readout + feedline with DeviceLayout.jl's SingleTransmon example
# (the generator behind Palace's examples/transmon), plus its Palace eigenmode config and GDS.
# usage: julia --project=. gen.jl <outdir> '<json kwargs in um>'
import JSON
using DeviceLayout, DeviceLayout.PreferredUnits
const ST_PATH = joinpath(pkgdir(DeviceLayout), "examples", "SingleTransmon")
include(joinpath(ST_PATH, "SingleTransmon.jl"))
outdir = abspath(ARGS[1]); kw = JSON.parse(ARGS[2]); mkpath(joinpath(outdir, "mesh"))
geom = Dict{Symbol,Any}()
for k in ("cap_width", "cap_length", "cap_gap", "total_length", "claw_gap", "w_claw", "l_claw", "w_shield")
    haskey(kw, k) && (geom[Symbol(k)] = kw[k] * 1.0μm)
end
haskey(kw, "n_meander_turns") && (geom[:n_meander_turns] = Int(kw["n_meander_turns"]))
sm = SingleTransmon.single_transmon(; save_mesh=true, save_gds=true, geom...)
mv(joinpath(ST_PATH, "single_transmon.msh2"), joinpath(outdir, "mesh", "transmon.msh2"), force=true)
isfile(joinpath(ST_PATH, "single_transmon.gds")) && mv(joinpath(ST_PATH, "single_transmon.gds"), joinpath(outdir, "design_full_chip.gds"), force=true)
cfg = SingleTransmon.configfile(sm; solver_order=2)
cfg["Model"]["Mesh"] = joinpath(outdir, "mesh", "transmon.msh2")
cfg["Problem"]["Output"] = joinpath(outdir, "postpro")
open(joinpath(outdir, "eig_base.json"), "w") do io JSON.print(io, cfg, 1) end
println("done ", outdir)

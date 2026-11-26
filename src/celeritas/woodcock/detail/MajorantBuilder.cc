//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/detail/MajorantBuilder.cc
//---------------------------------------------------------------------------//
#include "MajorantBuilder.hh"

#include <algorithm>
#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/mat/MaterialView.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsParams.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Find the highest energy of any tabulated cross section.
 */
units::MevEnergy find_max_energy(HostCRef<PhysicsParamsData> const& params,
                                 ParticleId particle)
{
    double max_loge{0};
    auto const& group = params.process_groups[particle];
    for (auto const& table : params.xs_tables[group.macro_xs])
    {
        for (XsGridId grid_id : params.xs_grid_ids[table.grids])
        {
            if (!grid_id)
                continue;

            auto const& xs = params.xs_grids[grid_id];
            for (auto const& grid : {xs.lower, xs.upper})
            {
                if (grid)
                {
                    max_loge = std::max<double>(max_loge, grid.grid.back);
                }
            }
        }
    }
    return units::MevEnergy(std::exp(max_loge));
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with shared data and lower energy.
 */
MajorantBuilder::MajorantBuilder(MaterialParams const& materials,
                                 ParticleParams const& particles,
                                 PhysicsParams const& physics,
                                 ParticleId particle,
                                 Energy min_energy)
    : materials_{materials}
    , particles_{particles}
    , physics_{physics}
    , particle_{particle}
    , min_energy_{min_energy}
    , max_energy_{find_max_energy(physics.host_ref(), particle)}
    , particle_state_{particles.host_ref(), 1}
    , physics_state_{physics.host_ref(), 1}
{
    CELER_EXPECT(particle_ < particles.size());
    CELER_VALIDATE(
        min_energy_ < max_energy_,
        << "minimum energy " << min_energy_.value()
        << " MeV for Woodcock tracking is not below the highest tabulated "
           "energy "
        << max_energy_.value() << " MeV");
}

//---------------------------------------------------------------------------//
/*!
 * Build the majorant for the given materials.
 */
inp::UniformGrid MajorantBuilder::operator()(VecMat const& materials)
{
    CELER_EXPECT(!materials.empty());

    double const lo = std::log(value_as<Energy>(min_energy_));
    double const hi = std::log(value_as<Energy>(max_energy_));
    auto const num_bins = std::max<size_type>(
        1,
        static_cast<size_type>(
            std::ceil((hi - lo) / std::log(10.0) * bins_per_decade())));
    double const delta = (hi - lo) / num_bins;

    // Evaluate at grid points
    inp::UniformGrid result;
    result.x = {lo, hi};
    result.y.resize(num_bins + 1);
    for (auto i : range(result.y.size()))
    {
        result.y[i] = this->calc_max_xs(materials, std::exp(lo + delta * i));
    }

    // Increase both sides of each bin so that the interpolated majorant is
    // not exceeded. This can only increase the majorant in other bins.
    for (auto i : range(num_bins))
    {
        double const e_lo = std::exp(lo + delta * i);
        double const e_hi = std::exp(lo + delta * (i + 1));
        double& y_lo = result.y[i];
        double& y_hi = result.y[i + 1];

        double max_xs{std::max(y_lo, y_hi)};
        bool exceeded{false};
        for (auto j : range(size_type{1}, samples_per_bin()))
        {
            double const energy
                = std::exp(lo + delta * (i + double(j) / samples_per_bin()));
            double const frac = (energy - e_lo) / (e_hi - e_lo);
            double const interp = (1 - frac) * y_lo + frac * y_hi;
            double const xs = this->calc_max_xs(materials, energy);
            max_xs = std::max(max_xs, xs);
            exceeded = exceeded || xs > interp;
        }
        if (exceeded)
        {
            // Concave or discontinuous: use a constant
            y_lo = max_xs;
            y_hi = max_xs;
        }
    }

    for (double& y : result.y)
    {
        y *= safety_factor();
    }

    CELER_VALIDATE(
        std::all_of(
            result.y.begin(), result.y.end(), [](double y) { return y > 0; }),
        << "total macroscopic cross section is zero for all materials inside "
           "a Woodcock tracking region");
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the total macroscopic cross section [1/len].
 */
real_type MajorantBuilder::calc_xs(PhysMatId material, Energy energy)
{
    CELER_EXPECT(material < materials_.size());

    ParticleTrackView particle(
        particles_.host_ref(), particle_state_.ref(), TrackSlotId{0});
    particle = ParticleTrackInitializer{particle_, energy};
    PhysicsTrackView physics(physics_.host_ref(),
                             physics_state_.ref(),
                             particle,
                             material,
                             TrackSlotId{0});
    MaterialView material_view = materials_.get(material);

    real_type result{0};
    for (auto ppid : range(ParticleProcessId{physics.num_particle_processes()}))
    {
        result += physics.calc_xs(ppid, material_view, energy);
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum total cross section over all materials.
 */
double MajorantBuilder::calc_max_xs(VecMat const& materials, double energy)
{
    double result{0};
    for (PhysMatId mat : materials)
    {
        result = std::max<double>(result, this->calc_xs(mat, Energy(energy)));
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBEnergyDistribution.i.hh
//---------------------------------------------------------------------------//
#include <cmath>

#include "base/Algorithms.hh"
#include "physics/grid/TwodGridCalculator.hh"
#include "random/distributions/BernoulliDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and energy.
 *
 * The incident energy *must* be within the bounds of the SB table data, so the
 * Model's applicability must be consistent with the table data.
 */
CELER_FUNCTION
SBEnergyDistribution::SBEnergyDistribution(const SBData& data,
                                           Energy        inc_energy,
                                           ElementId     element,
                                           EnergySq      density_correction,
                                           Energy        min_gamma_energy)
    : inc_energy_{inc_energy.value()}
    , calc_xs_{this->make_xs_calc(data.differential_xs, element)}
    , inv_max_xs_{1 / this->calc_max_xs(data.differential_xs, element)}
    , dens_corr_(density_correction.value())
    , sample_log_exit_efrac_{this->make_lee_sampler(min_gamma_energy.value())}
{
    CELER_EXPECT(inc_energy > min_gamma_energy);
}

//---------------------------------------------------------------------------//
/*!
 * Sample the exiting energy by doing a table lookup and rejection.
 */
template<class Engine>
CELER_FUNCTION auto SBEnergyDistribution::operator()(Engine& rng) -> Energy
{
    const real_type inv_inc_energy = 1 / inc_energy_;

    // Sampled energy
    real_type exit_energy{};
    // Calculated cross section used inside rejection sampling
    real_type xs{};
    do
    {
        // Sample scaled energy and subtract correction factor
        real_type esq = std::exp(sample_log_exit_efrac_(rng)) - dens_corr_;
        CELER_ASSERT(esq >= 0);
        exit_energy = std::sqrt(esq);

        // Interpolate the differential cross setion at the sampled exit energy
        xs = calc_xs_(exit_energy * inv_inc_energy);
        CELER_ASSERT(xs >= 0 && xs <= 1 / inv_max_xs_);
    } while (!BernoulliDistribution(xs * inv_max_xs_)(rng));
    return Energy{exit_energy};
}

//---------------------------------------------------------------------------//
/*!
 * Construct the differential cross section calculator for exit energy.
 *
 * Note that this is done during construction so the initialization order for
 * member variables really matters!!
 */
CELER_FUNCTION TwodSubgridCalculator SBEnergyDistribution::make_xs_calc(
    const SBTables& xs_params, ElementId element) const
{
    CELER_EXPECT(element < xs_params.elements.size());
    CELER_EXPECT(inc_energy_ > 0);

    const TwodGridData& grid = xs_params.elements[element].grid;
    CELER_ASSERT(inc_energy_ >= std::exp(xs_params.reals[grid.x.front()])
                 && inc_energy_ < std::exp(xs_params.reals[grid.x.back()]));

    static_assert(
        std::is_same<Energy::unit_type, units::Mev>::value
            && std::is_same<SBElementTableData::EnergyUnits, units::LogMev>::value,
        "Inconsistent energy units");
    return TwodGridCalculator(grid, xs_params.reals)(std::log(inc_energy_));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate a bounding maximum of the differential cross section.
 *
 * This interpolates the maximum cross section for the given incident energy
 * by using the pre-calculated cross section maxima. The interpolated value is
 * typically exactly the
 * maximum (since the two \em y points are usually adjacent, and therefore the
 * linear interpolation between them is exact) but at worst (e.g. for the
 * double-peaked function of brems at lower energies) an upper bound which can
 * be proven by the triangle inequality.
 */
CELER_FUNCTION real_type SBEnergyDistribution::calc_max_xs(
    const SBTables& xs_params, ElementId element) const
{
    CELER_EXPECT(element);
    const SBElementTableData& el = xs_params.elements[element];

    auto get_value = [&xs_params, &el](size_type ix) -> real_type {
        // Index of the largest xs for exiting energy for the given incident
        // grid point
        size_type iy = xs_params.sizes[el.argmax[ix]];
        // Value of the maximum cross section
        return xs_params.reals[el.grid.at(ix, iy)];
    };

    const size_type x_idx  = calc_xs_.x_index();
    const real_type x_frac = calc_xs_.x_fraction();

    real_type xs_max = (1 - x_frac) * get_value(x_idx)
                       + x_frac * get_value(x_idx + 1);

    CELER_ENSURE(xs_max > 0);
    return xs_max;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a sampler for scaled log exiting energy.
 */
CELER_FUNCTION auto
SBEnergyDistribution::make_lee_sampler(real_type min_gamma_energy) const
    -> UniformSampler
{
    CELER_EXPECT(min_gamma_energy > 0);
    return UniformSampler(std::log(ipow<2>(min_gamma_energy) + dens_corr_),
                          std::log(ipow<2>(inc_energy_) + dens_corr_));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

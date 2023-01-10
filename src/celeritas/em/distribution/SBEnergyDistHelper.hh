//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/SBEnergyDistHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/SeltzerBergerData.hh"
#include "celeritas/grid/TwodGridCalculator.hh"
#include "celeritas/grid/TwodSubgridCalculator.hh"
#include "celeritas/random/distribution/ReciprocalDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Help sample exiting photon energy from Bremsstrahlung.
 *
 * This class simply preprocesses the input data needed for the
 * SBEnergyDistribution, which is templated on a dynamic cross section
 * correction factor.
 *
 * The cross section units are immaterial since the cross section merely acts
 * as a shape function for rejection: the sampled energy's cross section is
 * always divided by the maximium cross section.
 */
class SBEnergyDistHelper
{
  public:
    //!@{
    //! Type aliases
    using SBDXsec = NativeCRef<SeltzerBergerTableData>;
    using Xs = Quantity<SBElementTableData::XsUnits>;
    using Energy = units::MevEnergy;
    using EnergySq = Quantity<UnitProduct<units::Mev, units::Mev>>;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION SBEnergyDistHelper(SBDXsec const& differential_xs,
                                             Energy inc_energy,
                                             ElementId element,
                                             EnergySq density_correction,
                                             Energy min_gamma_energy);

    // Sample scaled energy (analytic component of exiting distribution)
    template<class Engine>
    inline CELER_FUNCTION Energy sample_exit_energy(Engine& rng) const;

    // Calculate tabulated cross section for a given energy
    inline CELER_FUNCTION Xs calc_xs(Energy energy) const;

    //! Maximum cross section calculated for rejection
    CELER_FUNCTION Xs max_xs() const { return Xs{max_xs_.xs}; }

    //! Cross section at the first reduced photon energy grid point
    CELER_FUNCTION Xs xs_zero() const { return Xs{max_xs_.xs_zero}; }

  private:
    //// IMPLEMENTATION TYPES ////

    using SBTables = NativeCRef<SeltzerBergerTableData>;
    using ReciprocalSampler = ReciprocalDistribution<real_type>;

    struct MaxXs
    {
        real_type xs;
        real_type xs_zero;
    };

    //// IMPLEMENTATION DATA ////

    const TwodSubgridCalculator calc_xs_;
    const MaxXs max_xs_;

    const real_type inv_inc_energy_;
    const real_type dens_corr_;
    const ReciprocalSampler sample_exit_esq_;

    //// CONSTRUCTION HELPER FUNCTIONS ////

    inline CELER_FUNCTION TwodSubgridCalculator make_xs_calc(
        SBTables const&, real_type inc_energy, ElementId element) const;

    inline CELER_FUNCTION MaxXs calc_max_xs(SBTables const& xs_params,
                                            ElementId element) const;

    inline CELER_FUNCTION ReciprocalSampler
    make_esq_sampler(real_type inc_energy, real_type min_gamma_energy) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and energy.
 *
 * The incident energy *must* be within the bounds of the SB table data, so the
 * Model's applicability must be consistent with the table data.
 */
CELER_FUNCTION
SBEnergyDistHelper::SBEnergyDistHelper(SBDXsec const& differential_xs,
                                       Energy inc_energy,
                                       ElementId element,
                                       EnergySq density_correction,
                                       Energy min_gamma_energy)
    : calc_xs_{this->make_xs_calc(differential_xs, inc_energy.value(), element)}
    , max_xs_{this->calc_max_xs(differential_xs, element)}
    , inv_inc_energy_(1 / inc_energy.value())
    , dens_corr_(density_correction.value())
    , sample_exit_esq_{
          this->make_esq_sampler(inc_energy.value(), min_gamma_energy.value())}
{
    CELER_EXPECT(inc_energy > min_gamma_energy);
}

//---------------------------------------------------------------------------//
/*!
 * Sample an exit energy on a scaled and adjusted reciprocal distribution.
 */
template<class Engine>
CELER_FUNCTION auto SBEnergyDistHelper::sample_exit_energy(Engine& rng) const
    -> Energy
{
    // Sample scaled energy and subtract correction factor
    real_type esq = sample_exit_esq_(rng) - dens_corr_;
    CELER_ASSERT(esq >= 0);
    return Energy{std::sqrt(esq)};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate tabulated cross section for a given energy.
 */
CELER_FUNCTION auto SBEnergyDistHelper::calc_xs(Energy e) const -> Xs
{
    CELER_EXPECT(e > zero_quantity());
    // Interpolate the differential cross setion at the given exit energy
    return Xs{calc_xs_(e.value() * inv_inc_energy_)};
}

//---------------------------------------------------------------------------//
/*!
 * Construct the differential cross section calculator for exit energy.
 */
CELER_FUNCTION TwodSubgridCalculator SBEnergyDistHelper::make_xs_calc(
    SBTables const& xs_params, real_type inc_energy, ElementId element) const
{
    CELER_EXPECT(element < xs_params.elements.size());
    CELER_EXPECT(inc_energy > 0);

    TwodGridData const& grid = xs_params.elements[element].grid;
    CELER_ASSERT(inc_energy >= std::exp(xs_params.reals[grid.x.front()])
                 && inc_energy < std::exp(xs_params.reals[grid.x.back()]));

    static_assert(
        std::is_same<Energy::unit_type, units::Mev>::value
            && std::is_same<SBElementTableData::EnergyUnits, units::LogMev>::value,
        "Inconsistent energy units");
    return TwodGridCalculator(grid, xs_params.reals)(std::log(inc_energy));
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
 *
 * The maximum scaled cross section for positrons will always be at the first
 * reduced photon energy grid point.
 *
 * \note This is called during construction, so \c calc_xs_ must be initialized
 * before whatever calls this.
 */
CELER_FUNCTION auto SBEnergyDistHelper::calc_max_xs(SBTables const& xs_params,
                                                    ElementId element) const
    -> MaxXs
{
    CELER_EXPECT(element);
    SBElementTableData const& el = xs_params.elements[element];

    const size_type x_idx = calc_xs_.x_index();
    const real_type x_frac = calc_xs_.x_fraction();
    MaxXs result;

    // Calc max xs
    auto get_value = [&xs_params, &el](size_type ix) -> real_type {
        // Index of the largest xs for exiting energy for the given
        // incident grid point
        size_type iy = xs_params.sizes[el.argmax[ix]];
        // Value of the maximum cross section
        return xs_params.reals[el.grid.at(ix, iy)];
    };
    result.xs = (1 - x_frac) * get_value(x_idx) + x_frac * get_value(x_idx + 1);

    // Calc max scaled xs
    result.xs_zero = (1 - x_frac) * xs_params.reals[el.grid.at(x_idx, 0)]
                     + x_frac * xs_params.reals[el.grid.at(x_idx + 1, 0)];

    CELER_ENSURE(result.xs > 0);
    CELER_ENSURE(result.xs_zero > 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a sampler for scaled exiting energy.
 */
CELER_FUNCTION auto
SBEnergyDistHelper::make_esq_sampler(real_type inc_energy,
                                     real_type min_gamma_energy) const
    -> ReciprocalSampler
{
    CELER_EXPECT(min_gamma_energy > 0);
    return ReciprocalSampler(ipow<2>(min_gamma_energy) + dens_corr_,
                             ipow<2>(inc_energy) + dens_corr_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

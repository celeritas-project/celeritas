//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBEnergyDistHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "physics/base/Units.hh"
#include "physics/grid/TwodSubgridCalculator.hh"
#include "random/distributions/ReciprocalDistribution.hh"
#include "SeltzerBerger.hh"

namespace celeritas
{
namespace detail
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
 *
 * Note that the *energy* of the maximum cross section is only needed for the
 * cross section scaling function used to correct the exiting energy
 * distribution for positrons.
 */
class SBEnergyDistHelper
{
  public:
    //!@{
    //! Type aliases
    using SBData
        = SeltzerBergerData<Ownership::const_reference, MemSpace::native>;
    using Xs       = Quantity<SBElementTableData::XsUnits>;
    using Energy   = units::MevEnergy;
    using EnergySq = Quantity<UnitProduct<units::Mev, units::Mev>>;
    //!@}

  public:
    // Construct from data
    inline CELER_FUNCTION SBEnergyDistHelper(const SBData& data,
                                             Energy        inc_energy,
                                             ElementId     element,
                                             EnergySq      density_correction,
                                             Energy        min_gamma_energy);

    // Sample scaled energy (analytic component of exiting distribution)
    template<class Engine>
    inline CELER_FUNCTION Energy sample_exit_energy(Engine& rng) const;

    // Calculate tabulated cross section for a given energy
    inline CELER_FUNCTION Xs calc_xs(Energy energy) const;

    //! Energy of maximum cross section
    CELER_FUNCTION Energy max_xs_energy() const
    {
        return Energy{max_xs_.energy};
    }

    //! Maximum cross section calculated for rejection
    CELER_FUNCTION Xs max_xs() const { return Xs{max_xs_.xs}; }

  private:
    //// IMPLEMENTATION TYPES ////

    using SBTables
        = SeltzerBergerTableData<Ownership::const_reference, MemSpace::native>;
    using ReciprocalSampler = ReciprocalDistribution<real_type>;

    struct MaxXs
    {
        real_type energy;
        real_type xs;
    };

    //// IMPLEMENTATION DATA ////

    const TwodSubgridCalculator calc_xs_;
    const MaxXs                 max_xs_;

    const real_type         inv_inc_energy_;
    const real_type   dens_corr_;
    const ReciprocalSampler sample_exit_esq_;

    //// CONSTRUCTION HELPER FUNCTIONS ////

    inline CELER_FUNCTION TwodSubgridCalculator make_xs_calc(
        const SBTables&, real_type inc_energy, ElementId element) const;

    inline CELER_FUNCTION MaxXs calc_max_xs(const SBTables& xs_params,
                                            real_type       inc_energy,
                                            ElementId       element) const;

    inline CELER_FUNCTION ReciprocalSampler
    make_esq_sampler(real_type inc_energy, real_type min_gamma_energy) const;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "SBEnergyDistHelper.i.hh"

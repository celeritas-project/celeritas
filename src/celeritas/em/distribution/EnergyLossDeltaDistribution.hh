//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/EnergyLossDeltaDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Quantities.hh"

#include "EnergyLossHelper.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Passthrough model for "no distribution" energy loss.
 *
 * The distribution is a Delta function so that the "sampled" value is always
 * the mean energy.
 */
class EnergyLossDeltaDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    //! Construct from helper-calculated mean
    explicit CELER_FUNCTION
    EnergyLossDeltaDistribution(EnergyLossHelper const& helper)
        : mean_energy_(helper.mean_loss())
    {
    }

    //! Result is always the mean energy
    template<class Generator>
    CELER_FUNCTION Energy operator()(Generator&) const
    {
        return mean_energy_;
    }

  private:
    Energy mean_energy_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

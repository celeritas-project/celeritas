//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Temporary class for testing optical interactors.
 *
 * \todo Fill with actual code.
 */
class OpticalTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

  public:
    inline CELER_FUNCTION OpticalTrackView(Energy energy, Real3 const& pol)
        : energy_(energy), polarization_(pol)
    {
    }

    inline CELER_FUNCTION Energy energy() const { return energy_; }

    inline CELER_FUNCTION Real3 const& polarization() const
    {
        return polarization_;
    }

  private:
    Energy energy_;
    Real3 polarization_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

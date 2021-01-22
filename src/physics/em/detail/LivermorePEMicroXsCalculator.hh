//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "physics/material/Types.hh"
#include "LivermorePE.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate photoelectric effect cross sections using the Livermore data.
 */
class LivermorePEMicroXsCalculator
{
  public:
    //!@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    LivermorePEMicroXsCalculator(const LivermorePEPointers& shared,
                                 const ParticleTrackView&   particle);

    // Compute cross section
    inline CELER_FUNCTION real_type operator()(ElementDefId el_id) const;

  private:
    // Shared constant physics properties
    const LivermorePEPointers& shared_;
    // Incident gamma energy
    const MevEnergy inc_energy_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#include "LivermorePEMicroXsCalculator.i.hh"

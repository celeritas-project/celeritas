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
#include "LivermorePEParamsPointers.hh"
#include "LivermorePEInteractorPointers.hh"

namespace celeritas
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
    LivermorePEMicroXsCalculator(const LivermorePEInteractorPointers& shared,
                                 const LivermorePEParamsPointers&     data,
                                 const ParticleTrackView& particle);

    // Compute cross section
    inline CELER_FUNCTION real_type operator()(ElementDefId el_id) const;

  private:
    // Shared constant physics properties
    const LivermorePEInteractorPointers& shared_;
    // Livermore EPICS2014 photoelectric cross section data
    const LivermorePEParamsPointers& data_;
    // Incident gamma energy
    const MevEnergy inc_energy_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "LivermorePEMicroXsCalculator.i.hh"

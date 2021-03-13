//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermoreXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Storage for energy and cross sections.
 * TODO: replace with GenericGridData
 */
struct LivermoreValueGrid
{
    Span<const real_type> energy;
    Span<const real_type> xs;
    Interp                interp;
};

//---------------------------------------------------------------------------//
/*!
 * Find and interpolate cross section data.
 * TODO: temporary
 */
class LivermoreXsCalculator
{
  public:
    //@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //@}

  public:
    // Construct from state-independent data
    explicit inline CELER_FUNCTION
    LivermoreXsCalculator(const LivermoreValueGrid& data);

    // Find and interpolate basesd on the particle track's current energy
    inline CELER_FUNCTION real_type operator()(const real_type energy) const;

  private:
    const LivermoreValueGrid& data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "LivermoreXsCalculator.i.hh"

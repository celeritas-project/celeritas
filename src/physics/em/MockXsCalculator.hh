//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MockXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Storage for energy and cross sections.
 * TODO: replace with GenericGridPointers
 */
struct ValueGrid
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
class XsCalculator
{
  public:
    //@{
    //! Type aliases
    using MevEnergy = units::MevEnergy;
    //@}

  public:
    // Construct from state-independent data
    explicit inline CELER_FUNCTION XsCalculator(const ValueGrid& data);

    // Find and interpolate basesd on the particle track's current energy
    inline CELER_FUNCTION real_type operator()(const real_type energy) const;

  private:
    const ValueGrid& data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MockXsCalculator.i.hh"

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGMacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/em/detail/EPlusGG.hh"
#include "physics/material/MaterialView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculates the macroscopic cross section.
 */
class EPlusGGMacroXsCalculator
{
  public:
    //!@{
    //! Type aliases
    using MevEnergy       = units::MevEnergy;
    using XsUnits         = units::NativeUnit;
    using EPlusGGPointers = detail::EPlusGGPointers;
    //!@}

  public:
    // Construct with shared data and material
    inline CELER_FUNCTION
    EPlusGGMacroXsCalculator(const EPlusGGPointers& shared,
                             const MaterialView&    material);

    // Compute cross section on the fly at the given energy
    inline CELER_FUNCTION real_type operator()(MevEnergy energy) const;

  private:
    const real_type electron_mass_;
    const real_type electron_density_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "EPlusGGMacroXsCalculator.i.hh"

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEMacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Span.hh"
#include "base/Types.hh"
#include "physics/material/MaterialView.hh"
#include "physics/em/detail/LivermorePEMicroXsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculates the macroscopic cross section.
 */
class LivermorePEMacroXsCalculator
{
  public:
    //!@{
    //! Type aliases
    using Energy         = detail::LivermorePEMicroXsCalculator::Energy;
    using MicroXsUnits   = detail::LivermorePEMicroXsCalculator::XsUnits;
    using XsUnits        = units::NativeUnit;
    using LivermorePERef = detail::LivermorePERef;
    //!@}

  public:
    // Construct with shared data and material
    inline CELER_FUNCTION
    LivermorePEMacroXsCalculator(const LivermorePERef& shared,
                                 const MaterialView&   material);

    // Compute cross section on the fly at the given energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    const LivermorePERef&           shared_;
    Span<const MatElementComponent> elements_;
    real_type                       number_density_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "LivermorePEMacroXsCalculator.i.hh"

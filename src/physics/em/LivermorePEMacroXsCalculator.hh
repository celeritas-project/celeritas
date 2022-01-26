//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEMacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
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
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared model and material data.
 */
CELER_FUNCTION LivermorePEMacroXsCalculator::LivermorePEMacroXsCalculator(
    const LivermorePERef& shared, const MaterialView& material)
    : shared_(shared)
    , elements_(material.elements())
    , number_density_(material.number_density())
{
    CELER_EXPECT(!elements_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Compute macroscopic cross section for the photoelectric effect on the fly at
 * the given energy.
 */
CELER_FUNCTION real_type
LivermorePEMacroXsCalculator::operator()(Energy energy) const
{
    real_type                            result = 0.;
    detail::LivermorePEMicroXsCalculator calc_micro_xs(shared_, energy);
    for (const auto& el_comp : elements_)
    {
        const real_type micro_xs = calc_micro_xs(el_comp.element);
        CELER_ASSERT(micro_xs >= 0);
        result += micro_xs * el_comp.fraction;
    }
    result *= MicroXsUnits::value() * number_density_;
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LivermorePEMacroXsCalculator.i.hh
//---------------------------------------------------------------------------//

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared model and material data.
 */
CELER_FUNCTION LivermorePEMacroXsCalculator::LivermorePEMacroXsCalculator(
    const LivermorePEPointers& shared, const MaterialView& material)
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

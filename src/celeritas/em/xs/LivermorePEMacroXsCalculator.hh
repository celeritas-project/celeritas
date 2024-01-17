//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/xs/LivermorePEMacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/mat/MaterialView.hh"

#include "LivermorePEMicroXsCalculator.hh"

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
    //! \name Type aliases
    using Energy = LivermorePEMicroXsCalculator::Energy;
    using MicroXsUnits = LivermorePEMicroXsCalculator::XsUnits;
    using XsUnits = units::Native;
    //!@}

  public:
    // Construct with shared data and material
    inline CELER_FUNCTION
    LivermorePEMacroXsCalculator(LivermorePERef const& shared,
                                 MaterialView const& material);

    // Compute cross section on the fly at the given energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    LivermorePERef const& shared_;
    Span<MatElementComponent const> elements_;
    real_type number_density_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared model and material data.
 */
CELER_FUNCTION LivermorePEMacroXsCalculator::LivermorePEMacroXsCalculator(
    LivermorePERef const& shared, MaterialView const& material)
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
    real_type result = 0.;
    LivermorePEMicroXsCalculator calc_micro_xs(shared_, energy);
    for (auto const& el_comp : elements_)
    {
        real_type const micro_xs = calc_micro_xs(el_comp.element);
        CELER_ASSERT(micro_xs >= 0);
        result += micro_xs * el_comp.fraction;
    }
    result *= MicroXsUnits::value() * number_density_;
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

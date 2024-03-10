//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/MacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/mat/MaterialView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculates the macroscopic cross section.
 *
 * \tparam MicroXsT microscopic (element) cross section calculator
 */
template<class MicroXsT>
class MacroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = typename MicroXsT::ParamsRef;
    using Energy = typename MicroXsT::Energy;
    using MicroXs = units::BarnXs;
    using MacroXsUnits = units::Native;  // [1/len]
    //!@}

  public:
    // Construct with shared and material
    inline CELER_FUNCTION
    MacroXsCalculator(ParamsRef const& shared, MaterialView const& material);

    // Compute the macroscopic cross section on the fly at the given energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    ParamsRef const& shared_;
    Span<MatElementComponent const> elements_;
    real_type number_density_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and material data.
 */
template<class MicroXsT>
CELER_FUNCTION
MacroXsCalculator<MicroXsT>::MacroXsCalculator(ParamsRef const& shared,
                                               MaterialView const& material)
    : shared_(shared)
    , elements_(material.elements())
    , number_density_(material.number_density())
{
    CELER_EXPECT(!elements_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Compute the macroscopic cross section on the fly at the given energy.
 */
template<class MicroXsT>
CELER_FUNCTION real_type
MacroXsCalculator<MicroXsT>::operator()(Energy energy) const
{
    real_type result = 0.;
    MicroXsT calc_micro_xs(shared_, energy);
    for (auto const& el_comp : elements_)
    {
        real_type micro_xs = value_as<MicroXs>(calc_micro_xs(el_comp.element));
        CELER_ASSERT(micro_xs >= 0);
        result += micro_xs * el_comp.fraction;
    }
    result = native_value_from(MicroXs{result}) * number_density_;
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

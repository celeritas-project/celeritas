//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/xs/NeutronElasticMacroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "celeritas/UnitTypes.hh"
#include "celeritas/mat/MaterialView.hh"

#include "NeutronElasticMicroXsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Calculates the macroscopic cross section.
 */
class NeutronElasticMacroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    using XsUnits = units::Native;  // [1/len]
    //!@}

  public:
    // Construct with shared data and material
    inline CELER_FUNCTION
    NeutronElasticMacroXsCalculator(NeutronElasticRef const& shared,
                                    MaterialView const& material);

    // Compute cross section on the fly at the given energy
    inline CELER_FUNCTION real_type operator()(Energy energy) const;

  private:
    NeutronElasticRef const& shared_;
    Span<MatElementComponent const> elements_;
    real_type number_density_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared model and material data.
 */
CELER_FUNCTION NeutronElasticMacroXsCalculator::NeutronElasticMacroXsCalculator(
    NeutronElasticRef const& shared, MaterialView const& material)
    : shared_(shared)
    , elements_(material.elements())
    , number_density_(material.number_density())
{
    CELER_EXPECT(!elements_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Compute macroscopic cross section for the neutron elastic on the fly at
 * the given energy.
 */
CELER_FUNCTION real_type
NeutronElasticMacroXsCalculator::operator()(Energy energy) const
{
    real_type result = 0.;
    NeutronElasticMicroXsCalculator calc_micro_xs(shared_, energy);
    for (auto const& el_comp : elements_)
    {
        real_type const micro_xs = calc_micro_xs(el_comp.element);
        CELER_ASSERT(micro_xs >= 0);
        result += micro_xs * el_comp.fraction;
    }
    result *= XsUnits::value() * number_density_;
    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

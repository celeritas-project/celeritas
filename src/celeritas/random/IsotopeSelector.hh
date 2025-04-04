//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/IsotopeSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "celeritas/mat/ElementView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Make a weighted random selection of an isotope.
 */
class IsotopeSelector
{
  public:
    // Construct with element information
    inline CELER_FUNCTION IsotopeSelector(ElementView const& element);

    // Sample with the given RNG
    template<class Engine>
    inline CELER_FUNCTION IsotopeComponentId operator()(Engine& rng) const;

  private:
    ElementView element_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with element.
 */
CELER_FUNCTION IsotopeSelector::IsotopeSelector(ElementView const& element)
    : element_(element)
{
    CELER_EXPECT(element_.num_isotopes());
}

//---------------------------------------------------------------------------//
/*!
 * Sample the isotope with the given RNG.
 */
template<class Engine>
CELER_FUNCTION IsotopeComponentId IsotopeSelector::operator()(Engine& rng) const
{
    auto const& isotopes = element_.isotopes();
    real_type cumulative = -generate_canonical(rng);
    size_type imax = isotopes.size() - 1;
    size_type i = 0;
    for (; i < imax; ++i)
    {
        cumulative += isotopes[i].fraction;
        if (cumulative > 0)
            break;
    }
    return IsotopeComponentId{i};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/IsotopeSelector.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/random/distribution/Selector.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/ElementView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Make a sampler for a number density-weighted selection of an isotope.
 */
inline CELER_FUNCTION auto make_isotope_selector(ElementView const& element)
{
    CELER_EXPECT(element.num_isotopes() > 0);
    return make_selector(
        [isotopes = element.isotopes()](IsotopeComponentId ic_id) {
            return isotopes[ic_id.get()].fraction;
        },
        id_cast<IsotopeComponentId>(element.num_isotopes()));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

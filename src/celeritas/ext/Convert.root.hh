//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/Convert.root.hh
//---------------------------------------------------------------------------//
#pragma once

#include <TLeaf.h>

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Fetch single-dimension leaves.
 */
template<class T>
inline auto from_leaf(TLeaf const& leaf) -> T
{
    CELER_EXPECT(!leaf.IsZombie());
    return static_cast<T>(leaf.GetValue());
}

//---------------------------------------------------------------------------//
/*!
 * Fetch leaves containing `std::array<double, 3>`.
 */
inline Real3 from_array_leaf(TLeaf const& leaf)
{
    CELER_EXPECT(!leaf.IsZombie());
    CELER_ASSERT(leaf.GetLen() == 3);
    return {static_cast<real_type>(leaf.GetValue(0)),
            static_cast<real_type>(leaf.GetValue(1)),
            static_cast<real_type>(leaf.GetValue(2))};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

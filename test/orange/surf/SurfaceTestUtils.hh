//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceTestUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>

#include "corecel/cont/Array.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Find the minimum intersection if "zero" means "infinite".
 */
template<size_type N>
real_type min_intersection(Array<real_type, N> const& i)
{
    auto iter = std::min_element(
        i.begin(), i.end(), [](real_type left, real_type right) {
            if (left == right)
                return false;
            if (left == 0)
                return false;
            if (right == 0)
                return true;
            return (left < right);
        });
    return *iter;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/inp/Bih.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
/*!
 * Construction options for BIH tree.
 */
struct BIHBuilder
{
    //! Maximum number of bboxes that can reside on a leaf node without
    //! triggering a partitioning attempt
    size_type max_leaf_size = 1;

    //! Whether the options are valid
    explicit operator bool() const { return max_leaf_size >= 1; }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

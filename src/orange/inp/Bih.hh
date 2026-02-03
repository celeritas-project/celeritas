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
    //! Minimum number of bboxes needed to trigger a partitioning attempt
    size_type min_split_size = 2;

    //! Whether the options are valid
    explicit operator bool() const { return min_split_size >= 2; }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

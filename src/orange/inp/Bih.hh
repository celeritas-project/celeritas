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

    //! Hard limit on the depth of most the embedded node (where 1 is the root
    //! node)
    size_type depth_limit = 32;

    //! The number of partition candidates to check per axis when partitioning
    //! a node during BIH construction
    size_type num_part_cands = 3;

    //! Whether the options are valid
    explicit operator bool() const
    {
        return max_leaf_size >= 1 && depth_limit >= 1 && num_part_cands >= 1;
    }
};

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas

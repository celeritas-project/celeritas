//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/SurfaceInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "orange/Data.hh"
#include "orange/Types.hh"

namespace celeritas
{
struct SurfaceInput;
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct surfaces on the host.
 *
 * This appends all surfaces from a single unit. We could potentially
 * deduplicate surfaces across universes with a remapping?
 */
class SurfaceInserter
{
  public:
    //!@{
    //! Type aliases
    using Data         = HostVal<OrangeParamsData>;
    using SurfaceRange = ItemRange<struct Surface>;
    //!@}

  public:
    // Construct with reference to surfaces to build
    explicit SurfaceInserter(Data* params);

    // Create a bunch of surfaces (experimental)
    SurfaceRange operator()(const SurfaceInput& all_surfaces);

  private:
    Data* orange_data_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

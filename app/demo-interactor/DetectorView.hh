//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/StackAllocatorView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store a detector hit into the buffer.
 */
class DetectorView
{
  public:
    // Construct from pointers
    explicit inline CELER_FUNCTION
    DetectorView(const DetectorPointers& pointers);

    // Record a hit
    inline CELER_FUNCTION void operator()(const Hit& hit);

  private:
    StackAllocatorView<Hit> allocate_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "DetectorView.i.hh"

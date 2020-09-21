//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/StackAllocatorPointers.hh"
#include "base/UniformGrid.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Energy deposition event in the detector.
 *
 * Note that most of the data is discarded at integration time.
 */
struct Hit
{
    Real3            pos;
    Real3            dir;
    ThreadId         thread;
    real_type        time;
    units::MevEnergy energy_deposited;
};

//---------------------------------------------------------------------------//
/*!
 * Interface to detector hit buffer.
 */
struct DetectorPointers
{
    StackAllocatorPointers<Hit> hit_buffer;
    UniformGrid::Params         tally_grid;
    span<real_type>             tally_deposition;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const { return bool(hit_buffer); }

    //! Total capacity of hit buffer
    CELER_FUNCTION size_type capacity() const { return hit_buffer.capacity(); }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

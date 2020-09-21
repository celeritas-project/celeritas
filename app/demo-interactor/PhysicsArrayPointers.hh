//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsArrayPointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Span.hh"
#include "base/UniformGrid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interface for passing physics array data to device.
 */
struct PhysicsArrayPointers
{
    UniformGrid::Params   log_energy;
    real_type             prime_energy;
    span<const real_type> xs;

    //! Whether the interface is initialized
    explicit CELER_FUNCTION operator bool() const
    {
        return log_energy && !xs.empty();
    }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

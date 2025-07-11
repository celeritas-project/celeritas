//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/GeneratorCounters.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Current sizes of the buffered generator data.
 *
 * The number of pending and generated tracks are updated by value on the host
 * at each core step. To allow accumulation over many steps which each may have
 * many photons, the type is templated.
 */
template<class T = ::celeritas::size_type>
struct GeneratorCounters
{
    using size_type = T;

    //! Number of generators
    size_type buffer_size{0};
    //! Number of photons remaining to be generated
    size_type num_pending{0};
    //! Number of photons generated
    size_type num_generated{0};

    //! True if any queued generators/tracks exist
    CELER_FUNCTION bool empty() const
    {
        return buffer_size == 0 && num_pending == 0;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Stopwatch.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Start the count at construction
 */
Stopwatch::Stopwatch() : start_(Clock::now()) {}

//---------------------------------------------------------------------------//
/*!
 * Get the current elapsed time in seconds
 */
real_type Stopwatch::operator()() const
{
    using DurationSec = std::chrono::duration<real_type>;

    auto duration = Clock::now() - start_;
    return std::chrono::duration_cast<DurationSec>(duration).count();
}

//---------------------------------------------------------------------------//
} // namespace celeritas

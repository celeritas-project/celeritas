//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.perfetto.cc
//! \brief The perfetto implementation of \c ScopedProfiling
//---------------------------------------------------------------------------//
#include "ScopedProfiling.hh"

#include <perfetto.h>

#include "corecel/io/Logger.hh"

#include "Environment.hh"

#include "detail/TrackEvent.perfetto.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Start a thread-local slice track event
 */
void ScopedProfiling::activate([[maybe_unused]] Input const& input) noexcept
{
    TRACE_EVENT_BEGIN(detail::perfetto_track_event_category,
                      perfetto::DynamicString{std::string{input.name}});
}

/*!
 * End the slice track event that was started on the current thread
 */
void ScopedProfiling::deactivate() noexcept
{
    TRACE_EVENT_END(detail::perfetto_track_event_category);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
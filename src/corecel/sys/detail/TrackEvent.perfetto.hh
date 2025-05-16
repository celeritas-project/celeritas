//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/detail/TrackEvent.perfetto.hh
//! \brief Define perfetto track event categories
//---------------------------------------------------------------------------//
#pragma once

#include <perfetto.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Global category name for celeritas
//! TODO: add more categories for fine-grained control of events to record
inline constexpr auto* perfetto_track_event_category{"celeritas"};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas

// This declaration needs to be available when calling perfetto's
// TRACE_EVENT_BEGIN (i.e. ScopedProfiling TU) and when initializing a tracing
// session (i.e. TracingSession TU). Adding this in a public header would
// propagate perfetto dependency to downstream so hide it in a header to be
// included by corecel TU only.
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category(celeritas::detail::perfetto_track_event_category)
        .SetDescription(R"(Events from the celeritas library)"));

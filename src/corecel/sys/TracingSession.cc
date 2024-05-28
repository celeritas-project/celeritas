//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TracingSession.cc
//! \brief RAII class for managing a perfetto session and its resources.
//---------------------------------------------------------------------------//
#include "TracingSession.hh"

#include <fcntl.h>
#include <perfetto.h>

#include "corecel/Assert.hh"

#include "Environment.hh"
#include "ScopedProfiling.hh"

#include "detail/TrackEvent.perfetto.hh"

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
using celeritas::TracingMode;

//---------------------------------------------------------------------------//
/*!
 * Initialize the session for the given mode if profiling is enabled.
 */
std::unique_ptr<perfetto::TracingSession>
initialize_session(TracingMode mode) noexcept
{
    if (!celeritas::use_profiling())
    {
        return nullptr;
    }
    perfetto::TracingInitArgs args;
    args.backends |= [&] {
        switch (mode)
        {
            case TracingMode::InProcess:
                return perfetto::kInProcessBackend;
            case TracingMode::System:
                return perfetto::kSystemBackend;
            default:
                return perfetto::kSystemBackend;
        }
    }();
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
    return perfetto::Tracing::NewTrace();
}

//---------------------------------------------------------------------------//
/*!
 * Configure the session to record Celeritas track events.
 */
perfetto::TraceConfig configure_session() noexcept
{
    perfetto::protos::gen::TrackEventConfig track_event_cfg;
    track_event_cfg.add_disabled_categories("*");
    track_event_cfg.add_enabled_categories(
        celeritas::detail::perfetto_track_event_category);

    perfetto::TraceConfig cfg;
    constexpr int mb_kb = 1024;
    uint32_t buffer_size_kb = 20 * mb_kb;
    if (std::string var = celeritas::getenv("CELER_PERFETTO_BUFFER_SIZE_MB");
        !var.empty())
    {
        buffer_size_kb = std::stoul(var) * mb_kb;
    }
    cfg.add_buffers()->set_size_kb(buffer_size_kb);
    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());
    return cfg;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// Defined here as we need the full definition of perfetto::TracingSession
TracingSession::TracingSession(TracingSession&&) noexcept = default;
TracingSession& TracingSession::operator=(TracingSession&&) noexcept = default;

//---------------------------------------------------------------------------//
/*!
 * Start a system tracing session.
 */
TracingSession::TracingSession() noexcept
    : session_{initialize_session(TracingMode::System)}
{
    if (session_)
    {
        session_->Setup(configure_session());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Start an in-process tracing session.
 */
TracingSession::TracingSession(std::string_view filename) noexcept
    : session_{initialize_session(filename.empty() ? TracingMode::System
                                                   : TracingMode::InProcess)}
{
    if (session_)
    {
        if (!filename.empty())
        {
            fd_ = open(filename.data(), O_RDWR | O_CREAT | O_TRUNC, 0660);
        }
        session_->Setup(configure_session(), fd_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Block until the current session is closed.
 */
TracingSession::~TracingSession()
{
    if (session_)
    {
        if (started_)
        {
            session_->StopBlocking();
        }
        if (fd_ != -1)
        {
            close(fd_);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Start the profiling session.
 */
void TracingSession::start() noexcept
{
    if (session_)
    {
        started_ = true;
        session_->StartBlocking();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

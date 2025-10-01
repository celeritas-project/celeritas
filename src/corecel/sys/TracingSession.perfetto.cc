//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TracingSession.perfetto.cc
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
//! Supported tracing mode
enum class TracingMode
{
    in_process,  //!< Record in-process, writing to a file
    system  //!< Record in a system daemon
};

//---------------------------------------------------------------------------//
/*!
 * Initialize the session for the given mode if profiling is enabled.
 */
std::unique_ptr<perfetto::TracingSession>
initialize_session(TracingMode mode) noexcept
{
    if (!celeritas::ScopedProfiling::enabled())
    {
        return nullptr;
    }
    perfetto::TracingInitArgs args;
    args.backends |= [&] {
        switch (mode)
        {
            case TracingMode::in_process:
                return perfetto::kInProcessBackend;
            case TracingMode::system:
                [[fallthrough]];
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
/*!
 * Flush profiling events associated with the calling thread.
 *
 * In multi-threaded applications, this should be called from each
 * worker thread to ensure that their track events are correctly written.
 *
 * This is used by the Geant4 interface class \c
 * LocalTransporter which may not have access to the session instance.
 */
void TracingSession::flush() noexcept
{
    if (ScopedProfiling::enabled())
    {
        perfetto::TrackEvent::Flush();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Start a system tracing session.
 */
TracingSession::TracingSession() noexcept : TracingSession(std::string{}) {}

//---------------------------------------------------------------------------//
/*!
 * Start an in-process tracing session.
 */
TracingSession::TracingSession(std::string const& filename) noexcept
    : session_{initialize_session(filename.empty() ? TracingMode::system
                                                   : TracingMode::in_process)}
{
    if (ScopedProfiling::enabled())
    {
        CELER_ASSERT(!session_);
        if (!filename.empty())
        {
            CELER_LOG(warning)
                << R"(Skipping Perfetto tracing: profiling is disabled)";
        }
    }
    else if (session_)
    {
        auto msg = CELER_LOG(info);
        msg << "Opening Perfetto tracing session ";
        if (!filename.empty())
        {
            msg << "to " << filename;
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
            fd_ = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0660);
        }
        else
        {
            msg << "to system daemon";
        }
        session_->Setup(configure_session(), fd_);
        session_->StartBlocking();
    }
    else
    {
        CELER_LOG(warning) << "Failed to open tracing session";
    }
}

// Default move construct
TracingSession::TracingSession(TracingSession&&) = default;

//---------------------------------------------------------------------------//
/*!
 * Block until the current session is closed.
 */
TracingSession::~TracingSession()
{
    if (session_)
    {
        TracingSession::flush();
        session_->StopBlocking();
        if (fd_ != system_fd_)
        {
            close(fd_);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

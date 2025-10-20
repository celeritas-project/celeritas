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
#include "corecel/io/Logger.hh"

#include "Environment.hh"
#include "ScopedProfiling.hh"

#include "detail/TrackEvent.perfetto.hh"

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace celeritas
{
namespace
{
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

//! Forward perfetto log messages to the Celeritas logger
void perfetto_log_adapter(perfetto::LogMessageCallbackArgs args)
{
    // Map perfetto log levels to Celeritas log levels
    LogLevel celer_level = [&] {
        switch (args.level)
        {
            case perfetto::LogLev::kLogDebug:
                return LogLevel::debug;
            case perfetto::LogLev::kLogInfo:
                return LogLevel::diagnostic;
            case perfetto::LogLev::kLogImportant:
                return LogLevel::info;
            case perfetto::LogLev::kLogError:
                return LogLevel::error;
            default:
                return LogLevel::diagnostic;
        }
    }();

    // Create log provenance from perfetto's file/line info
    char const* filename = args.filename ? args.filename : "perfetto";
    world_logger()({filename, args.line}, celer_level) << args.message;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// PIMPL class
struct TracingSession::Impl
{
    static constexpr int system_fd{-1};

    int fd{system_fd};
    std::unique_ptr<perfetto::TracingSession> session;
};

//---------------------------------------------------------------------------//
/*!
 * Destroy the tracing session.
 */
void TracingSession::ImplDeleter::operator()(Impl* impl) noexcept
{
    if (impl->session)
    {
        TracingSession::flush();

        CELER_LOG(debug) << "Finalizing Perfetto";
        impl->session->StopBlocking();
        if (impl->fd != impl->system_fd)
        {
            close(impl->fd);
        }
        impl->session.reset();
        delete impl;
    }
}

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
        CELER_LOG_LOCAL(debug) << "Flushing Perfetto tracing session";
        perfetto::TrackEvent::Flush();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Start a system tracing session.
 */
TracingSession::TracingSession() : TracingSession(std::string{}) {}

//---------------------------------------------------------------------------//
/*!
 * Start an in-process tracing session.
 */
TracingSession::TracingSession(std::string const& filename)
{
    if (!ScopedProfiling::enabled())
    {
        if (!filename.empty())
        {
            CELER_LOG(warning)
                << R"(Skipping Perfetto tracing: profiling is disabled)";
        }
        return;
    }

    // Create implementation that stores tracing session and file handle
    impl_.reset(new Impl);
    perfetto::TracingInitArgs args;
    args.log_message_callback = &perfetto_log_adapter;
    if (filename.empty())
    {
        CELER_LOG(info) << "Starting Perfetto system tracing session";
        args.backends |= perfetto::kSystemBackend;
        CELER_ASSERT(impl_->fd == Impl::system_fd);
    }
    else
    {
        CELER_LOG(info) << "Saving Perfetto in-app tracing session to "
                        << filename;

        args.backends |= perfetto::kInProcessBackend;

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        impl_->fd = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0660);
    }

    // Start tracing and cancel if it failed
    perfetto::Tracing::Initialize(args);
    CELER_VALIDATE(
        perfetto::Tracing::IsInitialized(),
        << R"(failed to initialize Perfetto (re-run with CELER_ENABLE_PROFILING=0))");

    perfetto::TrackEvent::Register();
    auto session = perfetto::Tracing::NewTrace();
    CELER_VALIDATE(
        session,
        << R"(failed to open Perfetto tracing session (re-run with CELER_ENABLE_PROFILING=0))");

    session->Setup(configure_session(), impl_->fd);
    session->StartBlocking();
    impl_->session = std::move(session);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

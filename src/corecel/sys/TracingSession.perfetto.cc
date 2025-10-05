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
TracingSession::TracingSession() noexcept : TracingSession(std::string{}) {}

//---------------------------------------------------------------------------//
/*!
 * Start an in-process tracing session.
 */
TracingSession::TracingSession(std::string const& filename) noexcept
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
    if (!perfetto::Tracing::IsInitialized())
    {
        CELER_LOG(warning) << "Failed to initialize Perfetto";
        impl_.reset();
        return;
    }

    perfetto::TrackEvent::Register();
    auto session = perfetto::Tracing::NewTrace();
    if (!session)
    {
        CELER_LOG(warning) << "Failed to open Perfetto tracing session";
        impl_.reset();
        return;
    }

    session->Setup(configure_session(), impl_->fd);
    session->StartBlocking();
    impl_->session = std::move(session);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/PerfettoSession.cc
//! \brief RAII class for managing a perfetto session and its resources.
//---------------------------------------------------------------------------//
#include "PerfettoSession.hh"

#include <fcntl.h>
#include <perfetto.h>

#include "ScopedProfiling.hh"

#include "detail/TrackEvent.perfetto.hh"

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace
{
using celeritas::ProfilingBackend;

std::unique_ptr<perfetto::TracingSession>
initialize_session(ProfilingBackend backend)
{
    if (!celeritas::use_profiling())
    {
        return nullptr;
    }
    perfetto::TracingInitArgs args;
    args.backends |= [&] {
        switch (backend)
        {
            case ProfilingBackend::InProcess:
                return perfetto::kInProcessBackend;
            case ProfilingBackend::System:
                return perfetto::kSystemBackend;
            default:
                return perfetto::kSystemBackend;
        }
    }();
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
    return perfetto::Tracing::NewTrace();
}

perfetto::TraceConfig configure_session()
{
    perfetto::protos::gen::TrackEventConfig track_event_cfg;
    track_event_cfg.add_disabled_categories("*");
    track_event_cfg.add_enabled_categories(
        celeritas::detail::perfetto_track_event_category);
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(1024 * 512);
    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());
    return cfg;
}
}  // namespace

namespace celeritas
{

PerfettoSession::PerfettoSession()
    : session_{initialize_session(ProfilingBackend::System)}
{
    if (use_profiling())
    {
        session_->Setup(configure_session());
    }
}

PerfettoSession::PerfettoSession(std::string_view filename)
    : session_{initialize_session(ProfilingBackend::InProcess)}, fd_{[&] {
        return use_profiling()
                   ? open(filename.data(), O_RDWR | O_CREAT | O_TRUNC, 0660)
                   : -1;
    }()}
{
    if (use_profiling())
    {
        session_->Setup(configure_session(), fd_);
    }
}

PerfettoSession::~PerfettoSession()
{
    if (use_profiling())
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
void PerfettoSession::start()
{
    if (use_profiling())
    {
        started_ = true;
        session_->Start();
    }
}
}  // namespace celeritas
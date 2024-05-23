//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TracingSession.hh
//! \brief RAII class for managing a perfetto session and its resources.
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "celeritas_config.h"
#include "corecel/Macros.hh"

namespace perfetto
{
class TracingSession;
}

namespace celeritas
{

enum class ProfilingBackend : uint32_t
{
    InProcess,
    System
};

#if CELERITAS_USE_PERFETTO
class TracingSession
{
  public:
    TracingSession();
    explicit TracingSession(std::string_view);
    ~TracingSession();

    void start();
    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DEFAULT_MOVE_DELETE_COPY(TracingSession);
    //!@}

  private:
    bool started_{false};
    std::unique_ptr<perfetto::TracingSession> session_;
    int fd_{-1};
};
#else

//! noop
class TracingSession
{
  public:
    TracingSession() = default;
    explicit TracingSession(std::string_view) {}

    void start() {};
};
#endif

}  // namespace celeritas

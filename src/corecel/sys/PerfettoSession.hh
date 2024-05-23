//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/PerfettoSession.hh
//! \brief RAII class for managing a perfetto session and its resources.
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string_view>

#include "celeritas_config.h"
#include "corecel/Macros.hh"

#include "PerfettoSession.hh"

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

class PerfettoSession
{
  public:
    explicit PerfettoSession();
    PerfettoSession(std::string_view);
    ~PerfettoSession();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DEFAULT_MOVE_DELETE_COPY(PerfettoSession);
    //!@}
    void start() const;

  private:
    std::unique_ptr<perfetto::TracingSession> session_;
    int fd_{-1};
};

}  // namespace celeritas

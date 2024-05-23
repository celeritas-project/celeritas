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
// FORWARD DECLARATION
class TracingSession;

//---------------------------------------------------------------------------//
}  // namespace perfetto

namespace celeritas
{

//! Supported tracing mode
enum class TracingMode : uint32_t
{
    InProcess,  //!< Record in-process, writting to a file
    System  //!< Record in a system daemon
};

#if CELERITAS_USE_PERFETTO
/*!
 * RAII wrapper for a tracing session.
 *
 * Constructors will only configure an initialize the session. It needs to
 * be started explicitely by calling \c TracingSession::start
 * Only a single tracing mode is supported. If you are only interested in
 * application-level events (\c ScopedProfiling and \c Counter),
 * then the in-process mode is sufficient and is enabled by providing the
 * trace data filename to the constructor.
 *
 * If no filename is provided, start a system tracing session which records
 * both application-level events and kernel events. Root privilege and
 * Linux ftrace https://kernel.org/doc/Documentation/trace/ftrace.txt are
 * required. To start the system daemons using the perfetto backend,
 * see https://perfetto.dev/docs/quickstart/linux-tracing#capturing-a-trace
 *
 * TODO: Support multiple tracing mode.
 */
class TracingSession
{
  public:
    // Configure a system session recording to a daemon
    TracingSession();
    // Configure an in-process session recording to filename
    explicit TracingSession(std::string_view filename);
    // Terminate thte session and close open files
    ~TracingSession();

    // Start the profiling session
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

/*!
 * Noop class if Perfetto is  disabled
 */
class TracingSession
{
  public:
    // noop
    TracingSession() = default;
    // noop
    explicit TracingSession(std::string_view) {}

    // noop
    void start() {};
};
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

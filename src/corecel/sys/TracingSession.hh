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

//---------------------------------------------------------------------------//
// Forward declarations
//---------------------------------------------------------------------------//

namespace perfetto
{
//---------------------------------------------------------------------------//
#if CELERITAS_USE_PERFETTO
class TracingSession;
#else
//! Dummy as celeritas::TracingSession::~TracingSession needs the definition
class TracingSession
{
};
#endif

//---------------------------------------------------------------------------//
}  // namespace perfetto

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Supported tracing mode
enum class TracingMode : uint32_t
{
    InProcess,  //!< Record in-process, writting to a file
    System  //!< Record in a system daemon
};

/*!
 * RAII wrapper for a tracing session.
 *
 * Constructors will only configure and initialize the session. It needs to
 * be started explicitly by calling \c TracingSession::start
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

    // Terminate the session and close open files
    ~TracingSession();

    // Start the profiling session
    void start();

    //! Prevent copying but allow moving, following \c std::unique_ptr
    //! semantics
    CELER_DEFAULT_MOVE_DELETE_COPY(TracingSession);

  private:
    [[maybe_unused]] bool started_{false};
    std::unique_ptr<perfetto::TracingSession> session_;
    [[maybe_unused]] int fd_{-1};
};

#if !CELERITAS_USE_PERFETTO

inline TracingSession::TracingSession() = default;

inline TracingSession::TracingSession(std::string_view) {}

inline TracingSession::~TracingSession() = default;

inline void TracingSession::start() {}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

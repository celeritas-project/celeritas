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

#include "corecel/Config.hh"

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
//! Dummy as celeritas::TracingSession::~TracingSession needs a definition
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

//---------------------------------------------------------------------------//
/*!
 * RAII wrapper for a tracing session.
 *
 * Constructors will only configure and initialize the session. It needs to
 * be started explicitly by calling \c TracingSession::start
 * Only a single tracing mode is supported. If you are only interested in
 * application-level events (\c ScopedProfiling and \c trace_counter),
 * then the in-process mode is sufficient and is enabled by providing the
 * trace data filename to the constructor. When using in-process tracing,
 * the buffer size can be configured by setting \c
 * CELER_PERFETTO_BUFFER_SIZE_MB.
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
    TracingSession() noexcept;

    // Configure an in-process session recording to filename
    explicit TracingSession(std::string_view filename) noexcept;

    // Terminate the session and close open files
    ~TracingSession();

    // Start the profiling session
    void start() noexcept;

    //! Prevent copying but allow moving, following \c std::unique_ptr
    //! semantics
    TracingSession(TracingSession const&) = delete;
    TracingSession& operator=(TracingSession const&) = delete;
    TracingSession(TracingSession&&) noexcept;
    TracingSession& operator=(TracingSession&&) noexcept;

  private:
    bool started_{false};
    std::unique_ptr<perfetto::TracingSession> session_;
    int fd_{-1};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELERITAS_USE_PERFETTO

inline TracingSession::TracingSession() noexcept = default;

inline TracingSession::TracingSession(std::string_view) noexcept {}

inline TracingSession::~TracingSession() = default;

inline void TracingSession::start() noexcept
{
    CELER_DISCARD(started_);
    CELER_DISCARD(fd_);
}

#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

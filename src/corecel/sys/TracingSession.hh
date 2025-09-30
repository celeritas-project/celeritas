//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TracingSession.hh
//! \brief RAII class for managing a perfetto session and its resources.
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

namespace perfetto
{
class TracingSession;
}  // namespace perfetto

namespace celeritas
{
//---------------------------------------------------------------------------//
// Flush perfetto track events without requiring a TracingSession instance.
void flush_tracing() noexcept;

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
    explicit TracingSession(std::string const& filename) noexcept;

    // Terminate the session and close open files
    ~TracingSession();

    // Start the profiling session
    void start() noexcept;

    // Flush the track events associated with the calling thread
    void flush() noexcept;

    CELER_DELETE_COPY_MOVE(TracingSession);

  private:
    static constexpr int system_fd_{-1};
    struct Deleter
    {
        void operator()(perfetto::TracingSession*);
    };

    bool started_{false};
    std::unique_ptr<perfetto::TracingSession, Deleter> session_;
    int fd_{system_fd_};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELERITAS_USE_PERFETTO
inline void flush_tracing() noexcept {}
inline TracingSession::TracingSession() noexcept = default;
inline TracingSession::TracingSession(std::string const&) noexcept {}
inline TracingSession::~TracingSession() = default;
inline void TracingSession::start() noexcept
{
    CELER_DISCARD(started_);
    CELER_DISCARD(fd_);
}
inline void TracingSession::flush() noexcept {}
inline void TracingSession::Deleter::operator()(perfetto::TracingSession*)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

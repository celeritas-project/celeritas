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
// DEPRECATED: remove in v1.0
[[deprecated]] inline void flush_tracing() noexcept;

//---------------------------------------------------------------------------//
/*!
 * RAII wrapper for a tracing session.
 *
 * Constructors will only configure and initialize the session.
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
 * TODO: Support multiple tracing modes.
 */
class TracingSession
{
  public:
    // Flush the track events associated with the calling thread
    static void flush() noexcept;

    // Configure a system session recording to a daemon
    TracingSession() noexcept;

    // Configure an in-process session recording to filename
    explicit TracingSession(std::string const& filename) noexcept;

    // Prevent copy/assignment; destructor deallocates forward-declared pointer
    TracingSession(TracingSession const&) = delete;
    TracingSession& operator=(TracingSession const&) = delete;
    TracingSession& operator=(TracingSession&&) = delete;

    // Default move constructor to handle unique_ptr hidden implementation
    TracingSession(TracingSession&&);

    // Terminate the session and close open files
    ~TracingSession();

    // Start the profiling session (DEPRECATED: remove in v1.0)
    //! The session is now started on construction; this is now a null-op
    [[deprecated]] void start() noexcept {}

  private:
    static constexpr int system_fd_{-1};

#if CELERITAS_USE_PERFETTO
    std::unique_ptr<perfetto::TracingSession> session_;
    int fd_{system_fd_};
#endif
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

inline void flush_tracing() noexcept
{
    return TracingSession::flush();
}

#if !CELERITAS_USE_PERFETTO
inline TracingSession::TracingSession() noexcept = default;
inline TracingSession::TracingSession(std::string const& s) noexcept
{
    if (!s.empty())
    {
        CELER_LOG(warning)
            << R"(Ignoring tracing session file: Perfetto is disabled)";
    }
}
inline TracingSession::TracingSession(TracingSession&&) = default;
inline TracingSession::~TracingSession() = default;
inline void TracingSession::flush() noexcept {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

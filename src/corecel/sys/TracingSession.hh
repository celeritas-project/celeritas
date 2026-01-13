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
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Flush perfetto track events without requiring a TracingSession instance.
// DEPRECATED: remove in v1.0
[[deprecated]] inline void flush_tracing() noexcept;

//---------------------------------------------------------------------------//
/*!
 * Record Perfetto events during the lifetime of this object.
 *
 * This RAII class manages a Perfetto tracing session.
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
 * \note Profiling is disabled unless the \c CELER_ENABLE_PROFILING environment
 * variable is set; see celeritas::ScopedProfiling.
 */
class TracingSession
{
  public:
    // Flush the track events associated with the calling thread
    static void flush() noexcept;

    // Configure a system session recording to a daemon
    TracingSession();

    // Configure an in-process session recording to filename
    explicit TracingSession(std::string const& filename);

    // Start the profiling session (DEPRECATED: remove in v1.0)
    //! The session is now started on construction; this is now a null-op
    [[deprecated]] void start() noexcept {}

    //! Return whether profiling is enabled
    explicit operator bool() const { return static_cast<bool>(impl_); }

  private:
    struct Impl;
    struct ImplDeleter
    {
        void operator()(Impl*) noexcept;
    };
    std::unique_ptr<Impl, ImplDeleter> impl_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

inline void flush_tracing() noexcept
{
    return TracingSession::flush();
}

#if !CELERITAS_USE_PERFETTO
inline TracingSession::TracingSession() = default;
inline TracingSession::TracingSession(std::string const& s)
{
    if (!s.empty())
    {
        CELER_LOG(warning)
            << R"(Ignoring tracing session file: Perfetto is disabled)";
    }
}
inline void TracingSession::flush() noexcept {}

inline void TracingSession::ImplDeleter::operator()(Impl*) noexcept
{
    CELER_UNREACHABLE;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

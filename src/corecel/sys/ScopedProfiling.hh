//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <string>

#include "celeritas_config.h"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"

#include "Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Whether profiling is enabled
bool use_profiling();

//---------------------------------------------------------------------------//
/*!
 * Input arguments for the nvtx implementation.
 */
struct ScopedProfilingInput
{
    std::string_view name;  //!< Name of the range
    uint32_t color{};  //!< ARGB
    int32_t payload{};  //!< User data
    uint32_t category{};  //!< Category, used to group ranges together

    ScopedProfilingInput(std::string_view n) : name{n} {}
};

//---------------------------------------------------------------------------//
/*!
 * RAII class for scoped profiling.
 *
 * Implementations should support multithreaded context where each thread have
 * one or more alive instance of this class.
 *
 * This is useful for wrapping specific code fragment in a range for profiling,
 * e.g. ignoring of VecGeom instantiation kernels, profiling a specific action
 * or loop on the CPU.
 *
 * \note The nvtx implementation of \c ScopedProfiling only does something when
 * the application using Celeritas is ran through a tool that supports nvtx,
 * e.g. nsight compute with the --nvtx argument. If this is not the case, API
 * calls to nvtx are no-ops.
 *
 * \note The AMD roctx implementation requires the roctx library, which may not
 * be available on all systems.
 *
 * \note The CPU implementation requires Perfetto. It is not supported when
 * Celeritas is built with device support (CUDA/HIP)
 */
class ScopedProfiling
{
  public:
    //!@{
    //! \name Type aliases
    using Input = ScopedProfilingInput;
    //!@}

  public:
    // Activate profiling with options
    explicit inline ScopedProfiling(Input const& input);
    // Activate profiling with just a name
    explicit inline ScopedProfiling(std::string_view name);

    // Deactivate profiling
    inline ~ScopedProfiling();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedProfiling);
    //!@}

  private:
    bool activated_;

    void activate(Input const& input) noexcept;
    void deactivate() noexcept;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Activate device profiling with options.
 */
ScopedProfiling::ScopedProfiling(Input const& input)
    : activated_{use_profiling()}
{
    if (activated_)
    {
        ScopedProfiling::activate(input);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Activate device profiling with just a name.
 */
ScopedProfiling::ScopedProfiling(std::string_view name)
    : ScopedProfiling{Input{name}}
{
}

//---------------------------------------------------------------------------//
/*!
 * Deactivate a profiling scope.
 */
ScopedProfiling::~ScopedProfiling()
{
    if (activated_)
    {
        ScopedProfiling::deactivate();
    }
}

#if !CELER_USE_DEVICE && !CELERITAS_USE_PERFETTO
inline void ScopedProfiling::activate(Input const&) noexcept
{
    CELER_UNREACHABLE;
}
inline void ScopedProfiling::deactivate() noexcept
{
    CELER_UNREACHABLE;
}
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas

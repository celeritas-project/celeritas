//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
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

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input arguments for the nvtx implementation.
 */
struct ScopedProfilingInput
{
    std::string name;  //!< Name of the range
    uint32_t color{0xFF00FF00u};  //!< ARGB
    int32_t payload{0};  //!< User data
    uint32_t category{0};  //!< Category, used to group ranges together

    ScopedProfilingInput(std::string const& name) : name{name} {}
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
 */
class ScopedProfiling
{
  public:
    //!@{
    //! \name Type aliases
    using Input = ScopedProfilingInput;
    //!@}

  public:
#if CELER_USE_DEVICE
    // Whether profiling is enabled
    static bool enable_profiling();
#else
    // Profiling is never enabled if CUDA isn't available
    constexpr static bool enable_profiling() { return false; }
#endif

    // Activate profiling with options
    explicit inline ScopedProfiling(Input const& input);
    // Activate profiling with just a name
    explicit inline ScopedProfiling(std::string const& name);

    // Deactivate profiling
    inline ~ScopedProfiling();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedProfiling);
    //!@}

  private:
    bool activated_;

    void activate_(Input const& input) noexcept;
    void deactivate_() noexcept;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Activate nvtx profiling with options.
 */
ScopedProfiling::ScopedProfiling(Input const& input)
    : activated_{ScopedProfiling::enable_profiling()}
{
    if (activated_)
    {
        this->activate_(input);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Activate nvtx profiling with just a name.
 */
ScopedProfiling::ScopedProfiling(std::string const&)
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
        this->deactivate_();
    }
}

#if !CELER_USE_DEVICE
inline void activate_(Input const&)
{
    CELER_ASSERT_UNREACHABLE();
}
inline void deactivate_()
{
    CELER_ASSERT_UNREACHABLE();
}
#endif
//---------------------------------------------------------------------------//
}  // namespace celeritas

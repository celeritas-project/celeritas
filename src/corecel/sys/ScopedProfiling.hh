//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedProfiling.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstdint>
#include <string>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"

#include "Environment.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Input arguments for the most richly annotated implementation (NVTX).
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
 * Enable and annotate performance profiling during the lifetime of this class.
 *
 * This RAII class annotates the profiling output so that, during its scope,
 * events and timing are associated with the given name. For use cases inside
 * separate begin/end functions of a class (often seen in Geant4), use
 * \c std::optional to start and end the class lifetime.
 *
 * This is useful for wrapping specific code fragment in a range for profiling,
 * e.g., ignoring of VecGeom instantiation kernels, or profiling a specific
 * action.  It is very similar to the [NVTX
 * `scoped_range`](https://nvidia.github.io/NVTX/doxygen-cpp/#scoped_range).
 *
 * Example: \code
 * void do_program()
 * {
 *     do_setup()
 *     ScopedProfiling profile_this{"run"};
 *     do_run();
 * }
 * \endcode
 *
 * Caveats:
 * - The Nvidia/CUDA implementation of \c ScopedProfiling only does something
 *   when the application using Celeritas is run through a tool that supports
 *   NVTX, e.g., nsight compute with the --nvtx argument. If this is not the
 *   case, API calls to nvtx are no-ops.
 * - The HIP/AMD ROCTX implementation requires the roctx library, which may not
 *   be available on all systems.
 * - The CPU implementation requires Perfetto. It is not available when
 *   Celeritas is built with device support (CUDA/HIP).
 *
 * \internal All profiling library implementations must support multithreaded
 * contexts since each thread may have one or more active instances of this
 * class.
 */
class ScopedProfiling
{
  public:
    //!@{
    //! \name Type aliases
    using Input = ScopedProfilingInput;
    //!@}

  public:
    // Whether profiling is enabled
    static bool enabled();

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
    : activated_{ScopedProfiling::enabled()}
{
    if (activated_)
    {
        this->activate(input);
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
        this->deactivate();
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

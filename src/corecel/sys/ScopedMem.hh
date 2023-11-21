//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedMem.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string_view>

#include "corecel/cont/InitializedValue.hh"

#include "MemRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Record the change in memory usage between construction and destruction.
 *
 * \code
    {
      ScopedMem record_mem("create objects");
      this->create_stuff();
    }
   \endcode
 *
 * This class is \em not thread safe because it writes to a shared global
 * index.  In a multithreaded environment a "null" scoped memory can be used:
 * \code
 * {
     auto record_mem = (stream_id == StreamId{0} ? ScopedMem{"label"}
                                                 : ScopedMem{});
     this->do_stuff();
 * }
 * \endcode
 *
 * \note The memory reported will likely only be valid if running a single
 * task on the GPU, because the start and stop values are per \em GPU rather
 * than per \em process. Be wary of the result.
 */
class ScopedMem
{
  public:
    // Default constructor for "null-op" recording
    ScopedMem() = default;

    // Construct with name and registries
    ScopedMem(std::string_view label, MemRegistry* registry);

    //! Construct with name and default registry
    explicit ScopedMem(std::string_view label)
        : ScopedMem{label, &celeritas::mem_registry()}
    {
    }

    // Register data on destruction
    ~ScopedMem();

    //!@{
    //! Default move assign and construct; no copying
    ScopedMem(ScopedMem&&) = default;
    ScopedMem(ScopedMem const&) = delete;
    ScopedMem& operator=(ScopedMem&&) = default;
    ScopedMem& operator=(ScopedMem const&) = delete;
    //!@}

  private:
    using value_type = KibiBytes::value_type;

    InitializedValue<MemRegistry*> registry_;
    MemUsageId id_;
    value_type cpu_start_hwm_{0};
    value_type gpu_start_used_{0};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

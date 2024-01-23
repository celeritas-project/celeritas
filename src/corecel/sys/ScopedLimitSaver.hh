//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedLimitSaver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Save and restore CUDA limits inside the current scope.
 *
 * This is useful for calling poorly behaved external libraries that change
 * CUDA limits unexpectedly.
 */
class ScopedLimitSaver
{
  public:
    ScopedLimitSaver();
    ~ScopedLimitSaver();

    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedLimitSaver);
    //!@}

  private:
#if CELER_USE_DEVICE
    using Limit_t = CELER_DEVICE_PREFIX(Limit);
#else
    using Limit_t = int;
#endif

    static Array<Limit_t, 2> const cuda_attrs_;
    static Array<char const*, 2> const cuda_attr_labels_;
    Array<std::size_t, 2> orig_limits_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
#if !CELERITAS_USE_CUDA
// Construction is a null-op since we only save with CUDA
inline ScopedLimitSaver::ScopedLimitSaver()
{
    CELER_DISCARD(orig_limits_);
#    if CELERITAS_USE_HIP
    CELER_NOT_IMPLEMENTED("HIP limit restoration");
#    endif
}

// Destruction is a null-op since we only save with CUDA
inline ScopedLimitSaver::~ScopedLimitSaver() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

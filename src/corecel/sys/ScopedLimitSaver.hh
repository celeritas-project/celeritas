//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ScopedLimitSaver.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>

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
 * CUDA limits unexpectedly. We don't use this with HIP because it's only
 * needed at present for VecGeom.
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
    if constexpr (CELERITAS_USE_HIP)
    {
        CELER_NOT_IMPLEMENTED("HIP limit restoration");
    }
}

// Destruction is a null-op since we only save with CUDA
inline ScopedLimitSaver::~ScopedLimitSaver() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

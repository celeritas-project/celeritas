//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_USE_COVFIE || __DOXYGEN__

#    include "corecel/Macros.hh"
#    include "corecel/Types.hh"

#    include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Primary template; specializations for value/host, const_reference/host, and
// value/device are in RZMapFieldData.covfie.hh (requires C++20 and covfie).
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData;

// const_reference/device is defined here (no covfie types needed) so that
// public headers included by C++17 TUs can use
// NativeCRef<RZMapFieldParamsData> on device without pulling in covfie's C++20
// headers.
template<>
struct RZMapFieldParamsData<Ownership::const_reference, MemSpace::device>
{
    CELER_FUNCTION explicit operator bool() const { return field_view; }

    FieldDriverOptions options;
    //! Opaque pointer to view_t in device memory; cast to view_t const* in .cu
    void const* field_view{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#else

#    include "corecel/Types.hh"

#    include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Minimal stub so that RZMapFieldParamsData is always a complete type.
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData
{
    FieldDriverOptions options;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // CELERITAS_USE_COVFIE

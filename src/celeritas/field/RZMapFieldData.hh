//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Types.hh"

#include "FieldDriverOptions.hh"

#if CELERITAS_USE_COVFIE || __DOXYGEN__
#    include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData;

// Specialized and type-erased covfie field_view to hide c++20 dependency
template<>
struct RZMapFieldParamsData<Ownership::const_reference, MemSpace::device>
{
    CELER_FUNCTION explicit operator bool() const { return field_view; }

    RZMapFieldParamsData& operator=(
        RZMapFieldParamsData<Ownership::value, MemSpace::device> const& other);

    FieldDriverOptions options;
    real_type min_r{};
    real_type max_r{};
    real_type min_z{};
    real_type max_z{};
    // underlying type: CovfieRZFieldTraits<MemSpace::device>::field_t:view_t
    void const* field_view{};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
#else

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

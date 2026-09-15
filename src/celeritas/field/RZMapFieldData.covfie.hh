//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldData.covfie.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/DeviceVector.hh"
#include "celeritas/Types.hh"

#include "RZMapFieldData.hh"  // primary template + const_reference/device spec

#include "detail/CovfieRZFieldTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<MemSpace M>
struct RZMapFieldParamsDataBase
{
    using field_t = typename detail::CovfieRZFieldTraits<M>::field_t;
    using view_t = typename field_t::view_t;

    FieldDriverOptions options;
    real_type min_r{};
    real_type max_r{};
    real_type min_z{};
    real_type max_z{};
};

// Specializations for value/host, const_reference/host, and value/device are
// defined below. const_reference/device is in RZMapFieldData.hh (no covfie
// dependency, safe for C++17 TUs).

template<>
struct RZMapFieldParamsData<Ownership::value, MemSpace::host>
    : RZMapFieldParamsDataBase<MemSpace::host>
{
    CELER_FUNCTION view_t get_view() const { return view_t(*field); }

    CELER_FUNCTION explicit operator bool() const { return field.get(); }

    std::unique_ptr<field_t> field;
};
template<>
struct RZMapFieldParamsData<Ownership::const_reference, MemSpace::host>
    : RZMapFieldParamsDataBase<MemSpace::host>
{
    CELER_FUNCTION view_t const& get_view() const { return field_view; }

    CELER_FUNCTION explicit operator bool() const { return true; }

    view_t field_view;
};

template<>
struct RZMapFieldParamsData<Ownership::value, MemSpace::device>
    : RZMapFieldParamsDataBase<MemSpace::device>
{
    view_t const& get_view() const { return field_view.device_ref()[0]; }

    explicit operator bool() const
    {
        return field.get() && field_view.size() == 1;
    }

    RZMapFieldParamsData& operator=(
        RZMapFieldParamsData<Ownership::value, MemSpace::host> const& other)
    {
        using host_field_t
            = detail::CovfieRZFieldTraits<MemSpace::host>::field_t;
        if constexpr (!std::is_same_v<field_t, host_field_t>)
        {
            // Use covfie's cross-type field constructor: propagates through
            // all transformer layers (affine->clamp->linear->strided), with
            // strided's cross-type constructor performing the H2D transfer
            // into cuda_device_array at the bottom of the chain.
            field = std::make_unique<field_t>(*other.field);

            // Store view_t in device memory; pass pointer to kernel
            field_view = DeviceVector<view_t>{1};
            field_view.copy_to_device(make_span<view_t const>({{*field}}));
        }
        else
        {
            field = std::make_unique<field_t>(*other.field);
        }
        options = other.options;
        min_r = other.min_r;
        max_r = other.max_r;
        min_z = other.min_z;
        max_z = other.max_z;
        return *this;
    }

    std::unique_ptr<field_t> field;
    DeviceVector<view_t> field_view;
};

//---------------------------------------------------------------------------//
/*!
 * Convert shared parameter data to a covfie field view.
 */
template<MemSpace M>
CELER_FUNCTION inline auto get_rz_map_field_view(
    RZMapFieldParamsData<Ownership::const_reference, M> const& params) ->
    typename detail::CovfieRZFieldTraits<M>::field_t::view_t const&
{
    if constexpr (M == MemSpace::host)
    {
        return params.get_view();
    }
    else
    {
        using view_t = typename detail::CovfieRZFieldTraits<M>::field_t::view_t;
        CELER_EXPECT(params.field_view);
        return *static_cast<view_t const*>(params.field_view);
    }
}

//---------------------------------------------------------------------------//
inline auto
RZMapFieldParamsData<Ownership::const_reference, MemSpace::device>::operator=(
    RZMapFieldParamsData<Ownership::value, MemSpace::device> const& other)
    -> RZMapFieldParamsData&
{
    field_view = static_cast<void const*>(&other.get_view());
    options = other.options;
    min_r = other.min_r;
    max_r = other.max_r;
    min_z = other.min_z;
    max_z = other.max_z;
    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

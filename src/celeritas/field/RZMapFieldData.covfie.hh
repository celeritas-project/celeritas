//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldData.covfie.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/DeviceVector.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/FieldDriverOptions.hh"

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
};

// We need to specialize this for every combination of ownership and memory
// space to handle covfie move, ownership semantics.
template<Ownership W, MemSpace M>
struct RZMapFieldParamsData;

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
        return *this;
    }

    std::unique_ptr<field_t> field;
    DeviceVector<view_t> field_view;
};
template<>
struct RZMapFieldParamsData<Ownership::const_reference, MemSpace::device>
    : RZMapFieldParamsDataBase<MemSpace::device>
{
    CELER_FUNCTION view_t const& get_view() const { return *field_view; }

    CELER_FUNCTION explicit operator bool() const { return field_view; }

    RZMapFieldParamsData& operator=(
        RZMapFieldParamsData<Ownership::value, MemSpace::device> const& other)
    {
        field_view = &other.field_view.device_ref()[0];
        options = other.options;
        return *this;
    }

    view_t const* field_view{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

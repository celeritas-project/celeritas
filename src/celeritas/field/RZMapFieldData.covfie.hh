//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldData.covfie.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>

#include "corecel/Config.hh"

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

    //! Grid bounds for validity check
    real_type min_r{};
    real_type max_r{};
    real_type min_z{};
    real_type max_z{};
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
    CELER_FUNCTION view_t const& get_view() const
    {
        return field_view.device_ref()[0];
    }

    CELER_FUNCTION explicit operator bool() const
    {
        return field.get() && field_view.size() == 1;
    }

    RZMapFieldParamsData& operator=(
        RZMapFieldParamsData<Ownership::value, MemSpace::host> const& other)
    {
        if constexpr (!std::is_same_v<
                          field_t,
                          detail::CovfieRZFieldTraits<MemSpace::host>::field_t>)
        {
            if constexpr (CELERITAS_USE_HIP)
            {
                // No texture memory support: simply copy from the host field
                field = std::make_unique<field_t>(*other.field);
            }
            else
            {
                auto const& host_backend = other.field->backend();
                auto const& strided_backend
                    = host_backend.get_backend().get_backend().get_backend();
                field = std::make_unique<field_t>(covfie::make_parameter_pack(
                    host_backend.get_configuration(), strided_backend));
            }
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
        min_r = other.min_r;
        max_r = other.max_r;
        min_z = other.min_z;
        max_z = other.max_z;
        return *this;
    }

    view_t const* field_view;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

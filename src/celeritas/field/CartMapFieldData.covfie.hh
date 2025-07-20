//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldData.covfie.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/DeviceVector.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/FieldDriverOptions.hh"

#include "detail/CovfieFieldTraits.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<MemSpace M>
struct CartMapFieldParamsDataBase
{
    using field_t = typename detail::CovfieFieldTraits<M>::field_t;
    using view_t = typename field_t::view_t;

    FieldDriverOptions options;
};

// We need to specialize this for every combination of ownership and memory
// space to handle covfie move, ownership semantics.
template<Ownership W, MemSpace M>
struct CartMapFieldParamsData;

template<>
struct CartMapFieldParamsData<Ownership::value, MemSpace::host>
    : CartMapFieldParamsDataBase<MemSpace::host>
{
    CELER_FUNCTION view_t get_view() const { return view_t(*field); }

    CELER_FUNCTION explicit operator bool() const { return field.get(); }

    std::unique_ptr<field_t> field;
};
template<>
struct CartMapFieldParamsData<Ownership::const_reference, MemSpace::host>
    : CartMapFieldParamsDataBase<MemSpace::host>
{
    CELER_FUNCTION view_t const& get_view() const { return field_view; }

    CELER_FUNCTION explicit operator bool() const { return true; }

    view_t field_view;
};

template<>
struct CartMapFieldParamsData<Ownership::value, MemSpace::device>
    : CartMapFieldParamsDataBase<MemSpace::device>
{
    CELER_FUNCTION view_t const& get_view() const
    {
        return field_view.device_ref()[0];
    }

    CELER_FUNCTION explicit operator bool() const
    {
        return field.get() && field_view.size() == 1;
    }

    CartMapFieldParamsData& operator=(
        CartMapFieldParamsData<Ownership::value, MemSpace::host> const& other)
    {
        if constexpr (!std::is_same_v<
                          field_t,
                          detail::CovfieFieldTraits<MemSpace::host>::field_t>)
        {
            field = std::make_unique<field_t>(covfie::make_parameter_pack(
                other.field->backend().get_configuration(),
                other.field->backend().get_backend().get_backend()));
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
struct CartMapFieldParamsData<Ownership::const_reference, MemSpace::device>
    : CartMapFieldParamsDataBase<MemSpace::device>
{
    CELER_FUNCTION view_t const& get_view() const { return *field_view; }

    CELER_FUNCTION explicit operator bool() const { return field_view; }

    CartMapFieldParamsData& operator=(
        CartMapFieldParamsData<Ownership::value, MemSpace::device> const& other)
    {
        field_view = &other.field_view.device_ref()[0];
        options = other.options;
        return *this;
    }

    view_t const* field_view;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

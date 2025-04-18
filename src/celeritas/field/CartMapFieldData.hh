//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/field/FieldDriverOptions.hh"

#include "detail/CovfieFieldType.hh"

namespace celeritas
{
//! Real type for cartesian map field data
using cartmap_real_type = float;

//---------------------------------------------------------------------------//
/*!
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct CartMapFieldParamsData
{
    using real_type = cartmap_real_type;
    using field_t = CovfieFieldTrait<M>::field_t;

    // Can we use a view instead of pointer?
    ObserverPtr<field_t const, M> field;  //!< Covfie field data

    //! Field propagation and substepping tolerances
    FieldDriverOptions options;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(field);
    }
    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CartMapFieldParamsData&
    operator=(CartMapFieldParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        static_assert(M == M2,
                      "Cannot assign references between different memory "
                      "spaces");
        if constexpr (W2 == Ownership::value)
        {
            field = ObserverPtr<field_t const, M>(other.field.get());
        }
        else
        {
            field = other.field;
        }
        options = other.options;
        return *this;
    }
};

template<MemSpace M>
struct CartMapFieldParamsData<Ownership::value, M>
{
    using real_type = cartmap_real_type;
    using field_t = CovfieFieldTrait<M>::field_t;

    std::unique_ptr<field_t> field;  //!< Covfie field data

    //! Field propagation and substepping tolerances
    FieldDriverOptions options;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return field->backend().get_backend().get_backend().get_backend().m_size
               > 0;
    }
    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CartMapFieldParamsData&
    operator=(CartMapFieldParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        field = std::make_unique<field_t>(*other.field);
        options = other.options;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
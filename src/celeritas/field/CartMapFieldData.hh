//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
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

    field_t field;  //!< Covfie field data

    //! Field propagation and substepping tolerances
    FieldDriverOptions options;

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        // TODO: how to check with covfie
        return true;
    }
    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CartMapFieldParamsData&
    operator=(CartMapFieldParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        field = other.field;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
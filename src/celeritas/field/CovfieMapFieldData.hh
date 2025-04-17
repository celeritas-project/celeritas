//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CovfieMapFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"

#include "detail/CovfieFieldType.hh"

namespace celeritas
{
//! Real type for cylindrical map field data
using cylmap_real_type = float;

//---------------------------------------------------------------------------//
/*!
 * Device data for interpolating field values.
 */
template<Ownership W, MemSpace M>
struct CovfieMapFieldParamsData
{
    using real_type = cylmap_real_type;
    using field_t = CovfieFieldTrait<M>::field_t;

    field_t field;  //!< Covfie field data

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        // TODO: how to check with covfie
        return true;
    }
    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CovfieMapFieldParamsData&
    operator=(CovfieMapFieldParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        field = other.field;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
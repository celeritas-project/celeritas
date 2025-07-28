//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformFieldData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/ArrayUtils.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data and options for a uniform field.
 *
 * If \c has_field is non-empty, the field will only be present in those
 * volumes.
 */
template<Ownership W, MemSpace M>
struct UniformFieldParamsData
{
    using Real3 = Array<real_type, 3>;
    template<class T>
    using VolumeItems = celeritas::Collection<T, W, M, VolumeId>;

    Real3 field{0, 0, 0};  //!< Field strength (native units)
    FieldDriverOptions options;
    VolumeItems<char> has_field;  //!< Volumes where field is present

    //! Check whether the data is assigned
    explicit inline CELER_FUNCTION operator bool() const
    {
        return options && norm(field) > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    UniformFieldParamsData&
    operator=(UniformFieldParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        field = other.field;
        options = other.options;
        has_field = other.has_field;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

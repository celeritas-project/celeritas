//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/UnifiedReflectionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    UnifiedReflectionData ...;
   \endcode
 */
template<Ownership W, MemSpace M>
struct UnifiedReflectionData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;
    //!@}

    //! Probability grids
    Items<NonuniformGridRecord> specular_spike;
    Items<NonuniformGridRecord> specular_lobe;
    Items<NonuniformGridRecord> backscatter;

    //! Backend storage
    Items<real_type> reals;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !reals.empty() && !specular_spike.empty()
               && !specular_lobe.empty() && !backscatter.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    UnifiedReflectionData<W, M>&
    operator=(UnifiedReflectionData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        reals = other.reals;
        specular_spike = other.specular_spike;
        specular_lobe = other.specular_lobe;
        backscatter = other.backscatter;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

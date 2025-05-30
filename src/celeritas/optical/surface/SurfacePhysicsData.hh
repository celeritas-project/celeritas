//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsData.hh
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
// TYPE ALIASES
//---------------------------------------------------------------------------//

using SurfaceId = OpaqueId<struct OpticalSurface_>;

using ValueGrid = NonuniformGridRecord;
using ValudGridId = OpaqueId<ValueGrid>;

//---------------------------------------------------------------------------//
/*!
 * Scalar quantities used by optical surface physics.
 */
struct SurfacePhysicsParamsScalars
{
    //!@{
    //! \name Global surface properties for testing purposes
    real_type global_reflectivity{0.5};
    real_type global_transmittance{0.3};
    //!@}


    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return global_reflectivity >= 0 && global_transmittance >= 0;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Persistent shared optical surface physics data.
 */
template<Ownership W, MemSpace M>
struct SurfacePhysicsParamsData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using Items = Collection<T, W, M>;

    template<class T>
    using SurfaceItems = Collection<T, W, M, SurfaceId>;
    //!@}

    //! Non-templated data
    SurfacePhysicsParamsScalars scalars;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(scalars);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SurfacePhysicsParamsData<W, M>& operator=(SurfacePhysicsParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        scalars = other.scalars;
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/DistributionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/Collection.hh"
#include "corecel/random/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Data for sampling a value from a delta distribution.
 */
template<class T>
struct DeltaDistributionRecord
{
    T value{};
};

//---------------------------------------------------------------------------//
/*!
 * Data for sampling from a normal distribution.
 */
struct NormalDistributionRecord
{
    real_type mean{0};
    real_type stddev{1};
};

//---------------------------------------------------------------------------//
/*!
 * Data for sampling a point uniformly on the unit sphere.
 */
struct IsotropicDistributionRecord
{
};

//---------------------------------------------------------------------------//
/*!
 * Data for sampling a point uniformly in a box.
 */
struct UniformBoxDistributionRecord
{
    Real3 lower{0, 0, 0};
    Real3 upper{0, 0, 0};
};

//---------------------------------------------------------------------------//
/*!
 * Storage for on-device sampling from arbitrary user-selected distributions.
 */
template<Ownership W, MemSpace M>
struct DistributionParamsData
{
    //// TYPES ////

    template<class T>
    using Items = Collection<T, W, M>;
    template<class T>
    using OnedDistributionItems = Collection<T, W, M, OnedDistributionId>;
    template<class T>
    using ThreedDistributionItems = Collection<T, W, M, ThreedDistributionId>;

    //// DATA ////

    //! 1D distributions
    OnedDistributionItems<OnedDistributionType> oned_types;
    OnedDistributionItems<size_type> oned_indices;

    Items<DeltaDistributionRecord<real_type>> delta_real;
    Items<NormalDistributionRecord> normal;

    //! 3D distributions
    ThreedDistributionItems<ThreedDistributionType> threed_types;
    ThreedDistributionItems<size_type> threed_indices;

    Items<DeltaDistributionRecord<Array<real_type, 3>>> delta_real3;
    Items<IsotropicDistributionRecord> isotropic;
    Items<UniformBoxDistributionRecord> uniform_box;

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return (!oned_types.empty() || !threed_types.empty())
               && oned_indices.size() == oned_types.size()
               && threed_indices.size() == threed_types.size();
    }

    //! Assign from another memory/ownership specialization
    template<Ownership W2, MemSpace M2>
    DistributionParamsData&
    operator=(DistributionParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);

        oned_types = other.oned_types;
        oned_indices = other.oned_indices;
        delta_real = other.delta_real;
        normal = other.normal;

        threed_types = other.threed_types;
        threed_indices = other.threed_indices;
        delta_real3 = other.delta_real3;
        isotropic = other.isotropic;
        uniform_box = other.uniform_box;

        CELER_ENSURE(static_cast<bool>(*this) == static_cast<bool>(other));
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

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

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Identifier for a distribution returning a single real type
using OnedDistributionId = OpaqueId<OnedDistributionType>;

//! Identifier for a distribution returning a length three array
using ThreedDistributionId = OpaqueId<ThreedDistributionType>;

//---------------------------------------------------------------------------//
/*!
 * Data for sampling a single value from a delta distribution.
 */
struct DeltaOnedDistributionRecord
{
    real_type value{};
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
 * Data for sampling a point from a delta distribution.
 */
struct DeltaThreedDistributionRecord
{
    using Real3 = Array<real_type, 3>;

    Real3 value{0, 0, 0};
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
    using Real3 = Array<real_type, 3>;

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

    Items<DeltaOnedDistributionRecord> delta_oned_records;
    Items<NormalDistributionRecord> normal_records;

    //! 3D distributions
    ThreedDistributionItems<ThreedDistributionType> threed_types;
    ThreedDistributionItems<size_type> threed_indices;

    Items<DeltaThreedDistributionRecord> delta_threed_records;
    Items<IsotropicDistributionRecord> isotropic_records;
    Items<UniformBoxDistributionRecord> uniform_box_records;

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
        delta_oned_records = other.delta_oned_records;
        normal_records = other.normal_records;

        threed_types = other.threed_types;
        threed_indices = other.threed_indices;
        delta_threed_records = other.delta_threed_records;
        isotropic_records = other.isotropic_records;
        uniform_box_records = other.uniform_box_records;

        CELER_ENSURE(static_cast<bool>(*this) == static_cast<bool>(other));
        return *this;
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

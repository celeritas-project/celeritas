//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/DistributionInserter.cc
//---------------------------------------------------------------------------//
#include "DistributionInserter.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/ArrayUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a reference to mutable host data.
 */
DistributionInserter::DistributionInserter(HostVal<DistributionParamsData>& data)
    : data_(data)
{
}

//---------------------------------------------------------------------------//
/*!
 * Add data for sampling a value from a 1D delta distribution.
 */
OnedDistributionId
DistributionInserter::operator()(inp::DeltaDistribution<double> const& d)
{
    DeltaDistributionRecord<real_type> record;
    record.value = d.value;
    auto id = CollectionBuilder{&data_.delta_real}.push_back(record);
    return (*this)(OnedDistributionType::delta, id.get());
}

//---------------------------------------------------------------------------//
/*!
 * Add data for sampling a value from a normal distribution.
 */
OnedDistributionId
DistributionInserter::operator()(inp::NormalDistribution const& d)
{
    NormalDistributionRecord record;
    record.mean = d.mean;
    record.stddev = d.stddev;
    auto id = CollectionBuilder{&data_.normal}.push_back(record);
    return (*this)(OnedDistributionType::normal, id.get());
}

//---------------------------------------------------------------------------//
/*!
 * Add data for sampling a point from a 3D delta distribution.
 */
ThreedDistributionId DistributionInserter::operator()(
    inp::DeltaDistribution<Array<double, 3>> const& d)
{
    DeltaDistributionRecord<Array<real_type, 3>> record;
    record.value = array_cast<real_type>(d.value);
    auto id = CollectionBuilder{&data_.delta_real3}.push_back(record);
    return (*this)(ThreedDistributionType::delta, id.get());
}

//---------------------------------------------------------------------------//
/*!
 * Add data for sampling a value from an isotropic distribution.
 */
ThreedDistributionId
DistributionInserter::operator()(inp::IsotropicDistribution const&)
{
    IsotropicDistributionRecord record;
    auto id = CollectionBuilder{&data_.isotropic}.push_back(record);
    return (*this)(ThreedDistributionType::isotropic, id.get());
}

//---------------------------------------------------------------------------//
/*!
 * Add data for sampling a value from a uniform box distribution.
 */
ThreedDistributionId
DistributionInserter::operator()(inp::UniformBoxDistribution const& d)
{
    UniformBoxDistributionRecord record;
    record.upper = array_cast<real_type>(d.upper);
    record.lower = array_cast<real_type>(d.lower);
    auto id = CollectionBuilder{&data_.uniform_box}.push_back(record);
    return (*this)(ThreedDistributionType::uniform_box, id.get());
}

//---------------------------------------------------------------------------//
/*!
 * Add a 1D distribution.
 */
OnedDistributionId
DistributionInserter::operator()(OnedDistributionType type, size_type idx)
{
    CELER_EXPECT(idx <= data_.oned_indices.size());

    auto result = CollectionBuilder{&data_.oned_types}.push_back(type);
    CollectionBuilder{&data_.oned_indices}.push_back(idx);

    CELER_ENSURE(data_.oned_indices.size() == data_.oned_types.size());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Add a 3D distribution.
 */
ThreedDistributionId
DistributionInserter::operator()(ThreedDistributionType type, size_type idx)
{
    CELER_EXPECT(idx <= data_.threed_indices.size());

    auto result = CollectionBuilder{&data_.threed_types}.push_back(type);
    CollectionBuilder{&data_.threed_indices}.push_back(idx);

    CELER_ENSURE(data_.threed_indices.size() == data_.threed_types.size());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

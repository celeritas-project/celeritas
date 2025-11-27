//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/DistributionInserter.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/inp/Distributions.hh"
#include "corecel/random/data/DistributionData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Help construct data for sampling from user-specified distributions.
 */
class DistributionInserter
{
  public:
    // Construct with a reference to mutable host data
    explicit DistributionInserter(HostVal<DistributionParamsData>&);

    // Add 1D distribution data
    OnedDistributionId operator()(inp::DeltaDistribution<double> const&);
    OnedDistributionId operator()(inp::NormalDistribution const&);

    // Add 3D distribution data
    ThreedDistributionId
    operator()(inp::DeltaDistribution<Array<double, 3>> const&);
    ThreedDistributionId operator()(inp::IsotropicDistribution const&);
    ThreedDistributionId operator()(inp::UniformBoxDistribution const&);

  private:
    HostVal<DistributionParamsData>& data_;

    OnedDistributionId operator()(OnedDistributionType type, size_type idx);
    ThreedDistributionId operator()(ThreedDistributionType type, size_type idx);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGeneratorOptions.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorOptions.hh"

#include "celeritas/random/distribution/DeltaDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a distribution for sampling the energy.
 */
std::function<real_type(PrimaryGeneratorEngine&)>
make_energy_sampler(DistributionOptions options)
{
    CELER_EXPECT(options);

    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            CELER_ASSERT(p.size() == 1);
            return DeltaDistribution<real_type>(p[0]);
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return a distribution for sampling the position.
 */
std::function<Real3(PrimaryGeneratorEngine&)>
make_position_sampler(DistributionOptions options)
{
    CELER_EXPECT(options);

    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            CELER_ASSERT(p.size() == 3);
            return DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
        case DistributionSelection::box:
            CELER_ASSERT(p.size() == 6);
            return UniformBoxDistribution<real_type>(Real3{p[0], p[1], p[2]},
                                                     Real3{p[3], p[4], p[5]});
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return a distribution for sampling the direction.
 */
std::function<Real3(PrimaryGeneratorEngine&)>
make_direction_sampler(DistributionOptions options)
{
    CELER_EXPECT(options);

    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            CELER_ASSERT(p.size() == 3);
            return DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
        case DistributionSelection::isotropic:
            CELER_ASSERT(p.empty());
            return IsotropicDistribution<real_type>();
        default:
            CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

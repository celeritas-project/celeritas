//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGeneratorOptions.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorOptions.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "celeritas/random/distribution/DeltaDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Validate the number of parameters.
 */
void check_params_size(char const* sampler,
                       std::size_t dimension,
                       DistributionOptions options)
{
    CELER_EXPECT(dimension > 0);
    std::size_t required_params = 0;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            required_params = dimension;
            break;
        case DistributionSelection::isotropic:
            required_params = 0;
            break;
        case DistributionSelection::box:
            required_params = 2 * dimension;
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    CELER_VALIDATE(options.params.size() == required_params,
                   << sampler << " input parameters have "
                   << options.params.size() << " elements but the '"
                   << to_cstring(options.distribution)
                   << "' distribution needs exactly " << required_params);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to the distribution type.
 */
char const* to_cstring(DistributionSelection value)
{
    static EnumStringMapper<DistributionSelection> const to_cstring_impl{
        "delta",
        "isotropic",
        "box",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Return a distribution for sampling the energy.
 */
std::function<real_type(PrimaryGeneratorEngine&)>
make_energy_sampler(DistributionOptions options)
{
    CELER_EXPECT(options);

    char const sampler_name[] = "energy";
    check_params_size(sampler_name, 1, options);
    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            return DeltaDistribution<real_type>(p[0]);
        default:
            CELER_VALIDATE(false,
                           << "invalid distribution type '"
                           << to_cstring(options.distribution) << "' for "
                           << sampler_name);
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

    char const sampler_name[] = "position";
    check_params_size(sampler_name, 3, options);
    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            return DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
        case DistributionSelection::box:
            return UniformBoxDistribution<real_type>(Real3{p[0], p[1], p[2]},
                                                     Real3{p[3], p[4], p[5]});
        default:
            CELER_VALIDATE(false,
                           << "invalid distribution type '"
                           << to_cstring(options.distribution) << "' for "
                           << sampler_name);
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

    char const sampler_name[] = "direction";
    check_params_size(sampler_name, 3, options);
    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            return DeltaDistribution<Real3>(Real3{p[0], p[1], p[2]});
        case DistributionSelection::isotropic:
            return IsotropicDistribution<real_type>();
        default:
            CELER_VALIDATE(false,
                           << "invalid distribution type '"
                           << to_cstring(options.distribution) << "' for "
                           << sampler_name);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

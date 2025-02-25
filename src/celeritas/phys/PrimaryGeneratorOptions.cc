//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/PrimaryGeneratorOptions.cc
//---------------------------------------------------------------------------//
#include "PrimaryGeneratorOptions.hh"

#include "corecel/io/EnumStringMapper.hh"
#include "celeritas/inp/Events.hh"

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
// Helper: Convert energy distribution to inp::EnergyDistribution
inp::EnergyDistribution inp_from_energy(DistributionOptions const& options)
{
    char const sampler_name[] = "energy";
    check_params_size(sampler_name, 1, options);
    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            return inp::Monoenergetic{units::MevEnergy{p[0]}};
        default:
            CELER_VALIDATE(false,
                           << "invalid distribution type '"
                           << to_cstring(options.distribution) << "' for "
                           << sampler_name);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
// Convert position distribution to inp::ShapeDistribution
inp::ShapeDistribution inp_from_position(DistributionOptions const& options)
{
    char const sampler_name[] = "position";
    check_params_size(sampler_name, 3, options);
    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            return inp::PointShape{Real3{p[0], p[1], p[2]}};
        case DistributionSelection::box:
            return inp::UniformBoxShape{Real3{p[0], p[1], p[2]},
                                        Real3{p[3], p[4], p[5]}};
        default:
            CELER_VALIDATE(false,
                           << "invalid distribution type '"
                           << to_cstring(options.distribution) << "' for "
                           << sampler_name);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
// Helper: Convert direction distribution to inp::AngleDistribution
inp::AngleDistribution inp_from_direction(DistributionOptions const& options)
{
    char const sampler_name[] = "direction";
    check_params_size(sampler_name, 3, options);
    auto const& p = options.params;
    switch (options.distribution)
    {
        case DistributionSelection::delta:
            return inp::MonodirectionalAngle{Real3{p[0], p[1], p[2]}};
        case DistributionSelection::isotropic:
            return inp::IsotropicAngle{};
        default:
            CELER_VALIDATE(false,
                           << "invalid distribution type '"
                           << to_cstring(options.distribution) << "' for "
                           << sampler_name);
    }
    CELER_ASSERT_UNREACHABLE();
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
 * Convert PrimaryGeneratorOptions to inp::PrimaryGenerator.
 */
inp::PrimaryGenerator to_input(PrimaryGeneratorOptions const& pgo)
{
    CELER_VALIDATE(pgo,
                   << "Invalid PrimaryGeneratorOptions: "
                   << "ensure all distributions and parameters are correctly "
                      "set.");

    inp::PrimaryGenerator result;

    // RNG seed
    result.seed = pgo.seed;

    // PDG numbers
    result.pdg = pgo.pdg;

    // Number of events and primaries per event
    result.num_events = pgo.num_events;
    result.primaries_per_event = pgo.primaries_per_event;

    // Distributions
    result.shape = inp_from_position(pgo.position);
    result.angle = inp_from_direction(pgo.direction);
    result.energy = inp_from_energy(pgo.energy);

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

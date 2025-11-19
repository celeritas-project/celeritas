//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/PrimaryGeneratorData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/random/distribution/NormalDistribution.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "geocel/random/UniformBoxDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/inp/Events.hh"

#include "../Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
//! Energy distribution types for optical photon primaries
enum class EnergyDistribution
{
    monoenergetic,
    gaussian,
    size_
};

//---------------------------------------------------------------------------//
//! Angular distribution types for optical photon primaries
enum class AngleDistribution
{
    monodirectional,
    isotropic,
    size_
};

//---------------------------------------------------------------------------//
//! Spatial distribution types for optical photon primaries
enum class ShapeDistribution
{
    point,
    uniform_box,
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical photon energy.
 */
struct EnergySampler
{
    union DistributionData
    {
        inp::MonoenergeticDistribution monoenergetic;
        inp::GaussianDistribution gaussian;

        //! Default constructor (uninitialized state)
        CELER_FORCEINLINE_FUNCTION DistributionData() {}
    };

    EnergyDistribution type{EnergyDistribution::size_};
    DistributionData data;

    // Sample an energy
    template<class Generator>
    inline CELER_FUNCTION units::MevEnergy operator()(Generator&) const;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return type != EnergyDistribution::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical photon direction.
 */
struct DirectionSampler
{
    union DistributionData
    {
        inp::MonodirectionalDistribution monodirectional;
        inp::IsotropicDistribution isotropic;

        //! Default constructor (uninitialized state)
        CELER_FORCEINLINE_FUNCTION DistributionData() {}
    };

    AngleDistribution type{AngleDistribution::size_};
    DistributionData data;

    // Sample a direction
    template<class Generator>
    inline CELER_FUNCTION Real3 operator()(Generator&) const;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return type != AngleDistribution::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical photon position.
 */
struct PositionSampler
{
    union DistributionData
    {
        inp::PointDistribution point;
        inp::UniformBoxDistribution uniform_box;

        //! Default constructor (uninitialized state)
        CELER_FORCEINLINE_FUNCTION DistributionData() {}
    };

    ShapeDistribution type{ShapeDistribution::size_};
    DistributionData data;

    // Sample a position
    template<class Generator>
    inline CELER_FUNCTION Real3 operator()(Generator&) const;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return type != ShapeDistribution::size_;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data for sampling optical photons from user-configurable distributions.
 */
struct PrimaryDistributionData
{
    size_type num_photons{};
    EnergySampler sample_energy;
    DirectionSampler sample_direction;
    PositionSampler sample_position;

    //! Check whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return num_photons > 0 && sample_energy && sample_direction
               && sample_position;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Sample an energy from the specified distribution.
 */
template<class Generator>
CELER_FUNCTION units::MevEnergy EnergySampler::operator()(Generator& rng) const
{
    CELER_EXPECT(*this);
    switch (type)
    {
        case EnergyDistribution::monoenergetic:
            return data.monoenergetic.energy;
        case EnergyDistribution::gaussian:
            return units::MevEnergy(NormalDistribution(
                data.gaussian.mean, data.gaussian.stddev)(rng));
        case EnergyDistribution::size_:
            CELER_ASSERT_UNREACHABLE();
    }
    CELER_ASSERT_UNREACHABLE();
};

//---------------------------------------------------------------------------//
/*!
 * Sample a direction from the specified distribution.
 */
template<class Generator>
CELER_FUNCTION Real3 DirectionSampler::operator()(Generator& rng) const
{
    CELER_EXPECT(*this);
    switch (type)
    {
        case AngleDistribution::monodirectional:
            return data.monodirectional.dir;
        case AngleDistribution::isotropic:
            return IsotropicDistribution()(rng);
        case AngleDistribution::size_:
            CELER_ASSERT_UNREACHABLE();
    }
    CELER_ASSERT_UNREACHABLE();
};

//---------------------------------------------------------------------------//
/*!
 * Sample a position from the specified distribution.
 */
template<class Generator>
CELER_FUNCTION Real3 PositionSampler::operator()(Generator& rng) const
{
    CELER_EXPECT(*this);
    switch (type)
    {
        case ShapeDistribution::point:
            return data.point.pos;
        case ShapeDistribution::uniform_box:
            return UniformBoxDistribution(data.uniform_box.lower,
                                          data.uniform_box.upper)(rng);
        case ShapeDistribution::size_:
            CELER_ASSERT_UNREACHABLE();
    }
    CELER_ASSERT_UNREACHABLE();
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas

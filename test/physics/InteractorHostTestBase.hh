//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InteractorHostTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>
#include <vector>
#include "gtest/Test.hh"
#include "base/Array.hh"
#include "base/ArrayIO.hh"
#include "base/Span.hh"
#include "base/StackAllocatorPointers.hh"
#include "base/Types.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleStatePointers.hh"
#include "physics/base/Secondary.hh"
#include "base/HostStackAllocatorStore.hh"

namespace celeritas
{
template<class T>
class StackAllocatorView;
class ParticleTrackView;
struct Interaction;
struct Secondary;
} // namespace celeritas

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for a host-side Interactor.
 *
 * This class initializes host versions of some of the common inputs to an
 * Interactor. It \b cannot be used for testing device instantiations.
 */
class InteractorHostTestBase : public celeritas::Test
{
  public:
    //@{
    //! Type aliases
    using RandomEngine = std::mt19937;

    using real_type              = celeritas::real_type;
    using PDGNumber              = celeritas::PDGNumber;
    using Interaction            = celeritas::Interaction;
    using ParticleParams         = celeritas::ParticleParams;
    using ParticleTrackView      = celeritas::ParticleTrackView;
    using Real3                  = celeritas::Real3;
    using Secondary              = celeritas::Secondary;
    using SecondaryAllocatorView = celeritas::StackAllocatorView<Secondary>;
    using constSpanSecondaries   = celeritas::span<const Secondary>;

    using HostSecondaryStore = HostStackAllocatorStore<Secondary>;
    //@}

  public:
    //@{
    //! Initialize and destroy
    InteractorHostTestBase();
    ~InteractorHostTestBase();
    //@}

    //@{
    //! Set and get particle params
    void set_particle_params(const ParticleParams::VecAnnotatedDefs& defs);
    const ParticleParams& particle_params() const;
    //@}

    //@{
    //! Incident particle properties and access
    void                     set_inc_particle(PDGNumber n, real_type energy);
    void                     set_inc_direction(const Real3& dir);
    const Real3&             direction() const { return inc_direction_; }
    const ParticleTrackView& particle_track() const
    {
        REQUIRE(pt_view_);
        return *pt_view_;
    }
    //@}

    //@{
    //! Secondary stack storage and access
    void                             resize_secondaries(int count);
    const HostSecondaryStore& secondaries() const { return secondaries_; }
    SecondaryAllocatorView& secondary_allocator()
    {
        REQUIRE(sa_view_);
        return *sa_view_;
    }
    //@}

    //@{
    //! Random number generator
    RandomEngine& rng() { return rng_; }
    //@}

    // Check for momentum and energy conservation
    void check_conservation(const Interaction& interaction) const;

  private:
    std::shared_ptr<ParticleParams> particle_params_;
    RandomEngine                    rng_;

    celeritas::ParticleTrackState     particle_state_;
    celeritas::ParticleParamsPointers pp_pointers_;
    celeritas::ParticleStatePointers  ps_pointers_;
    Real3                     inc_direction_ = {0, 0, 1};
    HostSecondaryStore                secondaries_;

    // Views
    std::shared_ptr<ParticleTrackView>      pt_view_;
    std::shared_ptr<SecondaryAllocatorView> sa_view_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas_test

//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/mat/MaterialData.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleParams.hh"

// Test helpers
#include "corecel/random/DiagnosticRngEngine.hh"

#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleTrackView;
class MaterialTrackView;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for Neutron physics models.
 */
class NeutronTestBase : public Test
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using MevEnergy = units::MevEnergy;
    using Action = Interaction::Action;
    //!@}

  public:
    //!@{
    //! Initialize and destroy
    NeutronTestBase();
    ~NeutronTestBase();
    //!@}

    //!@{
    //! Set and get material properties
    void set_material_params(MaterialParams::Input inp);
    std::shared_ptr<MaterialParams const> const& material_params() const
    {
        CELER_EXPECT(material_params_);
        return material_params_;
    }
    //!@}

    //!@{
    //! Set and get particle params
    void set_particle_params(ParticleParams::Input inp);
    std::shared_ptr<ParticleParams const> const& particle_params() const
    {
        CELER_EXPECT(particle_params_);
        return particle_params_;
    }
    //!@}

    //!@{
    //! Material properties
    void set_material(std::string const& name);
    MaterialTrackView& material_track()
    {
        CELER_EXPECT(mt_view_);
        return *mt_view_;
    }
    //!@}

    //!@{
    //! Incident particle properties and access
    void set_inc_particle(PDGNumber n, MevEnergy energy);
    void set_inc_direction(Real3 const& dir);
    Real3 const& direction() const { return inc_direction_; }
    ParticleTrackView const& particle_track() const
    {
        CELER_EXPECT(pt_view_);
        return *pt_view_;
    }
    //!@}

    //!@{
    //! Get random number generator with clean counter
    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }
    //!@}

  private:
    template<template<Ownership, MemSpace> class S>
    using StateStore = CollectionStateStore<S, MemSpace::host>;

    std::shared_ptr<MaterialParams const> material_params_;
    std::shared_ptr<ParticleParams const> particle_params_;
    RandomEngine rng_;

    StateStore<MaterialStateData> ms_;
    StateStore<ParticleStateData> ps_;

    Real3 inc_direction_ = {0, 0, 1};

    // Views
    std::shared_ptr<MaterialTrackView> mt_view_;
    std::shared_ptr<ParticleTrackView> pt_view_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
